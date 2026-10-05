"""Two-period comparison of one line: where did vehicles start (or stop) lingering?

``compare(a, b)`` takes two :class:`~microsegments.metrics.Analysis` of the same line, period A
("before") and period B ("after"), and compares them segment by segment and hour by hour.

Matching. Segments are matched on ``seg_key``; when the two periods were built from GTFS versions
whose link geometry differs slightly (new ``link_key`` hash for the same stop pair), links with the
same stop pair (``link_key`` up to ``~``), a length within max(5 %, 20 m) and the same number of
segments are matched segment by segment. Segments present in one period only are listed in
``Comparison.unmatched`` and left out of the diff. The layout (order, positions) is period B's
display pattern.

Per bin (segment x hour, plus the am / pm / day / evening bands), each period on its own included
days and its own references:

* ``obs_per_passage`` A / B (O), the evening reference and the excess over it, the excess over the
  line level (``metrics.line_level``);
* ``delta`` = O_B - O_A, observations per vehicle (x ``tick_s`` ≈ seconds per vehicle). Being per
  passage, it does not move with the frequency; the excess delta (``delta_excess``) also removes a
  change of the evening level;
* day bootstrap CI of the delta: each period's days are resampled independently (Rao-Wu, weekday
  stratified, as in ``hotspots``), draw b of the delta = draw b of B - draw b of A;
* ``significant``: |delta| beyond ``theta_bin_s`` seconds per vehicle (default 0) with ``level``
  confidence (CI bound) and Benjamini-Hochberg at ``q`` over all bins (one-sided z p-values of
  |delta| - theta from the bootstrap SE). A 30 m segment rarely moves by 2 s on its own with a
  tight CI; the size threshold ``theta_s`` applies to the stretch (below);
* passages per hour A / B (the frequency) and its relative change.

Stretches. Significant bins count when they form a run of ≥ 2 adjacent hours with the same sign,
or a whole am / pm / day band; such segments merge along the line with a ``gap``-bin tolerance,
never across a stop / running zone border nor across a sign change. A stretch is kept when its summed
delta at its strongest hour moves by more than ``theta_s`` seconds per vehicle with ``level``
confidence (bootstrap CI of the sum). Each stretch carries its delta
at its strongest hour (with CI), the change in vehicle time per day (Σ over its hours and segments
of delta x passages per hour in B x tick_s, seconds of vehicle time per day), the frequencies and
``frequency_changed`` (|B / A - 1| > ``freq_change``): there a timetable change may explain part of
the difference (bunching, dwell), so it should not be read as an infrastructure effect alone.
Stretches are ranked per direction by |vehicle time change per day|. Fewer than ``MIN_DAYS``
included days in a period: bins are still computed, no stretch (``params["status"]``). Weekday
strata are used only when every weekday has at least two days in the period.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from .hotspots import _bh, _erfc, bootstrap_weights, nanquantile0
from .metrics import BAND_HOUR, BANDS, Analysis, DirData

COMPARE_BANDS = ("am", "pm", "day", "evening")
MIN_DAYS = 5          # per period, as hotspots.MIN_HOTSPOT_DAYS: fewer days, no stretch is reported
STRETCH_BANDS = ("am", "pm", "day")


@dataclass
class _Dir:
    """Aligned segments of one direction: B display order, A match (-1 if none)."""
    dirid: int
    dA: DirData
    dB: DirData
    kB: np.ndarray          # (J,) K indices in B (display order)
    kA: np.ndarray          # (J,) K indices in A or -1
    how: list[str]          # "seg_key" | "stop_pair" | ""


@dataclass
class Comparison:
    a: Analysis
    b: Analysis
    bins: pl.DataFrame                 # one row per direction x B segment x column (hours, then bands)
    stretches: pl.DataFrame            # ranked changed stretches
    unmatched: pl.DataFrame            # direction_id, period ("a" | "b"), seg_key, link_key
    frequency: pl.DataFrame            # direction_id, hour, passages per hour A / B (median over segments)
    params: dict = field(default_factory=dict)
    _dirs: dict[int, _Dir] = field(default_factory=dict, repr=False)

    def to_contract(self, names: dict[str, str] | None = None) -> dict[str, Any]:
        """The additive ``compare`` section of the JSON contract (see ``contract.py``)."""
        return compare_contract(self, names)


# ------------------------------------------------------------------------------------ matching
def _pair(key: str) -> str:
    return key.split("~")[0]


def _link_len(dd: DirData, li: int) -> float:
    return float(dd.seg_len[dd.seg_link == li].sum())


def _align(dA: DirData, dB: DirData, dirid: int, len_rel: float = 0.05, len_m: float = 20.0) -> _Dir:
    kB = dB.pattern_segs[dB.display_uid]
    posA = {k: i for i, k in enumerate(dA.seg_keys.tolist())}
    kA = np.array([posA.get(k, -1) for k in dB.seg_keys[kB].tolist()], dtype=np.int64)
    how = ["seg_key" if k >= 0 else "" for k in kA]
    if (kA < 0).any():
        # same stop pair, similar length, same number of segments -> match in order
        used_a = set(kA[kA >= 0].tolist())
        a_links = {}
        for li, lk in enumerate(dA.link_keys.tolist()):
            segs = np.flatnonzero(dA.seg_link == li)
            if len(segs) and not (set(segs.tolist()) & used_a):
                a_links.setdefault(_pair(lk), []).append((li, segs))
        linkB = dB.seg_link[kB]
        for li in np.unique(linkB[kA < 0]):
            js = np.flatnonzero(linkB == li)
            if (kA[js] >= 0).any():
                continue
            cands = a_links.get(_pair(dB.link_keys[li]), [])
            LB = _link_len(dB, li)
            for c, (la, segs) in enumerate(cands):
                LA = _link_len(dA, la)
                if len(segs) == len(js) and abs(LA - LB) <= max(len_rel * max(LA, LB), len_m):
                    kA[js] = segs
                    for j in js:
                        how[j] = "stop_pair"
                    cands.pop(c)
                    break
    return _Dir(dirid=dirid, dA=dA, dB=dB, kB=kB, kA=kA, how=how)


# ------------------------------------------------------------------------------------ statistics
def _cols(an: Analysis, dd: DirData, hours: list[int]) -> np.ndarray:
    labels = list(hours) + [BAND_HOUR[b] for b in COMPARE_BANDS]
    return dd.col_index(labels)


def _pick(e: dict, name: str, ci: np.ndarray, k: np.ndarray) -> np.ndarray:
    return e[name][:, ci][:, :, k]


def _pph(pas: np.ndarray, ncell: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Passages per hour (per day): hour columns pas / n_cells; band columns from their hours."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ncell > 0, pas / np.where(ncell > 0, ncell, 1), np.nan)
        for b, (h0, h1) in BANDS.items():
            c = np.flatnonzero(labels == BAND_HOUR[b])
            hs = np.flatnonzero((labels >= h0) & (labels < h1))
            if len(c) and len(hs):
                P, N = pas[..., hs, :].sum(-2), ncell[..., hs, :].sum(-2)
                out[..., c[0], :] = np.where(N > 0, P / np.where(N > 0, N, 1), np.nan)
    return out


def _strat(dow: np.ndarray, stratified: bool) -> bool:
    """Weekday strata only when every weekday present has >= 2 days: a one-day stratum keeps weight 1
    in every draw (no variance), which would make a short period look exact."""
    if not stratified:
        return False
    _, n = np.unique(dow, return_counts=True)
    return bool(len(n) and n.min() >= 2)


def _draws(an: Analysis, dirid: int, W: np.ndarray, ci: np.ndarray, k: np.ndarray, chunk: int) -> np.ndarray:
    out = np.empty((W.shape[0], len(ci), len(k)), np.float32)
    for a in range(0, W.shape[0], chunk):
        e = an.estimate(dirid, W[a:a + chunk], coverage=False)
        out[a:a + chunk] = _pick(e, "obs_per_passage", ci, k)
    return out


def compare(a: Analysis, b: Analysis, *, B: int = 200, seed: int = 0, theta_s: float = 2.0, theta_bin_s: float = 0.0,
            q: float = 0.1,
            level: float = 0.95, freq_change: float = 0.2, gap: int = 1, stratified: bool = True,
            links: pl.DataFrame | None = None, chunk: int = 50) -> Comparison:
    """Compare period A (``a``, before) with period B (``b``, after) for the same line. See the module
    docstring. ``links`` (schema.LINK, e.g. ``network.links``) names the stops of the stretches."""
    tick = b.params.tick_s
    theta = theta_bin_s / tick
    hours = [h for h in b.hours if h in set(a.hours)]
    rng_a, rng_b = np.random.default_rng([seed, 0]), np.random.default_rng([seed, 1])
    names: dict[str, tuple] = {}
    if links is not None:
        for r in links.select("link_key", "from_name", "to_name").iter_rows():
            names.setdefault(r[0], (r[1], r[2]))

    bin_frames, str_rows, unm, freq_rows, dirs = [], [], [], [], {}
    status = "computed"
    for dirid in sorted(set(a.dirs) & set(b.dirs)):
        dA, dB = a.dirs[dirid], b.dirs[dirid]
        if len(dA.days) == 0 or len(dB.days) == 0:
            continue
        if min(len(dA.days), len(dB.days)) < MIN_DAYS:
            status = f"too_few_days: {len(dA.days)} / {len(dB.days)} included days < {MIN_DAYS}"
        al = _align(dA, dB, dirid)
        dirs[dirid] = al
        m = al.kA >= 0
        # unmatched segments, both ways (A: those of A's display pattern)
        for j in np.flatnonzero(~m):
            k = al.kB[j]
            unm.append((dirid, "b", dB.seg_keys[k], dB.link_keys[dB.seg_link[k]]))
        matched_a = set(al.kA[m].tolist())
        for k in dA.pattern_segs[dA.display_uid]:
            if int(k) not in matched_a:
                unm.append((dirid, "a", dA.seg_keys[k], dA.link_keys[dA.seg_link[k]]))
        ciA, ciB = _cols(a, dA, hours), _cols(b, dB, hours)
        labels = dB.cols[ciB]
        J = len(al.kB)
        kA = np.where(m, al.kA, 0)
        eA, eB = a.estimate(dirid), b.estimate(dirid)
        nanA = lambda x: np.where(m[None, :], x, np.nan)  # noqa: E731
        oa, ob = nanA(_pick(eA, "obs_per_passage", ciA, kA)[0]), _pick(eB, "obs_per_passage", ciB, al.kB)[0]
        xa, xb = nanA(_pick(eA, "excess_per_passage", ciA, kA)[0]), _pick(eB, "excess_per_passage", ciB, al.kB)[0]
        la, lb = nanA(_pick(eA, "excess_line_per_passage", ciA, kA)[0]), _pick(eB, "excess_line_per_passage", ciB, al.kB)[0]
        ra, rb = np.where(m, eA["ref"][0][kA], np.nan), eB["ref"][0][al.kB]
        fa = nanA(_pph(_pick(eA, "passages", ciA, kA)[0], _pick(eA, "n_cells", ciA, kA)[0], labels))
        fb = _pph(_pick(eB, "passages", ciB, al.kB)[0], _pick(eB, "n_cells", ciB, al.kB)[0], labels)
        delta = ob - oa
        dx = xb - xa
        # bootstrap: independent day resampling of each period, paired by draw index
        WA = bootstrap_weights(dA.dow, B, rng_a, _strat(dA.dow, stratified))
        WB = bootstrap_weights(dB.dow, B, rng_b, _strat(dB.dow, stratified))
        draws = _draws(b, dirid, WB, ciB, al.kB, chunk) - _draws(a, dirid, WA, ciA, kA, chunk)
        draws[:, :, ~m] = np.nan
        alpha = 1 - level
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = nanquantile0(draws, [alpha / 2, 1 - alpha / 2])
            se = np.nanstd(draws, axis=0, ddof=1)
            z = (np.abs(delta) - theta) / se
            z = np.where(se > 0, z, np.where(np.abs(delta) > theta, np.inf, -np.inf))
            p = 0.5 * _erfc(z / math.sqrt(2)).astype(float)
        p[~np.isfinite(delta)] = np.nan
        reject = _bh(p, q)
        sig = reject & ((lo > theta) | (hi < -theta))
        sign = np.where(sig, np.sign(delta), 0).astype(np.int8)
        with np.errstate(divide="ignore", invalid="ignore"):
            fchg = fb / fa - 1
        u = dB.display_uid
        C = len(labels)
        bin_frames.append(pl.DataFrame({
            "direction_id": np.full(J * C, dirid, np.int8),
            "pattern_uid": [u] * (J * C),
            "seg_key": np.repeat(dB.seg_keys[al.kB], C).tolist(),
            "seg_key_a": np.repeat(np.where(m, dA.seg_keys[kA], None), C).tolist(),
            "match": np.repeat(np.array(al.how, dtype=object), C).tolist(),
            "seg_idx": np.repeat(dB.pattern_seg_idx[u], C).astype(np.int32),
            "x0_m": np.repeat(dB.pattern_x0[u], C),
            "len_m": np.repeat(dB.seg_len[al.kB], C),
            "zone": np.repeat(dB.seg_zone[al.kB], C).tolist(),
            "link_key": np.repeat(dB.link_keys[dB.seg_link[al.kB]], C).tolist(),
            "hour": np.tile(labels, J).astype(np.int8),
            "obs_per_passage_a": oa.T.reshape(-1), "obs_per_passage_b": ob.T.reshape(-1),
            "ref_a": np.repeat(ra, C), "ref_b": np.repeat(rb, C),
            "excess_a": xa.T.reshape(-1), "excess_b": xb.T.reshape(-1),
            "excess_line_a": la.T.reshape(-1), "excess_line_b": lb.T.reshape(-1),
            "delta": delta.T.reshape(-1), "delta_s": (delta * tick).T.reshape(-1),
            "delta_excess": dx.T.reshape(-1),
            "ci_lo": lo.T.reshape(-1), "ci_hi": hi.T.reshape(-1), "se": se.T.reshape(-1), "p": p.T.reshape(-1),
            "significant": sig.T.reshape(-1), "sign": sign.T.reshape(-1),
            "passages_per_h_a": fa.T.reshape(-1), "passages_per_h_b": fb.T.reshape(-1),
            "freq_change": fchg.T.reshape(-1),
        }, nan_to_null=True))
        # direction-level frequency per hour (median over matched segments)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            ma, mb = np.nanmedian(fa, axis=1), np.nanmedian(fb, axis=1)
        for c, h in enumerate(labels):
            freq_rows.append((dirid, int(h), float(ma[c]), float(mb[c]),
                              float(mb[c] / ma[c] - 1) if ma[c] and np.isfinite(ma[c]) and ma[c] > 0 else float("nan")))
        if status == "computed":
            str_rows += _stretches(al, labels, sign, delta, draws, oa, ob, xa, xb, fa, fb, tick, alpha, gap,
                                   freq_change, names, len(dA.days), len(dB.days), theta_s / tick)

    bins = pl.concat(bin_frames) if bin_frames else pl.DataFrame()
    st = pl.DataFrame(str_rows, schema=STRETCH_SCHEMA, orient="row") if str_rows else pl.DataFrame(schema=STRETCH_SCHEMA)
    if st.height:
        st = (st.with_columns(pl.col("veh_time_s_per_day").abs().alias("_a"))
              .sort(["direction_id", "_a"], descending=[False, True]).drop("_a")
              .with_columns(pl.int_range(1, pl.len() + 1).over("direction_id").cast(pl.Int16).alias("rank")))
    unmatched = pl.DataFrame(unm, schema={"direction_id": pl.Int8, "period": pl.Utf8, "seg_key": pl.Utf8,
                                          "link_key": pl.Utf8}, orient="row")
    frequency = pl.DataFrame(freq_rows, schema={"direction_id": pl.Int8, "hour": pl.Int8, "passages_per_h_a": pl.Float64,
                                                "passages_per_h_b": pl.Float64, "freq_change": pl.Float64},
                             orient="row").fill_nan(None)
    params = {"B": B, "seed": seed, "theta_s": theta_s, "theta_bin_s": theta_bin_s, "q": q, "level": level, "freq_change": freq_change,
              "gap": gap, "tick_s": tick, "hours": hours, "status": status}
    return Comparison(a=a, b=b, bins=bins, stretches=st, unmatched=unmatched, frequency=frequency,
                      params=params, _dirs=dirs)


STRETCH_SCHEMA = {
    "direction_id": pl.Int8, "pattern_uid": pl.Utf8, "rank": pl.Int16, "sign": pl.Int8,
    "from_seg": pl.Int32, "to_seg": pl.Int32, "x0_m": pl.Float64, "x1_m": pl.Float64,
    "from_stop_name": pl.Utf8, "to_stop_name": pl.Utf8, "zone": pl.Utf8,
    "hours": pl.List(pl.Int8), "peak_hour": pl.Int8,
    "delta_per_passage": pl.Float32, "ci_lo": pl.Float32, "ci_hi": pl.Float32, "delta_s": pl.Float32,
    "obs_per_passage_a": pl.Float32, "obs_per_passage_b": pl.Float32,
    "excess_a": pl.Float32, "excess_b": pl.Float32,
    "veh_time_s_per_day": pl.Float32, "veh_time_s_per_day_all": pl.Float32,
    "passages_per_h_a": pl.Float32, "passages_per_h_b": pl.Float32, "freq_change": pl.Float32,
    "frequency_changed": pl.Boolean, "n_days_a": pl.Int32, "n_days_b": pl.Int32, "seg_keys": pl.List(pl.Utf8),
}


def _seg_hours(labels: np.ndarray, sign: np.ndarray, s: int) -> list[set[int]]:
    """Per segment, hours where the change of sign ``s`` passes the temporal rule."""
    hc = np.flatnonzero(labels >= 0)
    hrs = labels[hc]
    out = []
    for j in range(sign.shape[1]):
        c = sign[hc, j] == s
        keep: set[int] = set()
        for i in np.flatnonzero(c):
            prev = i > 0 and c[i - 1] and hrs[i - 1] == hrs[i] - 1
            nxt = i + 1 < len(hrs) and c[i + 1] and hrs[i + 1] == hrs[i] + 1
            if prev or nxt:
                keep.add(int(hrs[i]))
        for bnd in STRETCH_BANDS:
            bi = np.flatnonzero(labels == BAND_HOUR[bnd])
            if len(bi) and sign[bi[0], j] == s:
                h0, h1 = BANDS[bnd]
                keep |= {int(h) for h in hrs if h0 <= h < h1}
        out.append(keep)
    return out


def _stretches(al: _Dir, labels, sign, delta, draws, oa, ob, xa, xb, fa, fb, tick, alpha, gap, freq_tol,
               names, nda, ndb, theta_stretch) -> list[tuple]:
    dB = al.dB
    u = dB.display_uid
    zone = dB.seg_zone[al.kB]
    x0 = dB.pattern_x0[u]
    ln = dB.seg_len[al.kB]
    seg_idx = dB.pattern_seg_idx[u]
    hc = np.flatnonzero(labels >= 0)
    hrs = labels[hc]
    rows = []
    for s in (1, -1):
        hs = _seg_hours(labels, sign, s)
        groups: list[list] = []
        for j, h in enumerate(hs):
            if not h:
                continue
            if groups:
                g = groups[-1]
                if j - g[1] <= gap + 1 and all(zone[i] == zone[j] for i in range(g[1], j + 1)):
                    g[1] = j
                    g[2] |= h
                    continue
            groups.append([j, j, set(h)])
        for j0, j1, hset in groups:
            sl = slice(j0, j1 + 1)
            hidx = np.array([hc[np.flatnonzero(hrs == h)[0]] for h in sorted(hset)])
            sx = np.nansum(delta[hidx][:, sl], axis=1)
            pk = int(np.nanargmax(s * sx))
            pcol = hidx[pk]
            val = float(sx[pk])
            dr = np.nansum(draws[:, pcol, sl], axis=1)
            dr = dr[np.isfinite(dr)]
            lo, hi = (np.quantile(dr, [alpha / 2, 1 - alpha / 2]) if len(dr) else (np.nan, np.nan))
            if not ((s > 0 and lo > theta_stretch) or (s < 0 and hi < -theta_stretch)):
                continue                       # the stretch as a whole must move by more than theta_s
            vt = float(np.nansum(delta[hidx][:, sl] * fb[hidx][:, sl]) * tick)
            vt_all = float(np.nansum(delta[hc][:, sl] * fb[hc][:, sl]) * tick)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pa = float(np.nanmean(np.nanmean(fa[hidx][:, sl], axis=0)))
                pb = float(np.nanmean(np.nanmean(fb[hidx][:, sl], axis=0)))
            fc = pb / pa - 1 if pa > 0 and np.isfinite(pa) and np.isfinite(pb) else float("nan")
            lk0 = dB.link_keys[dB.seg_link[al.kB[j0]]]
            lk1 = dB.link_keys[dB.seg_link[al.kB[j1]]]
            rows.append((al.dirid, u, 0, s, int(seg_idx[j0]), int(seg_idx[j1]), float(x0[j0]), float(x0[j1] + ln[j1]),
                         names.get(lk0, (lk0, None))[0], names.get(lk1, (None, lk1))[1], str(zone[j0]),
                         sorted(int(h) for h in hset), int(labels[pcol]), val, float(lo), float(hi), val * tick,
                         float(np.nansum(oa[pcol, sl])), float(np.nansum(ob[pcol, sl])),
                         float(np.nansum(xa[pcol, sl])), float(np.nansum(xb[pcol, sl])), vt, vt_all, pa, pb, fc,
                         bool(np.isfinite(fc) and abs(fc) > freq_tol), nda, ndb, dB.seg_keys[al.kB[sl]].tolist()))
    return rows


# ------------------------------------------------------------------------------------ contract
def _period(an: Analysis) -> dict:
    return {"first": min(an.days).isoformat() if an.days else None,
            "last": max(an.days).isoformat() if an.days else None,
            "days": len(an.days), "per_dow": list(an.per_dow),
            "excluded": [{"date": d.isoformat(), "reason": r} for d, r in an.excluded_days.iter_rows()]}


def compare_contract(cmp: Comparison, names: dict[str, str] | None = None) -> dict[str, Any]:
    from .report import frame_records
    names = names or {}
    hours = cmp.params["hours"]
    cols: list = list(hours) + list(COMPARE_BANDS)
    labels = list(hours) + [BAND_HOUR[b] for b in COMPARE_BANDS]

    def r(v, nd=3):
        return None if v is None or not math.isfinite(v) else round(float(v), nd)
    dirs = []
    for dirid, al in cmp._dirs.items():
        g = cmp.bins.filter(pl.col("direction_id") == dirid)
        J, C = len(al.kB), len(labels)
        def mat(col, nd=3):
            a = g[col].to_numpy().reshape(J, C).T if g.height else np.zeros((C, J))
            return [[r(v, nd) for v in row] for row in a.tolist()]
        sg = g["sign"].to_numpy().reshape(J, C).T.astype(int).tolist() if g.height else []
        n_only_a = cmp.unmatched.filter((pl.col("direction_id") == dirid) & (pl.col("period") == "a")).height
        dirs.append({
            "dir": int(dirid), "pattern_uid": al.dB.display_uid,
            "seg_key": al.dB.seg_keys[al.kB].tolist(),
            "matched": [int(k >= 0) for k in al.kA.tolist()],
            "only_a": n_only_a, "only_b": int((al.kA < 0).sum()),
            "oa": mat("obs_per_passage_a"), "ob": mat("obs_per_passage_b"),
            "xa": mat("excess_a"), "xb": mat("excess_b"),
            "d": mat("delta"), "lo": mat("ci_lo"), "hi": mat("ci_hi"), "sig": sg,
            "fa": mat("passages_per_h_a", 2), "fb": mat("passages_per_h_b", 2),
        })
    ch = []
    if cmp.stretches.height:
        for rec in frame_records(cmp.stretches.drop("seg_keys")):
            rec["dir"] = rec["direction_id"]
            for k in ("from_stop_name", "to_stop_name"):
                if rec.get(k) in names:
                    rec[k] = names[rec[k]]
            ch.append(rec)
    p = dict(cmp.params)
    p.pop("hours", None)
    return {"a": _period(cmp.a), "b": _period(cmp.b), "params": p, "cols": cols, "dirs": dirs, "changes": ch}


# ------------------------------------------------------------------------------------ end to end
def _range(s: str) -> tuple[dt.date, dt.date]:
    a, b = s.split("..")
    return dt.date.fromisoformat(a), dt.date.fromisoformat(b)


@dataclass
class CompareResult:
    cfg: Any
    prepared: Any
    segments: pl.DataFrame
    a: Analysis
    b: Analysis
    comparison: Comparison
    mode: str | None = None
    hotspots_b: pl.DataFrame | None = None      # period B hotspots (the page's single-period view)

    def contract(self, *, title: str | None = None, source: str | None = None) -> dict:
        """Period B's page contract plus the ``compare`` section."""
        from .report import name_map, to_contract
        names = name_map(getattr(self.cfg.report, "names", None), self.prepared.network)
        c = to_contract(self.b, self.prepared.network, self.segments, self.hotspots_b, mode=self.mode, names=names,
                        title=title, source=source)
        c["compare"] = self.comparison.to_contract(names)
        return c

    def save(self, out_dir: str | Path, *, html: bool = True, lang: str = "fr", title: str | None = None,
             source: str | None = None) -> dict[str, Path]:
        import json
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name, df in (("compare_bins.parquet", self.comparison.bins),
                         ("compare_changes.parquet", self.comparison.stretches),
                         ("compare_unmatched.parquet", self.comparison.unmatched),
                         ("compare_frequency.parquet", self.comparison.frequency)):
            df.write_parquet(out / name)
            paths[name] = out / name
        c = self.contract(title=title, source=source)
        p = out / "compare.json"
        p.write_text(json.dumps(c, separators=(",", ":"), ensure_ascii=False, default=str))
        paths["compare.json"] = p
        if html:
            from .html import export
            paths["compare.html"] = export(c, out / "compare.html", lang=lang, title=title)
        return paths


def run_compare(cfg, a: str, b: str, *, B: int | None = None, seed: int = 0, log: Callable[[str], None] | None = None,
                **kw) -> CompareResult:
    """Read the union of the two periods once (one network, one key registry, so segment keys agree),
    count once, analyse each period on its own days, then :func:`compare`. ``a`` / ``b``:
    ``"YYYY-MM-DD..YYYY-MM-DD"``. ``kw`` go to :func:`compare`."""
    from .aggregate import count
    from .config import Config
    from .metrics import analyse
    from .pipeline import prepare
    from .segments import segment
    if not isinstance(cfg, Config):
        cfg = Config.from_toml(cfg)
    say = log or (lambda s: None)
    ra, rb = _range(a), _range(b)
    union = f"{min(ra[0], rb[0]).isoformat()}..{max(ra[1], rb[1]).isoformat()}"
    cfg = replace(cfg, select=replace(cfg.select, dates=union))
    pre = prepare(cfg, log=log)
    p = cfg.params
    segs = segment(pre.network, p.segment_m, p.phase_m, p.grid, p.stop_zone)
    cube = count(pre.placed, segs, p.tick_s, per_vehicle=pre.per_vehicle, method=p.per_vehicle)
    ans = []
    for rng_s in (a, b):
        sel = replace(cfg.select, dates=rng_s)
        an = analyse(cube, pre.coverage, pre.passages, segs, pre.network.pattern_days, sel, p, cfg.quality)
        say(f"period {rng_s}: {len(an.days)} days included, {an.excluded_days.height} excluded")
        ans.append(an)
    cmp = compare(ans[0], ans[1], B=p.bootstrap if B is None else B, seed=seed, links=pre.network.links, **kw)
    say(f"compare: {cmp.stretches.height} changed stretches, {cmp.unmatched.height} unmatched segments")
    from .hotspots import hotspot_status, hotspots
    hs = hotspots(ans[1], B=p.bootstrap if B is None else B, links=pre.network.links) if hotspot_status(ans[1])[0] else None
    return CompareResult(cfg=cfg, prepared=pre, segments=segs, a=ans[0], b=ans[1], comparison=cmp, mode=pre.mode,
                         hotspots_b=hs)


__all__ = ["Comparison", "CompareResult", "compare", "compare_contract", "run_compare"]
