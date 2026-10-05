"""Hotspots: stretches of segments where vehicles linger more than at the reference hours.

Per direction, on the display pattern, for every bin (segment x hour, plus the am / pm bands):

1. Day bootstrap (B draws, Rao-Wu rescaled, weekday-stratified, whole days resampled): numerator, passages and
   reference are recomputed from the same resampled days (``Analysis.estimate`` with day weights).
2. Candidate bin when
   * the lower bound of the percentile CI (``level``) of ``excess_per_passage * tick_s`` > ``theta_s``;
   * daily excess (day's obs/passages minus the full-sample reference) is positive on
     ≥ ``min_persistence`` of the days with passages;
   * it passes Benjamini-Hochberg at ``q`` over all tested bins. p-values are one-sided, from the
     bootstrap standard error (z = (x - θ) / se): percentile p-values cannot go below 1/(B+1), which
     BH over thousands of bins never accepts.
3. Temporal persistence: an hourly candidate counts only within a run of ≥ 2 adjacent candidate hours;
   a band candidate (am / pm) counts for all its hours.
4. Stretches: candidate segments merge along the pattern with a ``gap``-bin tolerance, and never
   across a stop / running zone border.
5. Classes (``fixed_score`` F and ``peak_ratio`` are returned as extra columns):
   * F = q20 over day hours 6-20 of the stretch occupancy per metre (obs / passage / m), divided by the
     median of the same quantity over non-candidate segments of the same zone within ±500 m (the
     "free-flow" level of the day; high when the delay never goes away during the day);
   * peak_ratio = mean summed excess over peak hours (am / pm bands) / mean over the other day hours;
   * infrastructure: F ≥ ``f_min`` and peak_ratio < ``peak_ratio``; congestion: peak_ratio ≥
     ``peak_ratio`` and F < ``f_min``; mixed otherwise.
   Note: a delay that is also present at the reference hours cancels in the excess and is not detected.

Stretches are ranked by excess observations per day (Σ over stretch hours and segments of
obs - ref * passages, divided by the number of days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .metrics import BANDS, BAND_HOUR, Analysis, DirData
from .schema import HOTSPOT, conform

PEAK_BANDS = ("am", "pm")


@dataclass
class BinStats:
    """Per-bin statistics of one direction, on the display pattern (columns = output hours + am + pm)."""
    dd: DirData
    ks: np.ndarray               # (Kd,) K indices, display order
    seg_idx: np.ndarray          # (Kd,)
    x0: np.ndarray
    length: np.ndarray
    zone: np.ndarray
    cols: np.ndarray             # (Hc,) labels: hours, then -1 (am), -2 (pm)
    x: np.ndarray                # (Hc,Kd) excess per passage (obs / vehicle)
    est: dict                    # full-sample estimate restricted to (Hc|all hours, Kd), see _cut
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    se: np.ndarray
    p: np.ndarray
    reject: np.ndarray
    persistence: np.ndarray
    candidate: np.ndarray        # per-bin rule (CI, persistence, BH)
    draws: np.ndarray            # (B,Hc,Kd) bootstrap excess per passage
    daily: np.ndarray            # (D,Hc,Kd) daily excess (nan where no passage)
    theta: float                 # threshold in obs per passage


def bootstrap_weights(dow: np.ndarray, B: int, rng: np.random.Generator, stratified: bool = True) -> np.ndarray:
    """(B, D) day weights of a Rao-Wu rescaled day bootstrap (within each weekday when ``stratified``):
    n_h - 1 days drawn with replacement in a stratum of n_h days, multiplicities scaled by
    n_h / (n_h - 1). This removes the (n_h - 1) / n_h variance shrinkage of the naive bootstrap,
    which matters with few days per weekday. A stratum with one day keeps weight 1."""
    D = len(dow)
    W = np.zeros((B, D), np.float32)
    groups = [np.flatnonzero(dow == w) for w in np.unique(dow)] if stratified else [np.arange(D)]
    rows = np.arange(B)[:, None]
    for g in groups:
        n = len(g)
        if n == 0:
            continue
        if n == 1:
            W[:, g] = 1.0
            continue
        pick = g[rng.integers(0, n, (B, n - 1))]
        np.add.at(W, (np.broadcast_to(rows, pick.shape), pick), n / (n - 1))
    return W


def _bh(p: np.ndarray, q: float) -> np.ndarray:
    flat = p.reshape(-1)
    ok = np.isfinite(flat)
    out = np.zeros(flat.shape, bool)
    pv = flat[ok]
    m = len(pv)
    if m == 0:
        return out.reshape(p.shape)
    order = np.argsort(pv)
    passed = pv[order] <= q * np.arange(1, m + 1) / m
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        sel = np.zeros(m, bool)
        sel[order[: kmax + 1]] = True
        out[np.flatnonzero(ok)[sel]] = True
    return out.reshape(p.shape)


_erfc = np.frompyfunc(math.erfc, 1, 1)


def bin_stats(an: Analysis, direction_id: int, *, B: int | None = None, theta_s: float = 2.0, q: float = 0.1,
              level: float = 0.95, min_persistence: float = 0.6, seed: int = 0, stratified: bool = True,
              chunk: int = 50) -> BinStats:
    dd = an.dirs[direction_id]
    B = an.params.bootstrap if B is None else B
    tick = an.params.tick_s
    theta = theta_s / tick
    ks = dd.pattern_segs[dd.display_uid]
    out_hours = set(an.hours)
    ci = np.array([i for i, c in enumerate(dd.cols) if (c >= 0 and int(c) in out_hours)]
                  + [int(np.flatnonzero(dd.cols == BAND_HOUR[b])[0]) for b in PEAK_BANDS])
    cols = dd.cols[ci]

    est = an.estimate(direction_id)
    x = est["excess_per_passage"][0][ci][:, ks]

    rng = np.random.default_rng(seed)
    W = bootstrap_weights(dd.dow, B, rng, stratified)
    draws = np.empty((B, len(ci), len(ks)), np.float32)
    for a in range(0, B, chunk):
        e = an.estimate(direction_id, W[a:a + chunk])
        draws[a:a + chunk] = e["excess_per_passage"][:, ci][:, :, ks]
    alpha = 1 - level
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
            se = np.nanstd(draws, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - theta) / se
        z = np.where(se > 0, z, np.where(x > theta, np.inf, -np.inf))
        p = 0.5 * _erfc(z / math.sqrt(2)).astype(float)
    p[~np.isfinite(x)] = np.nan
    reject = _bh(p, q)

    X = dd.stacked()
    NM, PM = X["NM"][:, ci][:, :, ks], X["PM"][:, ci][:, :, ks]
    ref = est["ref"][0][ks]
    with np.errstate(divide="ignore", invalid="ignore"):
        daily = np.where(PM > 0, NM / PM - ref[None, None, :], np.nan)
        npos = (daily > 0).sum(0)
        nval = np.isfinite(daily).sum(0)
        pers = np.where(nval > 0, npos / nval, np.nan)
    cand = (lo > theta) & (pers >= min_persistence) & reject
    cut = {k: v[0][ci][:, ks] for k, v in est.items() if k != "ref"}
    cut["ref"] = ref
    cut["all"] = {k: v[0][:, ks] for k, v in est.items() if k != "ref"}
    u = dd.display_uid
    return BinStats(dd=dd, ks=ks, seg_idx=dd.pattern_seg_idx[u], x0=dd.pattern_x0[u], length=dd.seg_len[ks],
                    zone=dd.seg_zone[ks], cols=cols, x=x, est=cut, ci_lo=lo, ci_hi=hi, se=se, p=p, reject=reject,
                    persistence=pers, candidate=cand, draws=draws, daily=daily, theta=theta)


def bins_frame(st: BinStats) -> pl.DataFrame:
    Hc, Kd = st.x.shape
    return pl.DataFrame({
        "pattern_uid": [st.dd.display_uid] * (Hc * Kd),
        "direction_id": np.full(Hc * Kd, st.dd.direction_id, np.int8),
        "seg_idx": np.tile(st.seg_idx, Hc),
        "hour": np.repeat(st.cols, Kd).astype(np.int8),
        "excess_per_passage": st.x.reshape(-1),
        "ci_lo": st.ci_lo.reshape(-1), "ci_hi": st.ci_hi.reshape(-1), "se": st.se.reshape(-1),
        "p": st.p.reshape(-1), "bh": st.reject.reshape(-1), "persistence": st.persistence.reshape(-1),
        "candidate": st.candidate.reshape(-1),
    }, nan_to_null=True)


def seg_hours(st: BinStats) -> list[set[int]]:
    """Hours per segment that pass the temporal rule (≥ 2 adjacent candidate hours, or a whole band)."""
    hcols = np.flatnonzero(st.cols >= 0)
    hrs = st.cols[hcols]
    out = []
    for j in range(st.x.shape[1]):
        c = st.candidate[hcols, j]
        keep: set[int] = set()
        for i in range(len(hrs)):
            if not c[i]:
                continue
            prev = i > 0 and c[i - 1] and hrs[i - 1] == hrs[i] - 1
            nxt = i + 1 < len(hrs) and c[i + 1] and hrs[i + 1] == hrs[i] + 1
            if prev or nxt:
                keep.add(int(hrs[i]))
        for b in PEAK_BANDS:
            bi = np.flatnonzero(st.cols == BAND_HOUR[b])
            if len(bi) and st.candidate[bi[0], j]:
                a, z = BANDS[b]
                keep |= {h for h in range(a, z) if h in set(hrs.tolist())}
        out.append(keep)
    return out


def stretches(st: BinStats, gap: int = 1) -> list[tuple[int, int, set[int]]]:
    """[(j0, j1, hours)] in display order (inclusive), merged with a ``gap``-bin tolerance, split at zones."""
    hs = seg_hours(st)
    out: list[list] = []
    for j, h in enumerate(hs):
        if not h:
            continue
        if out:
            j0, j1, hh = out[-1]
            if j - j1 <= gap + 1 and all(st.zone[i] == st.zone[j] for i in range(j1, j + 1)):
                out[-1] = [j0, j, hh | h]
                continue
        out.append([j, j, set(h)])
    return [tuple(s) for s in out]


def hotspots(an: Analysis, *, B: int | None = None, theta_s: float = 2.0, q: float = 0.1, level: float = 0.95,
             min_persistence: float = 0.6, gap: int = 1, f_min: float = 1.5, peak_ratio: float = 2.0,
             seed: int = 0, stratified: bool = True, links: pl.DataFrame | None = None,
             stats: dict[int, BinStats] | None = None) -> pl.DataFrame:
    """Hotspot stretches for every direction of ``an`` (schema.HOTSPOT + extra columns
    excess_obs_per_day, fixed_score, peak_ratio, seg_keys). ``links`` (schema.LINK) names the stops."""
    rows = []
    names = {}
    if links is not None:
        for r in links.select("link_key", "from_name", "to_name").iter_rows():
            names.setdefault(r[0], (r[1], r[2]))
    for dirid in an.dirs:
        if len(an.dirs[dirid].days) == 0:
            continue
        st = stats[dirid] if stats is not None and dirid in stats else bin_stats(
            an, dirid, B=B, theta_s=theta_s, q=q, level=level, min_persistence=min_persistence, seed=seed,
            stratified=stratified)
        rows += _describe(an, st, stretches(st, gap), names, level, f_min, peak_ratio)
    schema = {**HOTSPOT, "excess_obs_per_day": pl.Float32, "fixed_score": pl.Float32, "peak_ratio": pl.Float32,
              "seg_keys": pl.List(pl.Utf8)}
    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema, orient="row")
    df = df.sort(["direction_id", "excess_obs_per_day"], descending=[False, True])
    df = df.with_columns(pl.int_range(1, pl.len() + 1).over("direction_id").cast(pl.Int16).alias("rank"))
    return conform(df, schema)


def _describe(an, st: BinStats, strs, names, level, f_min, peak_ratio_min) -> list[tuple]:
    dd = st.dd
    hcols = np.flatnonzero(st.cols >= 0)
    hrs = st.cols[hcols]
    alpha = 1 - level
    allc = st.est["all"]
    hour_cols_all = dd.hour_cols
    hours_all = dd.cols[hour_cols_all]
    day_sel = hour_cols_all[(hours_all >= 6) & (hours_all <= 20)]
    opp_all = allc["obs_per_passage"]                       # (Hx, Kd)
    cand = np.zeros(len(st.ks), bool)
    for j0, j1, _ in strs:
        cand[j0:j1 + 1] = True
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            seg_ff = np.nanquantile(opp_all[day_sel] / st.length[None, :], 0.2, axis=0)
    mid = st.x0 + st.length / 2
    D = max(len(dd.days), 1)
    peak_hours = {h for b in PEAK_BANDS for h in range(*BANDS[b])}
    obs_h, pas_h, ref = st.est["obs"], st.est["passages"], st.est["ref"]
    out = []
    for j0, j1, hours in strs:
        sl = slice(j0, j1 + 1)
        hidx = np.array([hcols[np.flatnonzero(hrs == h)[0]] for h in sorted(hours)])
        sx = np.nansum(st.x[hidx][:, sl], axis=1)
        pk = int(np.nanargmax(sx))
        pcol = hidx[pk]
        peak_hour = int(st.cols[pcol])
        val = float(sx[pk])
        d = np.nansum(st.draws[:, pcol, sl], axis=1)
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        dly = st.daily[:, pcol, sl]
        okd = np.isfinite(dly).all(1)
        pers = float((dly[okd].sum(1) > 0).mean()) if okd.any() else float("nan")
        exc_day = float(np.nansum(obs_h[hidx][:, sl] - ref[None, sl] * pas_h[hidx][:, sl]) / D)
        # fixed score
        with np.errstate(all="ignore"):
            occ = np.nansum(opp_all[day_sel][:, sl], axis=1) / st.length[sl].sum()
            f_num = float(np.nanquantile(occ, 0.2)) if np.isfinite(occ).any() else float("nan")
        zone = st.zone[j0]
        c = 0.5 * (mid[j0] + mid[j1])
        base_m = (~cand) & (st.zone == zone) & np.isfinite(seg_ff)
        near = base_m & (np.abs(mid - c) <= 500)
        pool = seg_ff[near] if near.sum() >= 5 else seg_ff[base_m]
        base = float(np.median(pool)) if len(pool) else float("nan")
        F = f_num / base if base and base > 0 else float("nan")
        # peak ratio over day hours, from the full-sample per-hour summed excess
        dh = [h for h in range(6, 20) if h in set(hrs.tolist())]
        summed = {h: float(np.nansum(st.x[hcols[np.flatnonzero(hrs == h)[0]], sl])) for h in dh}
        pv = [v for h, v in summed.items() if h in peak_hours]
        ov = [v for h, v in summed.items() if h not in peak_hours]
        pmean = np.mean(pv) if pv else float("nan")
        omean = np.mean(ov) if ov else float("nan")
        ratio = float(pmean / max(omean, st.theta / 2)) if np.isfinite(pmean) else float("nan")
        if np.isfinite(F) and F >= f_min and not ratio >= peak_ratio_min:
            kind = "infrastructure"
        elif ratio >= peak_ratio_min and not (np.isfinite(F) and F >= f_min):
            kind = "congestion"
        else:
            kind = "mixed"
        lk0 = dd.link_keys[dd.seg_link[st.ks[j0]]]
        lk1 = dd.link_keys[dd.seg_link[st.ks[j1]]]
        n0 = names.get(lk0, (lk0, None))[0]
        n1 = names.get(lk1, (None, lk1))[1]
        out.append((dd.display_uid, dd.direction_id, 0, int(st.seg_idx[j0]), int(st.seg_idx[j1]), float(st.x0[j0]),
                    float(st.x0[j1] + st.length[j1]), n0, n1, zone, kind, sorted(int(h) for h in hours), peak_hour,
                    val, float(lo), float(hi), pers, int(dd.V[:, st.ks[sl]].sum(0).min()), exc_day, F, ratio,
                    dd.seg_keys[st.ks[sl]].tolist()))
    return out


def to_geojson(hotspots: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """GeoJSON FeatureCollection, one LineString per stretch (concatenated segment geometries)."""
    geo = {}
    for (u,), g in segments.select("pattern_uid", "seg_idx", "geometry").group_by(["pattern_uid"]):
        g = g.sort("seg_idx")
        geo[u] = (g["seg_idx"].to_numpy(), g["geometry"].to_list())
    feats = []
    for r in hotspots.iter_rows(named=True):
        coords: list = []
        if r["pattern_uid"] in geo:
            idx, gs = geo[r["pattern_uid"]]
            for i in np.flatnonzero((idx >= r["from_seg"]) & (idx <= r["to_seg"])):
                for pt in gs[i] or []:
                    if not coords or coords[-1] != list(pt):
                        coords.append(list(pt))
        props = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "LineString", "coordinates": coords} if len(coords) >= 2 else None})
    return {"type": "FeatureCollection", "features": feats}
