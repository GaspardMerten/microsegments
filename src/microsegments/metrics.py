"""Cube -> indicators per segment x hour (and per band), robust to missing data and GTFS versions.

Notation (per direction): days d (included service days), hours h, segments k (keyed by the stable
``seg_key``), links l (``link_key``).

* ``N[d,h,k]`` observations, ``C[d,h]`` coverage (share of the hour the feed was alive),
  ``P[d,h,l]`` passages.
* Day-hour (d,h) excluded (numerator and denominator) when coverage < ``min_hour_coverage``, or the
  hour carries a LOW_COVERAGE / FROZEN / LINE_ABSENT / DEVIATION / VEHICLE_RATIO flag, or it has no
  coverage row at all. A day is excluded when more than ``1 - min_day_coverage`` of its 6-21 h
  day-hours are excluded (listed in ``excluded_days`` with the dominant reason).
* Validity: segment k counts on day d only if its ``seg_key`` belongs to that day's main pattern for
  the direction (``PatternDay.is_main``). Every sum below runs over valid (d, h, k) only (D_l), so a
  segment cut on some days is aggregated over its other days, reference included.
* Observations and passages of every pattern of the direction (short workings, detours) are summed
  per seg_key / link_key: keys are physical, a vehicle seen in a segment is seen there.

Indicators (ratio of sums; optionally post-stratified by weekday, weights = selected days per
weekday)::

    obs_per_h         = Σ N / Σ C
    obs_per_passage O = Σ N / Σ P            (null if passages per day < min_passages_per_day)
    reference r       = Σ_{h in ref} N / Σ_{h in ref} P     (evening; same valid days)
                        or q20 over hours 5-23 of O        (reference="freeflow")
    excess_per_passage = O - r
    excess_obs_per_h   = (Σ N - r Σ P) / Σ C
    log2_ratio         = log2((O + .5) / (r + .5))
    line reference r_l = m * len_k, m = median over running segments of Σ_{6-23 h} N / Σ P / len
                         (the typical running occupancy per metre of the direction; extra result
                         columns ``ref_line_obs_per_passage`` and ``excess_line_per_passage`` = O - r_l)

Bands are emitted as extra rows with ``hour`` = -1 (am 7-10), -2 (pm 16-19), -3 (day 6-21),
-4 (evening 20-23), all [start, end).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl

from .config import Params, Quality, Select
from .schema import RESULT, Flag, conform

BANDS: dict[str, tuple[int, int]] = {"am": (7, 10), "pm": (16, 19), "day": (6, 21), "evening": (20, 23)}
BAND_HOUR: dict[str, int] = {"am": -1, "pm": -2, "day": -3, "evening": -4}
DAY_HOURS = (6, 21)
LINE_HOURS = (6, 23)     # [start, end): window of the line-level reference (same as hotspots.ALLDAY_HOURS)
EXCLUDING = Flag.LOW_COVERAGE | Flag.FROZEN | Flag.LINE_ABSENT | Flag.DEVIATION | Flag.VEHICLE_RATIO
_REASONS = ((Flag.FROZEN, "frozen"), (Flag.LINE_ABSENT, "line_absent"), (Flag.DEVIATION, "deviation"),
            (Flag.VEHICLE_RATIO, "vehicle_ratio"))
ALPHA = 0.5


# ------------------------------------------------------------------------------------ containers
@dataclass
class DirData:
    """Dense day x hour x segment arrays of one direction (internal, used by hotspots / tune)."""
    direction_id: int
    display_uid: str
    days: np.ndarray                 # datetime64[D] (D,)
    dow: np.ndarray                  # (D,) 0 = Monday
    main_uid: np.ndarray             # (D,) main pattern of each day
    hours: np.ndarray                # (H,) internal hours
    cols: np.ndarray                 # (Hx,) column labels: hours, then band codes -1..-4
    seg_keys: np.ndarray             # (K,)
    seg_link: np.ndarray             # (K,) index into link_keys
    seg_len: np.ndarray              # (K,)
    seg_zone: np.ndarray             # (K,)
    link_keys: np.ndarray            # (Lk,)
    pattern_segs: dict[str, np.ndarray]    # uid -> K indices in seg_idx order
    pattern_seg_idx: dict[str, np.ndarray]
    pattern_x0: dict[str, np.ndarray]
    pattern_links: dict[str, np.ndarray]   # uid -> link indices in link order
    V: np.ndarray                    # (D,K) bool
    LV: np.ndarray                   # (D,Lk) bool
    E: np.ndarray                    # (D,H) bool, day-hour included
    C: np.ndarray                    # (D,H) coverage, 0 where excluded
    N: np.ndarray                    # (D,H,K)
    P: np.ndarray                    # (D,H,Lk)
    _x: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def col_index(self, labels) -> np.ndarray:
        pos = {int(c): i for i, c in enumerate(self.cols)}
        return np.array([pos[int(c)] for c in labels if int(c) in pos], dtype=np.int64)

    @property
    def hour_cols(self) -> np.ndarray:
        return np.flatnonzero(self.cols >= 0)

    def stacked(self) -> dict[str, np.ndarray]:
        """(D, Hx, K) float32 arrays NM, PM, CM, M: masked numerator, passages, coverage, cell count;
        band columns are sums over their hours (M: 1 if any hour of the band is included)."""
        if self._x:
            return self._x
        M = self.E[:, :, None] & self.V[:, None, :]
        NM = np.where(M, self.N, 0.0)
        PM = np.where(M, self.P[:, :, self.seg_link], 0.0)
        CM = np.where(M, self.C[:, :, None], 0.0)
        bands = []
        for b, (a, z) in BANDS.items():
            idx = np.flatnonzero((self.hours >= a) & (self.hours < z))
            bands.append(idx)
        def ext(A, red):
            return np.concatenate([A] + [red(A[:, i, :])[:, None, :] for i in bands], axis=1).astype(np.float32)
        self._x = {
            "NM": ext(NM, lambda a: a.sum(1)),
            "PM": ext(PM, lambda a: a.sum(1)),
            "CM": ext(CM, lambda a: a.sum(1)),
            "M": ext(M.astype(np.float32), lambda a: a.max(1) if a.shape[1] else a.sum(1)),
        }
        return self._x

    def take(self, idx: np.ndarray) -> "DirData":
        idx = np.asarray(idx)
        return replace(self, days=self.days[idx], dow=self.dow[idx], main_uid=self.main_uid[idx],
                       V=self.V[idx], LV=self.LV[idx], E=self.E[idx], C=self.C[idx], N=self.N[idx],
                       P=self.P[idx], _x={})


@dataclass
class Analysis:
    result: pl.DataFrame                    # schema.RESULT (+ seg_len_m, zone, link_key, n_cells)
    excluded_days: pl.DataFrame             # service_date, reason
    day_hours: pl.DataFrame                 # service_date, hour, coverage, flags, included
    days: list[dt.date]                     # included service days
    per_dow: list[int]                      # included days per weekday
    versions: pl.DataFrame                  # direction_id, pattern_uid, first, last, n_days, display, links_added, links_removed
    validity: pl.DataFrame                  # direction_id, service_date, pattern_uid (main that day)
    select: Select
    params: Params
    quality: Quality
    stratify: bool
    segments: pl.DataFrame
    hours: list[int]                        # output hours (select.hours)
    dirs: dict[int, DirData] = field(repr=False)
    pi: np.ndarray = field(repr=False, default_factory=lambda: np.full(7, 1 / 7))  # weekday weights

    # ------------------------------------------------------------------ estimation
    def estimate(self, direction_id: int, W: np.ndarray | None = None, params: Params | None = None, *,
                 coverage: bool = True) -> dict[str, np.ndarray]:
        """Indicators for one direction with day weights W (B, D) (bootstrap multiplicities);
        arrays (B, Hx, K) except ``ref``, ``ref_line`` (B, K) and ``line_level`` (B,).
        ``coverage=False`` skips the covered-hour sums (``obs_per_h`` and ``excess_obs_per_h`` are then
        NaN): a quarter less work in per-passage bootstraps."""
        dd = self.dirs[direction_id]
        if W is None:
            W = np.ones((1, len(dd.days)), np.float32)
        return _estimate(dd, W, params or self.params, self.quality, self.pi if self.stratify else None, coverage)

    def subset(self, days) -> "Analysis":
        """Same analysis on a subset of the included days (split-half, leave-out)."""
        keep = {np.datetime64(d, "D") for d in days}
        dirs = {}
        for k, dd in self.dirs.items():
            idx = np.array([i for i, d in enumerate(dd.days) if d in keep], dtype=np.int64)
            dirs[k] = dd.take(idx)
        out = replace(self, dirs=dirs, days=[d for d in self.days if np.datetime64(d, "D") in keep])
        out.result = _result(out)
        return out

    def with_params(self, params: Params) -> "Analysis":
        out = replace(self, params=params)
        out.result = _result(out)
        return out

    # ------------------------------------------------------------------ contract
    def to_contract_wk(self, direction_id: int | None = None, pattern_uid: str | None = None) -> dict[str, Any]:
        """Per-weekday totals for the JSON contract (``contract.py``), for the display pattern of each
        direction (or ``pattern_uid``): obs[dow][h][seg], cov[dow][h][link], p[dow][h][link],
        days[dow][link], n[dow][h][link] (included day-hours, so p / n = passages per hour). Hours are ``self.hours``. Returns {direction_id: wk} if direction_id is None."""
        if direction_id is None:
            return {k: self.to_contract_wk(k) for k in self.dirs}
        dd = self.dirs[direction_id]
        uid = pattern_uid or dd.display_uid
        ks, ls = dd.pattern_segs[uid], dd.pattern_links[uid]
        hi = np.array([int(np.flatnonzero(dd.hours == h)[0]) for h in self.hours if h in set(dd.hours.tolist())])
        M = dd.E[:, :, None] & dd.V[:, None, :]
        NM = np.where(M, dd.N, 0.0)[:, hi][:, :, ks]
        LM = dd.E[:, :, None] & dd.LV[:, None, :]
        CL = np.where(LM, dd.C[:, :, None], 0.0)[:, hi][:, :, ls]
        PL = np.where(LM, dd.P, 0.0)[:, hi][:, :, ls]
        NL = LM[:, hi][:, :, ls]
        obs, cov, p, days, n = [], [], [], [], []
        for w in range(7):
            m = dd.dow == w
            obs.append(np.rint(NM[m].sum(0)).astype(int).tolist())
            cov.append(np.round(CL[m].sum(0), 3).tolist())
            p.append(np.round(PL[m].sum(0), 2).tolist())
            days.append(dd.LV[m][:, ls].sum(0).astype(int).tolist())
            n.append(NL[m].sum(0).astype(int).tolist())
        return {"pattern_uid": uid, "hours": list(self.hours), "seg_key": dd.seg_keys[ks].tolist(),
                "link_key": dd.link_keys[ls].tolist(), "obs": obs, "cov": cov, "p": p, "days": days, "n": n}


# ------------------------------------------------------------------------------------ estimator
def _ratio(num: np.ndarray, den: np.ndarray, pi: np.ndarray | None, ok: np.ndarray | None = None):
    """num, den: (S, ...) per stratum when pi is given, else plain arrays. Ratio of sums, or
    post-stratified mean of per-stratum ratios (weights renormalised over strata with data)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if pi is None:
            r = num / den
            r[~(den > 0)] = np.nan
            if ok is not None:
                r[~ok] = np.nan
            return r
        good = den > 0
        if ok is not None:
            good &= ok
        r = np.where(good, num / np.where(good, den, 1), 0.0)
        w = pi.reshape((-1,) + (1,) * (num.ndim - 1)) * good
        tot = w.sum(0)
        out = (w * r).sum(0) / tot
        out[~(tot > 0)] = np.nan
        return out


def _estimate(dd: DirData, W: np.ndarray, params: Params, quality: Quality, pi: np.ndarray | None,
              coverage: bool = True) -> dict[str, np.ndarray]:
    X = dd.stacked()
    D, Hx, K = X["NM"].shape
    W = np.asarray(W, np.float32)
    B = W.shape[0]

    def sums(Wm):
        out = {k: (Wm @ X[k].reshape(D, -1)).reshape(B, Hx, K).astype(np.float64)
               for k in (("NM", "PM", "CM", "M") if coverage else ("NM", "PM", "M"))}
        if not coverage:
            out["CM"] = np.zeros((B, Hx, K))
        return out

    if pi is None:
        s = sums(W)
        S = None
    else:
        strata = [w for w in range(7) if (dd.dow == w).any()]
        per = [sums(W * (dd.dow == w)[None, :].astype(np.float32)) for w in strata]
        S = {k: np.stack([p[k] for p in per]) for k in ("NM", "PM", "CM", "M")}
        s = {k: S[k].sum(0) for k in S}
        pi = pi[strata] / pi[strata].sum()
    obs, pas, cov, ncell = s["NM"], s["PM"], s["CM"], s["M"]
    hc = dd.hour_cols
    hours = dd.cols[hc]

    def rate(nk, dk, ok=None):
        if S is None:
            return _ratio(s[nk], s[dk], None, ok)
        return _ratio(S[nk], S[dk], pi, ok)

    with np.errstate(divide="ignore", invalid="ignore"):
        enough = (pas / np.where(ncell > 0, ncell, np.nan)) >= quality.min_passages_per_day
    oph = rate("NM", "CM")
    opp = rate("NM", "PM", enough if S is None else None)
    if S is not None:
        opp[~enough] = np.nan

    if params.reference == "freeflow":
        sel = hc[(hours >= 5) & (hours <= 23)]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = np.nanquantile(opp[:, sel, :], 0.2, axis=1)
    else:
        a, z = params.reference_hours
        sel = hc[(hours >= a) & (hours < z)]
        if S is None:
            ref = _ratio(obs[:, sel].sum(1), pas[:, sel].sum(1), None)
        else:
            ref = _ratio(S["NM"][:, :, sel].sum(2), S["PM"][:, :, sel].sum(2), pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        exc_pp = opp - ref[:, None, :]
        if S is None:
            exc_oph = (obs - ref[:, None, :] * pas) / np.where(cov > 0, cov, np.nan)
        else:
            exc_oph = _ratio(S["NM"] - ref[None, :, None, :] * S["PM"], S["CM"], pi)
        log2 = np.log2((opp + ALPHA) / (ref[:, None, :] + ALPHA))
    level, ref_line = line_level(obs[:, hc], pas[:, hc], hours, dd.seg_len, dd.seg_zone == "running")
    with np.errstate(divide="ignore", invalid="ignore"):
        exc_line = opp - ref_line[:, None, :]
    return {"obs": obs, "passages": pas, "covered_h": cov, "n_cells": ncell, "obs_per_h": oph,
            "obs_per_passage": opp, "ref": ref, "excess_per_passage": exc_pp, "excess_obs_per_h": exc_oph,
            "log2_ratio": log2, "line_level": level, "ref_line": ref_line, "excess_line_per_passage": exc_line}


def line_level(obs: np.ndarray, pas: np.ndarray, hours: np.ndarray, seg_len: np.ndarray, running: np.ndarray,
               window: tuple[int, int] = LINE_HOURS) -> tuple[np.ndarray, np.ndarray]:
    """Line-level reference. ``obs``, ``pas``: (B, H, K) sums per hour column (``hours`` labels).
    Level m (B,) = median over running segments of the ``window`` hours' obs / passages / metre (the
    typical running occupancy per metre of the direction, the "persistent" hotspot criterion's level);
    reference per segment (B, K) = m x length. NaN when no running segment has passages."""
    import warnings
    sel = (hours >= window[0]) & (hours < window[1])
    B, K = obs.shape[0], obs.shape[-1]
    if not sel.any() or not running.any():
        return np.full(B, np.nan), np.full((B, K), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        O, P = obs[:, sel].sum(1), pas[:, sel].sum(1)
        occ = np.where(P > 0, O / np.where(P > 0, P, 1), np.nan)
        m = np.nanmedian(np.where(running[None, :], occ / seg_len[None, :], np.nan), axis=1)
    return m, m[:, None] * seg_len[None, :]


# ------------------------------------------------------------------------------------ analyse
def analyse(cube: pl.DataFrame, coverage: pl.DataFrame | None = None, passages: pl.DataFrame | None = None,
            segments: pl.DataFrame | None = None, pattern_days: pl.DataFrame | None = None,
            select: Select | None = None, params: Params | None = None, quality: Quality | None = None,
            *, stratify: bool = False, passage_col: str = "n") -> Analysis:
    """Aggregate a Cube over the selected days. See the module docstring for the rules.

    ``coverage`` None: every hour fully covered. ``pattern_days`` None: every pattern of ``segments``
    is main on every day, one direction per pattern. ``passage_col``: PASSAGES column used as passages
    ("n", "n_feed" or "n_events")."""
    if segments is None:
        raise ValueError("segments is required")
    select, params, quality = select or Select(), params or Params(), quality or Quality()

    # ---------------------------------------------------------------- day universe
    dates = set(cube["service_date"].unique().to_list())
    if coverage is not None:
        dates |= set(coverage["service_date"].unique().to_list())
    rng = select.date_range()
    if rng is not None:
        if pattern_days is not None:
            dates |= set(pattern_days["service_date"].unique().to_list())
        dates = {d for d in dates if rng[0] <= d <= rng[1]}
    dates = sorted(d for d in dates if d.weekday() in set(select.weekdays))
    user_ex = {dt.date.fromisoformat(s) for s in select.exclude_dates}

    day_pos = {d: i for i, d in enumerate(dates)}
    nd = len(dates)
    covA = np.full((nd, 28), np.nan)
    flA = np.zeros((nd, 28), np.int64)
    if coverage is None:
        covA[:] = 1.0
    elif nd:
        cv = coverage.filter(pl.col("service_date").is_in(dates) & pl.col("hour").is_between(0, 27))
        di = np.array([day_pos[d] for d in cv["service_date"].to_list()], dtype=np.int64)
        hh = cv["hour"].to_numpy().astype(np.int64)
        covA[di, hh] = np.clip(cv["coverage"].fill_null(0).to_numpy(), 0, 1)
        flA[di, hh] = cv["flags"].fill_null(0).to_numpy().astype(np.int64)
    ex = np.isnan(covA) | (np.nan_to_num(covA) < quality.min_hour_coverage) | ((flA & EXCLUDING) != 0)

    excluded, included = [], []
    a, z = DAY_HOURS
    for i, d in enumerate(dates):
        if d in user_ex:
            excluded.append((d, "excluded"))
            continue
        exd = ex[i, a:z]
        if exd.mean() > 1 - quality.min_day_coverage + 1e-9:
            excluded.append((d, _reason(covA[i, a:z], flA[i, a:z], exd, quality)))
        else:
            included.append(d)
    excluded_days = pl.DataFrame(excluded, schema={"service_date": pl.Date, "reason": pl.Utf8}, orient="row")
    inc_pos = {d: day_pos[d] for d in included}
    dh = []
    for d in dates:
        i = day_pos[d]
        for h in range(28):
            if not np.isnan(covA[i, h]):
                dh.append((d, h, float(covA[i, h]), int(flA[i, h]), d in inc_pos and not ex[i, h]))
    day_hours = pl.DataFrame(dh, schema={"service_date": pl.Date, "hour": pl.Int8, "coverage": pl.Float32,
                                         "flags": pl.UInt16, "included": pl.Boolean}, orient="row")

    # ---------------------------------------------------------------- patterns per day
    if pattern_days is None:
        uids = segments["pattern_uid"].unique(maintain_order=True).to_list()
        pattern_days = pl.DataFrame(
            [(d, None, i, u, 1, True) for i, u in enumerate(uids) for d in included],
            schema={"service_date": pl.Date, "route_id": pl.Utf8, "direction_id": pl.Int8, "pattern_uid": pl.Utf8,
                    "n_trips": pl.UInt32, "is_main": pl.Boolean}, orient="row")
    mains = (pattern_days.filter(pl.col("is_main") & pl.col("service_date").is_in(included))
             .sort(["service_date", "direction_id", "n_trips", "pattern_uid"], descending=[False, False, True, False])
             .unique(["service_date", "direction_id"], keep="first", maintain_order=True))
    if select.directions is not None:
        mains = mains.filter(pl.col("direction_id").is_in(select.directions))

    hours_out = list(range(select.hours[0], select.hours[1]))
    internal = set(hours_out) | set(range(*params.reference_hours)) | set(range(DAY_HOURS[0], 23))
    if params.reference == "freeflow":
        internal |= set(range(5, 24))
    hours = np.array(sorted(internal), dtype=np.int64)
    cols = np.concatenate([hours, np.array(list(BAND_HOUR.values()), dtype=np.int64)])

    seg = segments.select("pattern_uid", "seg_idx", "link_idx", "link_key", "seg_key", "x0_m", "len_m", "zone")
    dirs: dict[int, DirData] = {}
    for (dirid,), md in mains.group_by(["direction_id"], maintain_order=True):
        md = md.sort("service_date")
        dd = _build_dir(int(dirid), md, seg, cube, passages, passage_col, covA, ex, day_pos, hours, cols)
        if dd is not None:
            dirs[int(dirid)] = dd

    per_dow = [sum(1 for d in included if d.weekday() == w) for w in range(7)]
    sel_dow = np.array([sum(1 for d in dates if d.weekday() == w and d not in user_ex) for w in range(7)], float)
    pi = sel_dow / sel_dow.sum() if sel_dow.sum() else np.full(7, 1 / 7)

    versions, validity = _versions(dirs)
    an = Analysis(result=pl.DataFrame(), excluded_days=excluded_days, day_hours=day_hours, days=included,
                  per_dow=per_dow, versions=versions, validity=validity, select=select, params=params,
                  quality=quality, stratify=stratify, segments=segments, hours=hours_out, dirs=dirs, pi=pi)
    an.result = _result(an)
    return an


def _reason(cov, fl, exd, quality) -> str:
    votes: dict[str, int] = {}
    for c, f, e in zip(cov, fl, exd):
        if not e:
            continue
        if np.isnan(c):
            r = "no_data"
        else:
            r = next((name for bit, name in _REASONS if f & bit), "low_coverage")
        votes[r] = votes.get(r, 0) + 1
    return max(votes, key=votes.get) if votes else "low_coverage"


def _build_dir(dirid, md, seg, cube, passages, passage_col, covA, ex, day_pos, hours, cols) -> DirData | None:
    days_l = md["service_date"].to_list()
    main_uid = np.array(md["pattern_uid"].to_list(), dtype=object)
    if not days_l:
        return None
    uids, counts = np.unique(main_uid, return_counts=True)
    # display: the most frequent link_key sequence (patterns differing by shape edits only are pooled),
    # and within it the pattern with the most days
    seq = {u: tuple(g.sort("link_idx")["link_key"].unique(maintain_order=True).to_list())
           for (u,), g in seg.filter(pl.col("pattern_uid").is_in(uids.tolist())).group_by(["pattern_uid"])}
    tot: dict[tuple, int] = {}
    for u, c in zip(uids, counts):
        tot[seq.get(u, (u,))] = tot.get(seq.get(u, (u,)), 0) + int(c)
    order = sorted(range(len(uids)), key=lambda i: (-tot[seq.get(uids[i], (uids[i],))], -counts[i]))
    uids = uids[order]
    display = str(uids[0])
    s = seg.filter(pl.col("pattern_uid").is_in(uids.tolist()))
    if s.height == 0:
        return None
    # seg_key universe: display pattern first, then the others (first occurrence wins)
    rank = {u: i for i, u in enumerate(uids)}
    s = s.with_columns(pl.col("pattern_uid").replace_strict(rank, return_dtype=pl.Int32).alias("_r")).sort("_r", "seg_idx")
    uk = s.unique("seg_key", keep="first", maintain_order=True)
    seg_keys = np.array(uk["seg_key"].to_list(), dtype=object)
    kpos = {k: i for i, k in enumerate(seg_keys)}
    link_keys = np.array(s.unique("link_key", keep="first", maintain_order=True)["link_key"].to_list(), dtype=object)
    lpos = {k: i for i, k in enumerate(link_keys)}
    seg_link = np.array([lpos[k] for k in uk["link_key"].to_list()], dtype=np.int64)
    K, Lk = len(seg_keys), len(link_keys)

    pattern_segs, pattern_seg_idx, pattern_x0, pattern_links = {}, {}, {}, {}
    inc = np.zeros((len(uids), K), bool)
    linc = np.zeros((len(uids), Lk), bool)
    for i, u in enumerate(uids):
        su = s.filter(pl.col("pattern_uid") == u).sort("seg_idx")
        ks = np.array([kpos[k] for k in su["seg_key"].to_list()], dtype=np.int64)
        pattern_segs[u] = ks
        pattern_seg_idx[u] = su["seg_idx"].to_numpy()
        pattern_x0[u] = su["x0_m"].to_numpy()
        lk = su.sort("link_idx").unique("link_idx", keep="first", maintain_order=True)["link_key"].to_list()
        pattern_links[u] = np.array([lpos[k] for k in lk], dtype=np.int64)
        inc[i, ks] = True
        linc[i, pattern_links[u]] = True
    upos = {u: i for i, u in enumerate(uids)}
    mi = np.array([upos[u] for u in main_uid])
    V, LV = inc[mi], linc[mi]

    D, H = len(days_l), len(hours)
    gi = np.array([day_pos[d] for d in days_l])
    E = ~ex[gi][:, hours]
    C = np.where(E, np.nan_to_num(covA[gi][:, hours]), 0.0)

    days_df = pl.DataFrame({"service_date": days_l, "_d": np.arange(D)}, schema={"service_date": pl.Date, "_d": pl.Int64})
    hours_df = pl.DataFrame({"hour": hours, "_h": np.arange(H)}, schema={"hour": pl.Int8, "_h": pl.Int64})
    keys_df = pl.DataFrame({"seg_key": seg_keys.tolist(), "_k": np.arange(K)}, schema={"seg_key": pl.Utf8, "_k": pl.Int64})
    c = (cube.lazy().select("service_date", pl.col("hour").cast(pl.Int8), "seg_key", "obs")
         .join(days_df.lazy(), on="service_date").join(hours_df.lazy(), on="hour").join(keys_df.lazy(), on="seg_key")
         .select("_d", "_h", "_k", "obs").collect())
    flat = (c["_d"].to_numpy() * H + c["_h"].to_numpy()) * K + c["_k"].to_numpy()
    N = np.bincount(flat, weights=c["obs"].to_numpy().astype(np.float64), minlength=D * H * K).reshape(D, H, K)

    P = np.zeros((D, H, Lk))
    if passages is not None and passages.height:
        col = passage_col if passage_col in passages.columns else "n"
        lk_df = pl.DataFrame({"link_key": link_keys.tolist(), "_l": np.arange(Lk)}, schema={"link_key": pl.Utf8, "_l": pl.Int64})
        p = (passages.lazy().select("service_date", pl.col("hour").cast(pl.Int8), "link_key", pl.col(col).fill_null(0).alias("_n"))
             .join(days_df.lazy(), on="service_date").join(hours_df.lazy(), on="hour").join(lk_df.lazy(), on="link_key")
             .select("_d", "_h", "_l", "_n").collect())
        flat = (p["_d"].to_numpy() * H + p["_h"].to_numpy()) * Lk + p["_l"].to_numpy()
        P = np.bincount(flat, weights=p["_n"].to_numpy().astype(np.float64), minlength=D * H * Lk).reshape(D, H, Lk)

    days = np.array(days_l, dtype="datetime64[D]")
    dow = ((days.astype("int64") + 3) % 7).astype(np.int64)   # 1970-01-01 was a Thursday
    return DirData(direction_id=dirid, display_uid=display, days=days, dow=dow, main_uid=main_uid, hours=hours,
                   cols=cols, seg_keys=seg_keys, seg_link=seg_link, seg_len=uk["len_m"].to_numpy().astype(float),
                   seg_zone=np.array(uk["zone"].to_list(), dtype=object), link_keys=link_keys,
                   pattern_segs=pattern_segs, pattern_seg_idx=pattern_seg_idx, pattern_x0=pattern_x0,
                   pattern_links=pattern_links, V=V, LV=LV, E=E, C=C, N=N, P=P)


def version_groups(dd: DirData) -> dict[str, list[str]]:
    """Main patterns of a direction grouped by link_key sequence: {representative uid: uids}, the
    display pattern first. Patterns that differ only by small shape edits (same link keys, hence the
    same segments) are one version; the representative is the display pattern or the one with most days."""
    seqs: dict[tuple, list[str]] = {}
    for u in dd.pattern_segs:
        if (dd.main_uid == u).any():
            seqs.setdefault(tuple(dd.link_keys[dd.pattern_links[u]].tolist()), []).append(u)
    out: dict[str, list[str]] = {}
    groups = sorted(seqs.values(), key=lambda us: (dd.display_uid not in us, -sum(int((dd.main_uid == u).sum()) for u in us)))
    for us in groups:
        rep = dd.display_uid if dd.display_uid in us else max(us, key=lambda u: int((dd.main_uid == u).sum()))
        out[rep] = us
    return out


def _versions(dirs: dict[int, DirData]) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, val = [], []
    for k, dd in dirs.items():
        disp = set(dd.link_keys[dd.pattern_links[dd.display_uid]].tolist())
        for u, us in version_groups(dd).items():
            m = np.isin(dd.main_uid, np.array(us, dtype=object))
            links = set(dd.link_keys[dd.pattern_links[u]].tolist())
            rows.append({"direction_id": k, "pattern_uid": u, "first": dd.days[m].min().item(), "last": dd.days[m].max().item(),
                         "n_days": int(m.sum()), "display": u == dd.display_uid,
                         "links_added": sorted(links - disp), "links_removed": sorted(disp - links)})
        val.extend((k, d.item(), u) for d, u in zip(dd.days, dd.main_uid))
    vs = {"direction_id": pl.Int8, "pattern_uid": pl.Utf8, "first": pl.Date, "last": pl.Date, "n_days": pl.UInt16,
          "display": pl.Boolean, "links_added": pl.List(pl.Utf8), "links_removed": pl.List(pl.Utf8)}
    versions = pl.DataFrame(rows, schema=vs) if rows else pl.DataFrame(schema=vs)
    validity = pl.DataFrame(val, schema={"direction_id": pl.Int8, "service_date": pl.Date, "pattern_uid": pl.Utf8}, orient="row")
    return versions, validity


def _result(an: Analysis) -> pl.DataFrame:
    frames = []
    out_cols_label = set(an.hours) | set(BAND_HOUR.values())
    for dirid, dd in an.dirs.items():
        if len(dd.days) == 0:
            continue
        est = an.estimate(dirid)
        ci = np.array([i for i, c in enumerate(dd.cols) if int(c) in out_cols_label])
        n_days = dd.V.sum(0)
        D = len(dd.days)
        fl = np.where(n_days < an.quality.min_days, Flag.FEW_DAYS, 0) | np.where(n_days < D, Flag.PARTIAL_VERSIONS, 0)
        for u, ks in dd.pattern_segs.items():
            nk, nc = len(ks), len(ci)
            def g(name):
                return est[name][0][ci][:, ks].T.reshape(-1)       # seg-major
            ref = np.repeat(est["ref"][0][ks], nc)
            ref_line = np.repeat(est["ref_line"][0][ks], nc)
            frames.append(pl.DataFrame({
                "pattern_uid": [u] * (nk * nc),
                "direction_id": np.full(nk * nc, dirid),
                "seg_key": np.repeat(dd.seg_keys[ks], nc).tolist(),
                "seg_idx": np.repeat(dd.pattern_seg_idx[u], nc),
                "hour": np.tile(dd.cols[ci], nk),
                "obs": np.rint(g("obs")),
                "n_days": np.repeat(n_days[ks], nc),
                "covered_h": g("covered_h"),
                "obs_per_h": g("obs_per_h"),
                "passages": g("passages"),
                "obs_per_passage": g("obs_per_passage"),
                "ref_obs_per_passage": ref,
                "excess_obs_per_h": g("excess_obs_per_h"),
                "excess_per_passage": g("excess_per_passage"),
                "log2_ratio": g("log2_ratio"),
                "flags": np.repeat(fl[ks], nc),
                "link_key": np.repeat(dd.link_keys[dd.seg_link[ks]], nc).tolist(),
                "x0_m": np.repeat(dd.pattern_x0[u], nc),
                "len_m": np.repeat(dd.seg_len[ks], nc),
                "zone": np.repeat(dd.seg_zone[ks], nc).tolist(),
                "n_cells": g("n_cells"),
                "ref_line_obs_per_passage": ref_line,
                "excess_line_per_passage": g("excess_line_per_passage"),
            }, nan_to_null=True))
    if not frames:
        return conform(pl.DataFrame(schema=RESULT), RESULT)
    return conform(pl.concat(frames), RESULT)


# ------------------------------------------------------------------------------------ references
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pl.Series(a[ok]).rank("average").to_numpy()
    rb = pl.Series(b[ok]).rank("average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def reference_agreement(an: Analysis, hotspots: pl.DataFrame | None = None, *, top: float = 0.1) -> pl.DataFrame:
    """How much the two references agree, per direction (display pattern):

    * ``line_level_per_m``: line-level reference, obs / passage / m (running segments, 6-23 h);
    * ``evening_ref_per_m``: median over running segments of the evening reference per metre;
    * ``spearman_day``: Spearman correlation, over running segments, of the day-band (6-21 h) excess
      per passage vs the evening and vs the line level (1 = same ranking of where vehicles linger);
      ``spearman_day_all``: same over every segment (stop zones included);
    * ``top_overlap``: share of the top ``top`` running segments by excess vs the evening that are
      also in the top ``top`` by excess vs the line level;
    * with ``hotspots`` (``hotspots.hotspots`` output): counts per criterion ("peak" = vs the evening,
      "persistent" = vs the line level, "both") and ``hotspots_both_share`` = n_both / n_hotspots.
    """
    rows = []
    day_col = BAND_HOUR["day"]
    for dirid, dd in an.dirs.items():
        if len(dd.days) == 0:
            continue
        ks = dd.pattern_segs[dd.display_uid]
        est = an.estimate(dirid)
        c = int(np.flatnonzero(dd.cols == day_col)[0])
        xe = est["excess_per_passage"][0][c][ks]
        xl = est["excess_line_per_passage"][0][c][ks]
        run = dd.seg_zone[ks] == "running"
        ln = dd.seg_len[ks]
        with np.errstate(divide="ignore", invalid="ignore"):
            ev = np.nanmedian(np.where(run, est["ref"][0][ks] / ln, np.nan)) if run.any() else np.nan
        ok = run & np.isfinite(xe) & np.isfinite(xl)
        n_top = max(1, int(round(top * ok.sum())))
        if ok.sum() >= 3 and np.ptp(xe[ok]) > 0 and np.ptp(xl[ok]) > 0:
            ie = np.flatnonzero(ok)[np.argsort(-xe[ok])[:n_top]]
            il = np.flatnonzero(ok)[np.argsort(-xl[ok])[:n_top]]
            ov = len(set(ie.tolist()) & set(il.tolist())) / n_top
        else:
            ov = float("nan")
        r = {"direction_id": int(dirid), "line_level_per_m": float(est["line_level"][0]), "evening_ref_per_m": float(ev),
             "spearman_day": _spearman(np.where(run, xe, np.nan), np.where(run, xl, np.nan)),
             "spearman_day_all": _spearman(xe, xl), "top_overlap": float(ov)}
        if hotspots is not None:
            h = hotspots.filter(pl.col("direction_id") == dirid)
            crit = h["criterion"].to_list() if "criterion" in h.columns else ["peak"] * h.height
            n = len(crit)
            r.update({"n_hotspots": n, "n_peak": crit.count("peak"), "n_persistent": crit.count("persistent"),
                      "n_both": crit.count("both"), "hotspots_both_share": crit.count("both") / n if n else float("nan")})
        rows.append(r)
    return pl.DataFrame(rows)
