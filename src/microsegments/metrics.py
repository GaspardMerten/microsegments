"""Cube -> indicators per segment x hour (and per band), robust to missing data and GTFS versions.

Notation (per direction): days d (included service days), hours h, segments k (keyed by the stable
``seg_key``), links l (``link_key``).

* ``N[d,h,k]`` observations, ``C[d,h]`` coverage (share of the hour the feed was alive),
  ``P[d,h,l]`` passages. With ``params.tick_from_coverage``, each observation of hour (d,h) weighs
  ``covered_s / n_snapshots / tick_s``: the real time a distinct poll stands for (polls slower than
  ``tick_s``, repeated snapshots), so ``O * tick_s`` stays seconds.
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
-4 (evening 20-24), -5 (midday 10-16), all [start, end). The page offers ``PAGE_BANDS``; "day" stays
internal (hotspots, tune, reference agreement). The dense arrays and versions live in ``dirdata``, the
estimator in ``estimate`` (both re-exported here).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl

from .config import Params, Quality, Select
# DirData, the versions and the estimator live in dirdata / estimate; their names stay importable from here
from .dirdata import BAND_HOUR, BANDS, DISPLAY_MIN_DAYS, DISPLAY_MIN_SHARE, DirData, _build_dir, _versions  # noqa: F401
from .dirdata import version_groups  # noqa: F401
from .estimate import ALPHA, LINE_HOURS, _estimate, _ratio, line_level  # noqa: F401
from .schema import RESULT, Flag, conform
from .stats import spearman

PAGE_BANDS = ("am", "midday", "pm", "evening")    # the page's "Période" buttons, in this order
DAY_HOURS = (6, 21)
EXCLUDING = Flag.LOW_COVERAGE | Flag.FROZEN | Flag.LINE_ABSENT | Flag.DEVIATION | Flag.VEHICLE_RATIO
_REASONS = ((Flag.FROZEN, "frozen"), (Flag.LINE_ABSENT, "line_absent"), (Flag.DEVIATION, "deviation"),
            (Flag.VEHICLE_RATIO, "vehicle_ratio"))


# ------------------------------------------------------------------------------------ containers
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
            obs.append(_r1(NM[m].sum(0)))
            cov.append(np.round(CL[m].sum(0), 3).tolist())
            p.append(np.round(PL[m].sum(0), 2).tolist())
            days.append(dd.LV[m][:, ls].sum(0).astype(int).tolist())
            n.append(NL[m].sum(0).astype(int).tolist())
        return {"pattern_uid": uid, "hours": list(self.hours), "seg_key": dd.seg_keys[ks].tolist(),
                "link_key": dd.link_keys[ls].tolist(), "obs": obs, "cov": cov, "p": p, "days": days, "n": n}


def _r1(a: np.ndarray) -> list:
    """Weighted counts to 1 decimal for the contract, whole values written as ints. Line 7, Feb-May 2025: the
    contract grows by 23 % gzipped (2 decimals: +60 %) over whole counts, whose rounding error matters in
    stretches with few reports."""
    r = np.round(a, 1)
    return [[int(x) if x == int(x) else x for x in row] for row in r.tolist()]


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
    dates = _dates(cube, coverage, pattern_days, select)
    user_ex = {dt.date.fromisoformat(s) for s in select.exclude_dates}
    day_pos = {d: i for i, d in enumerate(dates)}
    covA, flA, wA = _hour_arrays(coverage, dates, day_pos, params)
    ex = np.isnan(covA) | (np.nan_to_num(covA) < quality.min_hour_coverage) | ((flA & EXCLUDING) != 0)
    included, excluded_days = _day_rule(dates, covA, flA, ex, user_ex, quality)
    day_hours = _day_hours(dates, covA, flA, ex, set(included))
    mains = _mains(pattern_days, segments, included, select)
    hours_out = list(range(select.hours[0], select.hours[1]))
    internal = set(hours_out) | set(range(*params.reference_hours)) | set(range(DAY_HOURS[0], 23))
    if params.reference == "freeflow":
        internal |= set(range(5, 24))
    hours = np.array(sorted(internal), dtype=np.int64)
    cols = np.concatenate([hours, np.array(list(BAND_HOUR.values()), dtype=np.int64)])
    seg = segments.select("pattern_uid", "seg_idx", "link_idx", "link_key", "seg_key", "x0_m", "len_m", "zone")
    dirs: dict[int, DirData] = {}
    for (dirid,), md in mains.group_by(["direction_id"], maintain_order=True):
        dd = _build_dir(int(dirid), md.sort("service_date"), seg, cube, passages, passage_col, covA, ex, day_pos, hours, cols, wA)
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


def _dates(cube, coverage, pattern_days, select: Select) -> list[dt.date]:
    """The day universe: days of the cube and the coverage (and, with a date range, of the patterns), within the
    range and on the selected weekdays."""
    dates = set(cube["service_date"].unique().to_list())
    if coverage is not None:
        dates |= set(coverage["service_date"].unique().to_list())
    rng = select.date_range()
    if rng is not None:
        if pattern_days is not None:
            dates |= set(pattern_days["service_date"].unique().to_list())
        dates = {d for d in dates if rng[0] <= d <= rng[1]}
    return sorted(d for d in dates if d.weekday() in set(select.weekdays))


def _hour_arrays(coverage, dates, day_pos, params: Params):
    """(days, 28 h) coverage (NaN: no row), flags and report weight (tick_from_coverage: real seconds per poll
    / tick_s, else 1)."""
    nd = len(dates)
    covA = np.full((nd, 28), np.nan)
    flA = np.zeros((nd, 28), np.int64)
    wA = np.ones((nd, 28))
    if coverage is None:
        covA[:] = 1.0
    elif nd:
        cv = coverage.filter(pl.col("service_date").is_in(dates) & pl.col("hour").is_between(0, 27))
        di = np.array([day_pos[d] for d in cv["service_date"].to_list()], dtype=np.int64)
        hh = cv["hour"].to_numpy().astype(np.int64)
        covA[di, hh] = np.clip(cv["coverage"].fill_null(0).to_numpy(), 0, 1)
        flA[di, hh] = cv["flags"].fill_null(0).to_numpy().astype(np.int64)
        if params.tick_from_coverage and {"covered_s", "n_snapshots"} <= set(cv.columns):
            n = cv["n_snapshots"].fill_null(0).to_numpy().astype(float)
            sec = np.divide(cv["covered_s"].fill_null(0).to_numpy().astype(float), n, out=np.full(n.shape, np.nan),
                            where=n > 0)
            wA[di, hh] = np.where(np.isfinite(sec) & (sec > 0), sec / params.tick_s, 1.0)
    return covA, flA, wA


def _day_rule(dates, covA, flA, ex, user_ex, quality: Quality) -> tuple[list[dt.date], pl.DataFrame]:
    """Included days, and the excluded ones with their reason (user's, or too many excluded 6-21 h day-hours)."""
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
    return included, pl.DataFrame(excluded, schema={"service_date": pl.Date, "reason": pl.Utf8}, orient="row")


def _day_hours(dates, covA, flA, ex, included: set) -> pl.DataFrame:
    dh = [(d, h, float(covA[i, h]), int(flA[i, h]), d in included and not ex[i, h])
          for i, d in enumerate(dates) for h in range(28) if not np.isnan(covA[i, h])]
    return pl.DataFrame(dh, schema={"service_date": pl.Date, "hour": pl.Int8, "coverage": pl.Float32,
                                    "flags": pl.UInt16, "included": pl.Boolean}, orient="row")


def _mains(pattern_days, segments, included, select: Select) -> pl.DataFrame:
    """The main pattern of each included day and direction (most trips, then uid); without ``pattern_days``
    every pattern of ``segments`` is its own direction, main every day."""
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
    return mains


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
             "spearman_day": spearman(np.where(run, xe, np.nan), np.where(run, xl, np.nan)),
             "spearman_day_all": spearman(xe, xl), "top_overlap": float(ov)}
        if hotspots is not None:
            h = hotspots.filter(pl.col("direction_id") == dirid)
            crit = h["criterion"].to_list() if "criterion" in h.columns else ["peak"] * h.height
            n = len(crit)
            r.update({"n_hotspots": n, "n_peak": crit.count("peak"), "n_persistent": crit.count("persistent"),
                      "n_both": crit.count("both"), "hotspots_both_share": crit.count("both") / n if n else float("nan")})
        rows.append(r)
    return pl.DataFrame(rows)
