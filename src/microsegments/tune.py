"""Segment-length selection and sensitivity analysis.

``segment_length`` computes, for every candidate length L (segments from ``segment_fn(L, 0.0)``):

* ``cv_deviance``: leave-one-day-out Poisson deviance. The rate per segment and hour (obs per covered
  hour) is fitted on the other days, spread uniformly over the segment, and scored against the held-out
  day's counts on a fine grid (``fine_m``, default 5 m) -- so every L is scored on the same data.
  ``cv_se`` is the standard error of the paired per-day difference to the best L.
* ``ss_cost``: Shimazaki-Shinomoto histogram cost (2 k̄ - v) / (n Δ)^2 on day-band counts.
* ``reliability``: split-half (weekday-stratified, ``n_splits`` random splits) Pearson correlation of
  the hour (6-20) x segment matrix of obs per covered hour per metre, Spearman-Brown corrected.
* ``snr``: signal variance / noise variance of that matrix (noise from the half difference).
* ``loc_spread_m``: median over the top-10 hotspots of the bootstrap SD of the peak position.
* ``jaccard``: split-half overlap of hotspot footprints (±1 segment tolerance).
* ``link_total_dev``: max relative deviation of per-link observation totals vs the first L (0 when the
  segmentation partitions every link).

Rule: the smallest L with reliability ≥ 0.8, localisation spread ≤ 30 m (or no hotspot) and CV
deviance within one standard error of the minimum. Fallback: the CV minimum.

``sensitivity`` re-runs the analysis over phase offsets, gap caps (needs ``coverage_fn``), stop zones
(incl. one estimated from the stop-aligned occupancy profile), references and passage sources, and
compares each variant with the baseline (Spearman on a 5 m profile, hotspot Jaccard, link total change,
share of the baseline hotspots found again).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import polars as pl

from .aggregate import count, link_totals
from .config import Params, Quality, Select
from .hotspots import BinStats, bin_stats, hotspots, stretches
from .metrics import BAND_HOUR, Analysis, analyse

SegmentFn = Callable[..., pl.DataFrame]


@dataclass
class TuneResult:
    table: pl.DataFrame                  # one row per L: the criteria
    recommended: float
    rule: str
    per_day_deviance: dict[float, np.ndarray] = field(repr=False, default_factory=dict)


@dataclass
class SensitivityResult:
    table: pl.DataFrame                  # tidy: variant, param, value, x_m, obs_per_h_m (day band profile)
    summary: pl.DataFrame                # one row per variant vs baseline
    hotspots: pl.DataFrame               # baseline top hotspots: boot share, phase count, stable
    stable: bool
    baseline_hotspots: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)


# ------------------------------------------------------------------------------------ helpers
def _split_weights(dow: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(2, D) indicator weights of a weekday-stratified half split."""
    W = np.zeros((2, len(dow)), np.float32)
    for w in np.unique(dow):
        g = rng.permutation(np.flatnonzero(dow == w))
        h = len(g) // 2 + (rng.random() < 0.5 if len(g) % 2 else 0)
        W[0, g[:h]] = 1
        W[1, g[h:]] = 1
    return W


def _fine_segments(segments: pl.DataFrame, fine_m: float) -> pl.DataFrame:
    """Fixed fine bins per (pattern, link), keyed by link_key so they are physical."""
    links = (segments.group_by("pattern_uid", "link_idx", "link_key")
             .agg((pl.col("offset_m") + pl.col("len_m")).max().alias("L")))
    rows = []
    for u, li, lk, L in links.iter_rows():
        e = np.arange(0.0, L, fine_m)
        e = np.append(e, L)
        if len(e) > 2 and e[-1] - e[-2] < 1.0:
            e = np.delete(e, -2)
        for j, (a, b) in enumerate(zip(e[:-1], e[1:])):
            rows.append((u, li, lk, f"{lk}/f{j}", j, float(a), float(b - a)))
    df = pl.DataFrame(rows, schema={"pattern_uid": pl.Utf8, "link_idx": pl.Int16, "link_key": pl.Utf8,
                                    "seg_key": pl.Utf8, "k": pl.Int32, "offset_m": pl.Float64, "len_m": pl.Float64},
                      orient="row").sort("pattern_uid", "link_idx", "k")
    return df.with_columns(pl.int_range(0, pl.len()).over("pattern_uid").cast(pl.Int32).alias("seg_idx"))


def _overlap(an: Analysis, dirid: int, fine: pl.DataFrame):
    """Fine bins of the display pattern and their overlap with the L segments: (fine_keys, link idx,
    k1, w1, k2, w2) with k = K indices and w = overlap length (m)."""
    dd = an.dirs[dirid]
    u = dd.display_uid
    ks = dd.pattern_segs[u]
    segs = an.segments.filter(pl.col("pattern_uid") == u).sort("seg_idx")
    s_link = segs["link_idx"].to_numpy()
    lo = segs["offset_m"].to_numpy()
    hi = lo + segs["len_m"].to_numpy()
    f = fine.filter(pl.col("pattern_uid") == u).sort("seg_idx")
    fl, fa = f["link_idx"].to_numpy(), f["offset_m"].to_numpy()
    fb = fa + f["len_m"].to_numpy()
    n = len(f)
    k1 = np.zeros(n, np.int64); k2 = np.zeros(n, np.int64)
    w1 = np.zeros(n); w2 = np.zeros(n)
    for li in np.unique(fl):
        m = np.flatnonzero(fl == li)
        sm = np.flatnonzero(s_link == li)
        if len(sm) == 0:
            continue
        a, b = fa[m], fb[m]
        j1 = np.clip(np.searchsorted(hi[sm], a, side="right"), 0, len(sm) - 1)
        j2 = np.clip(np.searchsorted(hi[sm], b, side="left"), 0, len(sm) - 1)
        k1[m], k2[m] = ks[sm[j1]], ks[sm[j2]]
        w1[m] = np.minimum(b, hi[sm][j1]) - a
        w2[m] = np.where(j2 != j1, b - np.maximum(a, lo[sm][j2]), 0.0)
    lpos = {k: i for i, k in enumerate(dd.link_keys)}
    flink = np.array([lpos.get(k, -1) for k in f["link_key"].to_list()])
    return np.array(f["seg_key"].to_list(), dtype=object), flink, k1, w1, k2, w2


def _fine_counts(cube_f: pl.DataFrame, dd, keys: np.ndarray) -> np.ndarray:
    D, H, J = len(dd.days), len(dd.hours), len(keys)
    days_df = pl.DataFrame({"service_date": dd.days.astype("datetime64[D]"), "_d": np.arange(D)})
    hours_df = pl.DataFrame({"hour": dd.hours.astype(np.int8), "_h": np.arange(H)})
    keys_df = pl.DataFrame({"seg_key": keys.tolist(), "_k": np.arange(J)}, schema={"seg_key": pl.Utf8, "_k": pl.Int64})
    c = (cube_f.select("service_date", pl.col("hour").cast(pl.Int8), "seg_key", "obs")
         .join(days_df, on="service_date").join(hours_df, on="hour").join(keys_df, on="seg_key"))
    flat = (c["_d"].to_numpy() * H + c["_h"].to_numpy()) * J + c["_k"].to_numpy()
    return np.bincount(flat, weights=c["obs"].to_numpy().astype(float), minlength=D * H * J).reshape(D, H, J)


def _loo_deviance(an: Analysis, dirid: int, Y: np.ndarray, ov) -> np.ndarray:
    """Per-day leave-one-out Poisson deviance on the fine grid."""
    dd = an.dirs[dirid]
    X = dd.stacked()
    hc = dd.hour_cols
    NM = X["NM"][:, hc].astype(float)
    CM = X["CM"][:, hc].astype(float)
    _, flink, k1, w1, k2, w2 = ov
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (NM.sum(0)[None] - NM) / (CM.sum(0)[None] - CM)       # (D,H,K) LOO obs per covered hour
    R = np.nan_to_num(R) / dd.seg_len[None, None, :]
    lam = R[:, :, k1] * w1 + R[:, :, k2] * w2                      # per covered hour, per fine bin
    Mf = dd.E[:, :, None] & dd.LV[:, flink][:, None, :]
    mu = np.maximum(lam * dd.C[:, :, None], 1e-6)
    y = Y
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return 2 * np.where(Mf, t, 0.0).sum((1, 2))


def _split_half(an: Analysis, dirid: int, n_splits: int, rng) -> tuple[float, float]:
    dd = an.dirs[dirid]
    ks = dd.pattern_segs[dd.display_uid]
    hc = dd.hour_cols
    hrs = dd.cols[hc]
    ci = hc[(hrs >= 6) & (hrs <= 20)]
    rel, snr = [], []
    for _ in range(n_splits):
        W = _split_weights(dd.dow, rng)
        e = an.estimate(dirid, W)["obs_per_h"][:, ci][:, :, ks] / dd.seg_len[ks][None, None, :]
        a, b = e[0].reshape(-1), e[1].reshape(-1)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        a, b = a[ok], b[ok]
        r = np.corrcoef(a, b)[0, 1]
        rel.append(2 * r / (1 + r))
        noise = np.var(a - b) / 4
        tot = np.var((a + b) / 2)
        snr.append((tot - noise) / noise if noise > 0 else np.inf)
    return (float(np.mean(rel)) if rel else np.nan, float(np.median(snr)) if snr else np.nan)


def _ss_cost(an: Analysis, dirid: int) -> float:
    dd = an.dirs[dirid]
    ks = dd.pattern_segs[dd.display_uid]
    col = int(np.flatnonzero(dd.cols == BAND_HOUR["day"])[0])
    k = dd.stacked()["NM"][:, col][:, ks].sum(0).astype(float)
    ln = dd.seg_len[ks]
    delta = float(ln.mean())
    k = k * delta / ln
    n = max(len(dd.days), 1)
    return float((2 * k.mean() - k.var()) / (n * delta) ** 2)


def _peak_cols(st: BinStats, j0: int, j1: int, hours) -> int:
    hcols = np.flatnonzero(st.cols >= 0)
    hrs = st.cols[hcols]
    hidx = np.array([hcols[np.flatnonzero(hrs == h)[0]] for h in sorted(hours)])
    sx = np.nansum(st.x[hidx][:, j0:j1 + 1], axis=1)
    return int(hidx[int(np.nanargmax(sx))])


def _loc_spread(st: BinStats, strs, top: int = 10) -> float:
    if not strs:
        return np.nan
    mid = st.x0 + st.length / 2
    order = sorted(strs, key=lambda s: -np.nansum(st.x[_peak_cols(st, s[0], s[1], s[2]), s[0]:s[1] + 1]))[:top]
    out = []
    for j0, j1, hours in order:
        pc = _peak_cols(st, j0, j1, hours)
        a, b = max(j0 - 2, 0), min(j1 + 2, len(mid) - 1)
        d = np.nan_to_num(st.draws[:, pc, a:b + 1], nan=-np.inf)
        pk = mid[a + d.argmax(1)]
        out.append(float(np.std(pk)))
    return float(np.median(out))


def _tol_jaccard(A: np.ndarray, B: np.ndarray, tol: int = 1) -> float:
    """Jaccard-like overlap of two boolean masks with a ±tol tolerance."""
    if not A.any() and not B.any():
        return 1.0
    if not A.any() or not B.any():
        return 0.0
    def dil(M):
        out = M.copy()
        for t in range(1, tol + 1):
            out[t:] |= M[:-t]
            out[:-t] |= M[t:]
        return out
    return float(((A & dil(B)).sum() + (B & dil(A)).sum()) / (A.sum() + B.sum()))


def _footprint(st: BinStats, strs) -> np.ndarray:
    m = np.zeros(len(st.ks), bool)
    for j0, j1, _ in strs:
        m[j0:j1 + 1] = True
    return m


# ------------------------------------------------------------------------------------ segment length
def segment_length(placed: pl.DataFrame, segment_fn: SegmentFn,
                   lengths=(10, 15, 20, 30, 40, 50, 75, 100), *, coverage: pl.DataFrame | None = None,
                   passages: pl.DataFrame | None = None, pattern_days: pl.DataFrame | None = None,
                   select: Select | None = None, params: Params | None = None, quality: Quality | None = None,
                   per_vehicle: bool = False, fine_m: float = 5.0, n_splits: int = 20, B: int = 100,
                   n_jaccard: int = 3, seed: int = 0, min_reliability: float = 0.8, max_spread_m: float = 30.0,
                   hotspot_kw: dict | None = None) -> TuneResult:
    params = params or Params()
    hotspot_kw = hotspot_kw or {}
    rng = np.random.default_rng(seed)
    if per_vehicle:  # thin once, then every L counts the same rows
        from .aggregate import thin
        placed = thin(placed.filter(pl.col("count").fill_null(True)), params.tick_s)
    rows, devs, base_tot = [], {}, None
    fine = cube_f = None
    for L in lengths:
        segs = segment_fn(L, 0.0)
        cube = count(placed, segs, params.tick_s, per_vehicle=False)
        an = analyse(cube, coverage, passages, segs, pattern_days, select, params, quality)
        if fine is None:
            fine = _fine_segments(segs, fine_m)
            cube_f = count(placed, fine, params.tick_s, per_vehicle=False)
        dev = 0.0
        rel, snr, spread, jac, ss = [], [], [], [], []
        for dirid, dd in an.dirs.items():
            if len(dd.days) < 2:
                continue
            ov = _overlap(an, dirid, fine)
            Y = _fine_counts(cube_f, dd, ov[0])
            dev = dev + _loo_deviance(an, dirid, Y, ov)
            r, s = _split_half(an, dirid, n_splits, rng)
            rel.append(r); snr.append(s); ss.append(_ss_cost(an, dirid))
            st = bin_stats(an, dirid, B=B, seed=seed, **hotspot_kw)
            strs = stretches(st)
            spread.append(_loc_spread(st, strs))
            js = []
            for _ in range(n_jaccard):
                W = _split_weights(dd.dow, rng)
                if (W.sum(1) < 2).any():
                    continue
                fp = []
                for h in range(2):
                    sub = an.subset(dd.days[W[h] > 0].astype(object))
                    sth = bin_stats(sub, dirid, B=max(B // 2, 20), seed=seed, **hotspot_kw)
                    fp.append(_footprint(sth, stretches(sth)))
                js.append(_tol_jaccard(fp[0], fp[1]))
            jac.append(float(np.mean(js)) if js else np.nan)
        devs[L] = np.atleast_1d(dev)
        tot = link_totals(cube, segs, by=())
        if base_tot is None:
            base_tot = tot
            ltd = 0.0
        else:
            j = base_tot.join(tot, on=["pattern_uid", "link_key"], how="full", suffix="_b").fill_null(0)
            a, b = j["obs"].to_numpy().astype(float), j["obs_b"].to_numpy().astype(float)
            ltd = float(np.max(np.abs(b - a) / np.maximum(a, 1))) if len(a) else 0.0
        n_seg = sum(len(dd.pattern_segs[dd.display_uid]) for dd in an.dirs.values())
        rows.append({"L": float(L), "n_segments": n_seg, "cv_deviance": float(np.mean(devs[L])),
                     "ss_cost": float(np.mean(ss)) if ss else np.nan,
                     "reliability": float(np.nanmean(rel)) if rel else np.nan,
                     "snr": float(np.nanmean(snr)) if snr else np.nan,
                     "loc_spread_m": float(np.nanmedian(spread)) if np.isfinite(spread).any() else np.nan,
                     "jaccard": float(np.nanmean(jac)) if jac else np.nan, "link_total_dev": ltd})
    # 1-SE rule with paired per-day differences to the best L
    Ls = [float(L) for L in lengths]
    cv = np.array([r["cv_deviance"] for r in rows])
    best = int(np.nanargmin(cv))
    db = devs[lengths[best]]
    se = []
    for L in lengths:
        d = devs[L] - db
        se.append(float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0)
    for r, s in zip(rows, se):
        r["cv_se"] = s
    table = pl.DataFrame(rows)
    ok = []
    for r in rows:
        cv_ok = r["cv_deviance"] - cv[best] <= max(r["cv_se"], 1e-12)
        rel_ok = r["reliability"] >= min_reliability
        sp_ok = not np.isfinite(r["loc_spread_m"]) or r["loc_spread_m"] <= max_spread_m
        ok.append(bool(cv_ok and rel_ok and sp_ok))
    table = table.with_columns(pl.Series("eligible", ok, dtype=pl.Boolean))
    if any(ok):
        rec = Ls[ok.index(True)]
        rule = (f"smallest L with reliability >= {min_reliability}, localisation spread <= {max_spread_m} m "
                f"and CV deviance within 1 SE of the minimum (L={Ls[best]:g})")
    else:
        rec = Ls[best]
        rule = "no L meets every criterion: CV deviance minimum"
    return TuneResult(table=table, recommended=rec, rule=rule, per_day_deviance=devs)


# ------------------------------------------------------------------------------------ sensitivity
def stop_zone_from_profile(placed: pl.DataFrame, segments: pl.DataFrame, max_m: float = 150.0, step: float = 5.0,
                           factor: float = 1.5) -> tuple[float, float]:
    """(up, down) metres around stops where the stop-aligned occupancy per metre exceeds ``factor`` x the
    baseline (median over 100-``max_m`` m from the stop), contiguous from the stop."""
    links = (segments.group_by("pattern_uid", "link_idx")
             .agg((pl.col("offset_m") + pl.col("len_m")).max().alias("_L")))
    df = (placed.filter(pl.col("count").fill_null(True)).select("pattern_uid", pl.col("link_idx").cast(pl.Int16), "pos_m")
          .join(links.with_columns(pl.col("link_idx").cast(pl.Int16)), on=["pattern_uid", "link_idx"]))
    pos = df["pos_m"].to_numpy().astype(float)
    Ln = df["_L"].to_numpy()
    lens = links["_L"].to_numpy()
    edges = np.arange(0, max_m + step, step)
    exposure = np.array([(lens >= e + step).sum() for e in edges[:-1]], float)  # links covering that bin
    out = []
    for d in (Ln - pos, pos):                       # up: before the next stop; down: after the stop
        h, _ = np.histogram(d, edges)
        dens = h / np.maximum(exposure, 1)
        base = np.median(dens[edges[:-1] >= 100]) if (edges[:-1] >= 100).any() else np.median(dens)
        above = dens > factor * base
        n = int(np.argmin(above)) if not above.all() else len(above)
        out.append(float(n * step))
    return out[0], out[1]


def _profile(an: Analysis, grid_m: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for dirid, dd in an.dirs.items():
        u = dd.display_uid
        ks = dd.pattern_segs[u]
        col = int(np.flatnonzero(dd.cols == BAND_HOUR["day"])[0])
        v = an.estimate(dirid)["obs_per_h"][0][col][ks] / dd.seg_len[ks]
        x0 = dd.pattern_x0[u]
        tot = x0[-1] + dd.seg_len[ks][-1]
        g = np.arange(grid_m / 2, tot, grid_m)
        j = np.clip(np.searchsorted(x0, g, side="right") - 1, 0, len(ks) - 1)
        out[dirid] = (g, v[j])
    return out


def _link_rates(an: Analysis) -> dict[tuple[int, str], float]:
    out = {}
    for dirid, dd in an.dirs.items():
        col = int(np.flatnonzero(dd.cols == BAND_HOUR["day"])[0])
        e = an.estimate(dirid)
        oph = np.nan_to_num(e["obs_per_h"][0][col])
        ks = dd.pattern_segs[dd.display_uid]
        for li in np.unique(dd.seg_link[ks]):
            out[(dirid, dd.link_keys[li])] = float(oph[ks][dd.seg_link[ks] == li].sum())
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pl.Series(a[ok]).rank("average").to_numpy()
    rb = pl.Series(b[ok]).rank("average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _hs_mask(hs: pl.DataFrame, dirid: int, g: np.ndarray, tol: float) -> np.ndarray:
    m = np.zeros(len(g), bool)
    for x0, x1 in hs.filter(pl.col("direction_id") == dirid).select("x0_m", "x1_m").iter_rows():
        m |= (g >= x0 - tol) & (g <= x1 + tol)
    return m


def _overlaps(row, hs: pl.DataFrame, tol: float) -> bool:
    h = hs.filter(pl.col("direction_id") == row["direction_id"])
    return bool(((h["x0_m"] <= row["x1_m"] + tol) & (h["x1_m"] >= row["x0_m"] - tol)).any()) if h.height else False


def sensitivity(placed: pl.DataFrame, segment_fn: SegmentFn, length: float = 30.0, *,
                coverage: pl.DataFrame | None = None, coverage_fn: Callable[[float], pl.DataFrame] | None = None,
                passages: pl.DataFrame | None = None, pattern_days: pl.DataFrame | None = None,
                select: Select | None = None, params: Params | None = None, quality: Quality | None = None,
                per_vehicle: bool = False, phases=(0.0, 0.25, 0.5, 0.75), gap_caps=(30.0, 40.0, 60.0, float("inf")),
                stop_zones=((15.0, 15.0), (30.0, 60.0), (50.0, 100.0), "data"),
                references=(("evening", (20, 23)), ("evening", (21, 23)), ("freeflow", (20, 23)), ("evening", (5, 6))),
                passage_sources=("n_feed", "n_events", "n"), B: int | None = None, top: int = 10,
                grid_m: float = 5.0, seed: int = 0, hotspot_kw: dict | None = None) -> SensitivityResult:
    """Sensitivity suite around a baseline (``length``, phase 0, ``params``). ``phases`` are fractions of
    L. Gap-cap variants need ``coverage_fn(gap_cap_s) -> COVERAGE``; stop-zone variants need a
    ``segment_fn`` accepting a ``stop_zone`` keyword. Variants that cannot run are skipped."""
    params = params or Params()
    hotspot_kw = {"seed": seed, **(hotspot_kw or {})}
    if B is not None:
        hotspot_kw["B"] = B
    tol = float(length)
    accepts_sz = "stop_zone" in inspect.signature(segment_fn).parameters or any(
        p.kind == p.VAR_KEYWORD for p in inspect.signature(segment_fn).parameters.values())
    cubes: dict = {}

    def run(segs_key, segs, cov, prm, pcol):
        if segs_key not in cubes:
            cubes[segs_key] = count(placed, segs, prm.tick_s, per_vehicle)
        an = analyse(cubes[segs_key], cov, passages, segs, pattern_days, select, prm, quality, passage_col=pcol)
        return an

    base_segs = segment_fn(length, 0.0)
    base = run(("phase", 0.0), base_segs, coverage, params, "n")
    st_base = {d: bin_stats(base, d, **hotspot_kw) for d in base.dirs if len(base.dirs[d].days)}
    hs_base = hotspots(base, stats=st_base)
    prof_base = _profile(base, grid_m)
    lr_base = _link_rates(base)

    variants = []  # (param, value, thunk)
    for f in phases:
        if f == 0:
            continue
        variants.append(("phase", f * length, lambda f=f: run(("phase", f), segment_fn(length, f * length), coverage, params, "n")))
    if coverage_fn is not None:
        for g in gap_caps:
            variants.append(("gap_cap_s", g, lambda g=g: run(("phase", 0.0), base_segs, coverage_fn(g), replace(params, gap_cap_s=g), "n")))
    if accepts_sz:
        for sz in stop_zones:
            if sz == "data":
                szv = stop_zone_from_profile(placed, base_segs)
                label = f"data {szv[0]:g}/{szv[1]:g}"
            else:
                szv, label = tuple(sz), f"{sz[0]:g}/{sz[1]:g}"
            variants.append(("stop_zone", label, lambda szv=szv: run(("sz", szv), segment_fn(length, 0.0, stop_zone=szv),
                                                                       coverage, replace(params, stop_zone=szv), "n")))
    for kind, hrs in references:
        label = "q20" if kind == "freeflow" else f"{hrs[0]}-{hrs[1]}"
        variants.append(("reference", label, lambda kind=kind, hrs=hrs: base.with_params(
            replace(params, reference=kind, reference_hours=tuple(hrs)))))
    if passages is not None:
        for src in passage_sources:
            if src not in passages.columns or passages[src].null_count() == passages.height:
                continue
            variants.append(("passages", src, lambda src=src: run(("phase", 0.0), base_segs, coverage, params, src)))

    tidy = []
    for dirid, (g, v) in prof_base.items():
        tidy.append(pl.DataFrame({"variant": "baseline", "param": "baseline", "value": "", "direction_id": dirid,
                                  "x_m": g, "obs_per_h_m": v}))
    summ, phase_hs = [], [hs_base]
    for param, value, thunk in variants:
        an = thunk()
        stv = {d: bin_stats(an, d, **hotspot_kw) for d in an.dirs if len(an.dirs[d].days)}
        hs = hotspots(an, stats=stv)
        if param == "phase":
            phase_hs.append(hs)
        prof = _profile(an, grid_m)
        name = f"{param}={value}"
        sp, jc = [], []
        for dirid, (g, v) in prof.items():
            tidy.append(pl.DataFrame({"variant": name, "param": param, "value": str(value), "direction_id": dirid,
                                      "x_m": g, "obs_per_h_m": v}))
            if dirid in prof_base:
                gb, vb = prof_base[dirid]
                n = min(len(gb), len(g))
                # project both profiles on the baseline segments, then rank-correlate
                dd = base.dirs[dirid]
                jb = np.clip(np.searchsorted(dd.pattern_x0[dd.display_uid], gb[:n], side="right") - 1, 0, None)
                cnt = np.bincount(jb)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pa = np.bincount(jb, np.nan_to_num(vb[:n])) / cnt
                    pv = np.bincount(jb, np.nan_to_num(v[:n])) / cnt
                sp.append(_spearman(pa, pv))
                jc.append(_tol_jaccard(_hs_mask(hs_base, dirid, gb[:n], 0), _hs_mask(hs, dirid, gb[:n], 0),
                                       tol=max(int(round(tol / grid_m)), 1)))
        lr = _link_rates(an)
        ch = [abs(lr[k] / lr_base[k] - 1) for k in lr_base if k in lr and lr_base[k] > 0]
        top_b = hs_base.filter(pl.col("rank") <= top)
        share = float(np.mean([_overlaps(r, hs, tol) for r in top_b.iter_rows(named=True)])) if top_b.height else 1.0
        s = {"variant": name, "param": param, "value": str(value),
             "spearman": float(np.nanmin(sp)) if sp else np.nan, "jaccard": float(np.nanmin(jc)) if jc else np.nan,
             "link_total_change": float(max(ch)) if ch else 0.0, "hotspots_stable_share": share, "n_hotspots": hs.height}
        s["stable"] = bool(s["spearman"] >= 0.9 and s["jaccard"] >= 0.7 and s["link_total_change"] <= 0.05)
        summ.append(s)

    # top hotspots: share of bootstrap draws above threshold, number of phases where found
    hrows = []
    for r in hs_base.filter(pl.col("rank") <= top).iter_rows(named=True):
        st = st_base[r["direction_id"]]
        j0 = int(np.flatnonzero(st.seg_idx == r["from_seg"])[0])
        j1 = int(np.flatnonzero(st.seg_idx == r["to_seg"])[0])
        pc = _peak_cols(st, j0, j1, r["hours"])
        boot = float((np.nansum(st.draws[:, pc, j0:j1 + 1], axis=1) > st.theta).mean())
        nph = sum(_overlaps(r, h, tol) for h in phase_hs)
        hrows.append({"direction_id": r["direction_id"], "rank": r["rank"], "x0_m": r["x0_m"], "x1_m": r["x1_m"],
                      "boot_share": boot, "n_phases": nph, "n_phase_variants": len(phase_hs),
                      "stable": bool(boot >= 0.8 and nph >= min(3, len(phase_hs)))})
    hdf = pl.DataFrame(hrows, schema={"direction_id": pl.Int8, "rank": pl.Int16, "x0_m": pl.Float64, "x1_m": pl.Float64,
                                      "boot_share": pl.Float64, "n_phases": pl.Int64, "n_phase_variants": pl.Int64,
                                      "stable": pl.Boolean})
    summary = pl.DataFrame(summ) if summ else pl.DataFrame()
    stable = bool((summary["stable"].all() if summary.height else True) and (hdf["stable"].all() if hdf.height else True))
    return SensitivityResult(table=pl.concat(tidy, how="vertical_relaxed"), summary=summary, hotspots=hdf,
                             stable=stable, baseline_hotspots=hs_base)
