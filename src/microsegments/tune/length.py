"""Segment-length selection (``segment_length``): the criteria per candidate length and the rule (see the
package docstring)."""
from __future__ import annotations

import numpy as np
import polars as pl

from ..aggregate import count, link_totals
from ..config import Params, Quality, Select
from ..hotspots import BinStats, bin_stats, stretches
from ..metrics import BAND_HOUR, Analysis, analyse
from .common import SegmentFn, TuneResult, _peak_cols, _split_weights, _tol_jaccard


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


def _sb(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman-Brown corrected Pearson correlation of two half estimates (NaN pairs dropped)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    # a negative half-half correlation means no reliability; unclipped, r near -1 would explode the mean
    r = max(float(r), 0.0) if np.isfinite(r) else np.nan
    return float(2 * r / (1 + r)) if np.isfinite(r) else np.nan


def _split_half(an: Analysis, dirid: int, n_splits: int, rng) -> tuple[float, float, float]:
    """(reliability of obs/h/m on the hour x segment matrix, its SNR, reliability of the time lost per
    vehicle per metre on the day profile), over ``n_splits`` weekday-stratified half splits. The time
    lost needs >= 4 days (two per half), else NaN."""
    dd = an.dirs[dirid]
    ks = dd.pattern_segs[dd.display_uid]
    ln = dd.seg_len[ks]
    hc = dd.hour_cols
    hrs = dd.cols[hc]
    ci = hc[(hrs >= 6) & (hrs <= 20)]
    dcol = int(np.flatnonzero(dd.cols == BAND_HOUR["day"])[0])
    rel, snr, rex = [], [], []
    for _ in range(n_splits):
        W = _split_weights(dd.dow, rng)
        est = an.estimate(dirid, W)
        e = est["obs_per_h"][:, ci][:, :, ks] / ln[None, None, :]
        a, b = e[0].reshape(-1), e[1].reshape(-1)
        r = _sb(a, b)
        if np.isfinite(r):
            rel.append(r)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            noise = np.var(a - b) / 4
            tot = np.var((a + b) / 2)
            snr.append((tot - noise) / noise if noise > 0 else np.inf)
        if len(dd.days) >= 4:
            x = est["excess_per_passage"][:, dcol, ks] / ln[None, :]
            r = _sb(x[0], x[1])
            if np.isfinite(r):
                rex.append(r)
    return (float(np.mean(rel)) if rel else np.nan, float(np.median(snr)) if snr else np.nan,
            float(np.mean(rex)) if rex else np.nan)


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


def _footprint(st: BinStats, strs) -> np.ndarray:
    m = np.zeros(len(st.ks), bool)
    for j0, j1, _ in strs:
        m[j0:j1 + 1] = True
    return m


def segment_length(placed: pl.DataFrame, segment_fn: SegmentFn,
                   lengths=(10, 15, 20, 30, 40, 50, 75, 100), *, coverage: pl.DataFrame | None = None,
                   passages: pl.DataFrame | None = None, pattern_days: pl.DataFrame | None = None,
                   select: Select | None = None, params: Params | None = None, quality: Quality | None = None,
                   per_vehicle: bool = False, fine_m: float = 5.0, n_splits: int = 20, B: int = 100,
                   n_jaccard: int = 3, seed: int = 0, min_reliability: float = 0.8, max_spread_m: float = 30.0,
                   min_jaccard: float = 0.7,
                   hotspot_kw: dict | None = None) -> TuneResult:
    """Criteria for every candidate length (see the module docstring) and the recommended one: the
    smallest L whose split-half reliability of the time lost per vehicle (``reliability_excess``) is at
    least ``min_reliability`` and whose localisation spread is at most ``max_spread_m``; the default
    ``params.segment_m`` when none passes. ``min_jaccard`` only sets the reported ``jaccard_ok`` column
    (the hotspot Jaccard and the CV 1-SE rule are diagnostics, not gates)."""
    params = params or Params()
    hotspot_kw = hotspot_kw or {}
    if per_vehicle:  # grid once, then every L counts the same rows
        from ..aggregate import per_vehicle_grid
        placed = per_vehicle_grid(placed, params.tick_s, params.per_vehicle)
    rows, devs, base_tot = [], {}, None
    fine = cube_f = None
    for L in lengths:
        rng = np.random.default_rng(seed)   # every L is scored on the same day splits
        segs = segment_fn(L, 0.0)
        cube = count(placed, segs, params.tick_s, per_vehicle=False)
        an = analyse(cube, coverage, passages, segs, pattern_days, select, params, quality)
        if fine is None:
            fine = _fine_segments(segs, fine_m)
            cube_f = count(placed, fine, params.tick_s, per_vehicle=False)
        row, devs[L] = _score(an, fine, cube_f, rng, n_splits=n_splits, B=B, n_jaccard=n_jaccard, seed=seed,
                              hotspot_kw=hotspot_kw)
        tot = link_totals(cube, segs, by=())
        if base_tot is None:
            base_tot = tot
        rows.append({"L": float(L), **row, "link_total_dev": _total_dev(base_tot, tot)})
    table = _cv_columns(rows, devs, lengths, min_jaccard)
    rec, rule, ok = _recommend(rows, [float(L) for L in lengths], params, min_reliability, max_spread_m)
    return TuneResult(table=table.with_columns(pl.Series("eligible", ok, dtype=pl.Boolean)), recommended=rec,
                      rule=rule, per_day_deviance=devs)


def _score(an: Analysis, fine, cube_f, rng, *, n_splits, B, n_jaccard, seed, hotspot_kw) -> tuple[dict, np.ndarray]:
    """The criteria of one length (directions with >= 2 days), and its per-day CV deviance."""
    dev = 0.0
    rel, snr, rex, spread, jac, ss = [], [], [], [], [], []
    for dirid, dd in an.dirs.items():
        if len(dd.days) < 2:
            continue
        ov = _overlap(an, dirid, fine)
        dev = dev + _loo_deviance(an, dirid, _fine_counts(cube_f, dd, ov[0]), ov)
        r, s, x = _split_half(an, dirid, n_splits, rng)
        rel.append(r); snr.append(s); rex.append(x); ss.append(_ss_cost(an, dirid))
        st = bin_stats(an, dirid, B=B, seed=seed, **hotspot_kw)
        spread.append(_loc_spread(st, stretches(st)))
        jac.append(_split_jaccard(an, dirid, rng, n_jaccard, B, seed, hotspot_kw))
    dev = np.atleast_1d(dev)
    n_seg = sum(len(dd.pattern_segs[dd.display_uid]) for dd in an.dirs.values())
    return {"n_segments": n_seg, "cv_deviance": float(np.mean(dev)),
            "ss_cost": float(np.mean(ss)) if ss else np.nan,
            "reliability_excess": float(np.nanmean(rex)) if np.isfinite(rex).any() else np.nan,
            "reliability": float(np.nanmean(rel)) if rel else np.nan,
            "snr": float(np.nanmean(snr)) if snr else np.nan,
            "loc_spread_m": float(np.nanmedian(spread)) if np.isfinite(spread).any() else np.nan,
            "jaccard": float(np.nanmean(jac)) if jac else np.nan}, dev


def _split_jaccard(an: Analysis, dirid: int, rng, n_jaccard: int, B: int, seed: int, hotspot_kw: dict) -> float:
    """Mean tolerant overlap of the hotspot footprints found on two halves of the days (``n_jaccard`` splits)."""
    dd = an.dirs[dirid]
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
    return float(np.mean(js)) if js else np.nan


def _total_dev(base_tot: pl.DataFrame, tot: pl.DataFrame) -> float:
    """Max relative deviation of the per-link observation totals from the first length's."""
    if tot is base_tot:
        return 0.0
    j = base_tot.join(tot, on=["pattern_uid", "link_key"], how="full", suffix="_b").fill_null(0)
    a, b = j["obs"].to_numpy().astype(float), j["obs_b"].to_numpy().astype(float)
    return float(np.max(np.abs(b - a) / np.maximum(a, 1))) if len(a) else 0.0


def _cv_columns(rows: list[dict], devs: dict, lengths, min_jaccard: float) -> pl.DataFrame:
    """Adds ``cv_se`` (SE of the paired per-day difference to the best L) and the diagnostics ``cv_within_1se``
    (it keeps falling as L shrinks, so it always favours the finest grid) and ``jaccard_ok``."""
    cv = np.array([r["cv_deviance"] for r in rows])
    best = int(np.nanargmin(cv))
    db = devs[lengths[best]]
    se = []
    for L in lengths:
        d = devs[L] - db
        se.append(float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0)
    for r, s in zip(rows, se):
        r["cv_se"] = s
    for k, r in enumerate(rows):
        r["cv_within_1se"] = bool(cv[k] - cv[best] <= max(se[k], 1e-12))
        r["jaccard_ok"] = bool(not np.isfinite(r["jaccard"]) or r["jaccard"] >= min_jaccard)
    return pl.DataFrame(rows)


def _recommend(rows: list[dict], Ls: list[float], params: Params, min_reliability: float, max_spread_m: float):
    """Gate: the time lost per vehicle (what the map shows) must replicate between two halves of the days, and
    the hotspots must stay in place; the smallest such L keeps the finest resolution. (L, rule, eligible)."""
    ok = []
    for r in rows:
        rel_ok = np.isfinite(r["reliability_excess"]) and r["reliability_excess"] >= min_reliability
        sp_ok = not np.isfinite(r["loc_spread_m"]) or r["loc_spread_m"] <= max_spread_m
        ok.append(bool(rel_ok and sp_ok))
    if any(ok):
        rec = min(L for L, o in zip(Ls, ok) if o)
        return rec, (f"smallest L with split-half reliability of the time lost per vehicle >= {min_reliability} "
                     f"and localisation spread <= {max_spread_m:g} m"), ok
    rec = float(params.segment_m)
    return rec, (f"no length passed (split-half reliability of the time lost per vehicle >= {min_reliability}, "
                 f"localisation spread <= {max_spread_m:g} m): default segment_m={rec:g} kept"
                 + ("" if rec in Ls else " (not among the lengths tried)")), ok
