"""Sensitivity suite (``sensitivity``): the analysis re-run over phases, gap caps, stop zones, references,
passage sources and segment lengths, each variant compared with the baseline (see the package docstring)."""
from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Callable

import numpy as np
import polars as pl

from ..aggregate import count
from ..config import Params, Quality, Select
from ..hotspots import bin_stats, hotspots
from ..metrics import BAND_HOUR, Analysis, analyse
from ..stats import spearman
from .common import SegmentFn, SensitivityResult, _peak_cols, _tol_jaccard


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
                passage_sources=("n_feed", "n_events", "n"), lengths=(15.0, 30.0, 60.0), B: int | None = None,
                top: int = 10, grid_m: float = 5.0, seed: int = 0, hotspot_kw: dict | None = None) -> SensitivityResult:
    """Sensitivity suite around a baseline (``length``, phase 0, ``params``). ``phases`` are fractions of
    L. Gap-cap variants need ``coverage_fn(gap_cap_s) -> COVERAGE``; stop-zone variants need a
    ``segment_fn`` accepting a ``stop_zone`` keyword. ``lengths``: other segment lengths compared with
    the baseline (``length`` itself is skipped; profiles are compared on the same 5 m grid, hotspots
    with a tolerance of the baseline length). Variants that cannot run are skipped."""
    params = params or Params()
    if per_vehicle:  # grid once, then every variant counts the same rows
        from ..aggregate import per_vehicle_grid
        placed = per_vehicle_grid(placed, params.tick_s, params.per_vehicle)
    hotspot_kw = {"seed": seed, **(hotspot_kw or {})}
    if B is not None:
        hotspot_kw["B"] = B
    tol = float(length)
    cubes: dict = {}

    def run(segs_key, segs, cov, prm, pcol):
        if segs_key not in cubes:
            cubes[segs_key] = count(placed, segs, prm.tick_s, False)
        return analyse(cubes[segs_key], cov, passages, segs, pattern_days, select, prm, quality, passage_col=pcol)

    base_segs = segment_fn(length, 0.0)
    base = run(("phase", 0.0), base_segs, coverage, params, "n")
    st_base = {d: bin_stats(base, d, **hotspot_kw) for d in base.dirs if len(base.dirs[d].days)}
    hs_base = hotspots(base, stats=st_base)
    ref = {"an": base, "hs": hs_base, "prof": _profile(base, grid_m), "lr": _link_rates(base)}
    variants = _variants(run, segment_fn, placed, base, base_segs, length, coverage, coverage_fn, passages, params,
                         phases, gap_caps, stop_zones, references, passage_sources, lengths)
    tidy = [_tidy("baseline", "baseline", "", dirid, g, v) for dirid, (g, v) in ref["prof"].items()]
    summ, phase_hs = [], [hs_base]
    for param, value, thunk in variants:
        an = thunk()
        hs = hotspots(an, stats={d: bin_stats(an, d, **hotspot_kw) for d in an.dirs if len(an.dirs[d].days)})
        if param == "phase":
            phase_hs.append(hs)
        s, frames = _vs_baseline(ref, an, hs, param, value, grid_m, tol, top)
        tidy += frames
        summ.append(s)
    hdf = _hotspot_stability(hs_base, st_base, phase_hs, top, tol)
    summary = pl.DataFrame(summ) if summ else pl.DataFrame()
    stable = bool((summary["stable"].all() if summary.height else True) and (hdf["stable"].all() if hdf.height else True))
    return SensitivityResult(table=pl.concat(tidy, how="vertical_relaxed"), summary=summary, hotspots=hdf,
                             stable=stable, baseline_hotspots=hs_base, n_days=len(base.days))


def _variants(run, segment_fn, placed, base, base_segs, length, coverage, coverage_fn, passages, params,
              phases, gap_caps, stop_zones, references, passage_sources, lengths) -> list[tuple]:
    """The variants that can run, as (param, value, thunk -> Analysis)."""
    sig = inspect.signature(segment_fn).parameters
    accepts_sz = "stop_zone" in sig or any(p.kind == p.VAR_KEYWORD for p in sig.values())
    variants = []
    for f in phases:
        if f == 0:
            continue
        variants.append(("phase", f * length, lambda f=f: run(("phase", f), segment_fn(length, f * length), coverage, params, "n")))
    for L in lengths or ():
        if float(L) == float(length):
            continue
        variants.append(("segment_m", float(L), lambda L=float(L): run(("L", L), segment_fn(L, 0.0), coverage,
                                                                       replace(params, segment_m=L), "n")))
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
    return variants


def _tidy(variant, param, value, dirid, g, v) -> pl.DataFrame:
    return pl.DataFrame({"variant": variant, "param": param, "value": value, "direction_id": dirid,
                         "x_m": g, "obs_per_h_m": v})


def _vs_baseline(ref: dict, an: Analysis, hs: pl.DataFrame, param, value, grid_m, tol, top) -> tuple[dict, list]:
    """One variant against the baseline: worst Spearman of the day profiles (projected on the baseline
    segments), worst hotspot Jaccard, largest link-total change, share of the baseline top hotspots found again."""
    name = f"{param}={value}"
    sp, jc, frames = [], [], []
    for dirid, (g, v) in _profile(an, grid_m).items():
        frames.append(_tidy(name, param, str(value), dirid, g, v))
        if dirid in ref["prof"]:
            gb, vb = ref["prof"][dirid]
            n = min(len(gb), len(g))
            dd = ref["an"].dirs[dirid]
            jb = np.clip(np.searchsorted(dd.pattern_x0[dd.display_uid], gb[:n], side="right") - 1, 0, None)
            cnt = np.bincount(jb)
            with np.errstate(invalid="ignore", divide="ignore"):
                pa = np.bincount(jb, np.nan_to_num(vb[:n])) / cnt
                pv = np.bincount(jb, np.nan_to_num(v[:n])) / cnt
            sp.append(spearman(pa, pv))
            jc.append(_tol_jaccard(_hs_mask(ref["hs"], dirid, gb[:n], 0), _hs_mask(hs, dirid, gb[:n], 0),
                                   tol=max(int(round(tol / grid_m)), 1)))
    lr, lr_base = _link_rates(an), ref["lr"]
    ch = [abs(lr[k] / lr_base[k] - 1) for k in lr_base if k in lr and lr_base[k] > 0]
    top_b = ref["hs"].filter(pl.col("rank") <= top)
    share = float(np.mean([_overlaps(r, hs, tol) for r in top_b.iter_rows(named=True)])) if top_b.height else 1.0
    s = {"variant": name, "param": param, "value": str(value),
         "spearman": float(np.nanmin(sp)) if sp else np.nan, "jaccard": float(np.nanmin(jc)) if jc else np.nan,
         "link_total_change": float(max(ch)) if ch else 0.0, "hotspots_stable_share": share, "n_hotspots": hs.height}
    s["stable"] = bool(s["spearman"] >= 0.9 and s["jaccard"] >= 0.7 and s["link_total_change"] <= 0.05)
    return s, frames


def _hotspot_stability(hs_base, st_base, phase_hs, top: int, tol: float) -> pl.DataFrame:
    """Baseline top hotspots: share of bootstrap draws above threshold, number of phases where found again."""
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
    return pl.DataFrame(hrows, schema={"direction_id": pl.Int8, "rank": pl.Int16, "x0_m": pl.Float64, "x1_m": pl.Float64,
                                       "boot_share": pl.Float64, "n_phases": pl.Int64, "n_phase_variants": pl.Int64,
                                       "stable": pl.Boolean})
