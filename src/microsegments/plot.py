"""Matplotlib figures (extra ``[plot]``): hour x micro-segment matrix, profile along the line, segment
map, segment-length tuning curves and sensitivity summary.

Every function accepts an :class:`~microsegments.metrics.Analysis` or a pipeline ``RunResult`` (which
also brings stop names and hotspots), returns a ``Figure`` and draws into ``ax`` when given.
Colours follow the HTML page: one fixed scale for every figure (``microsegments.scale``), never computed
from the data. Each value is brought to seconds per passage per 30 m of track and read on the STIB speed
ramp, green (fluid) -> red (slow). Profiles plot the value per 30 m (``obs_per_h_10m``: per 10 m) with a
fixed height, the top of the scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import warnings

from . import scale as S
from .metrics import BAND_HOUR, Analysis

METRICS = {
    "obs_per_h": "position reports per hour",
    "obs_per_h_10m": "position reports per hour per 10 m",
    "obs_per_passage": "time spent per vehicle, s",
    "excess_per_passage": "time lost vs the evening, s per vehicle",
    "excess_obs_per_h": "extra position reports per hour vs the evening",
    "log2_ratio": "log2 ratio vs the evening",
}
_ALIASES = {"oph": "obs_per_h", "oph10": "obs_per_h_10m", "opp": "obs_per_passage", "ex": "excess_per_passage",
            "excess": "excess_per_passage"}
_RAMP = list(S.RAMP)   # the STIB speed ramp (km/h stops), as on the page
NULL_COLOR = "#3a4560"
MASK_COLOR = "#18213a"     # masked stop zones: hatched (background + stripes), distinct from flat no-data grey
MASK_HATCH = "#5d6a88"
BG = "#0b1224"


def _plt():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install 'microsegments[plot]'") from e
    return matplotlib, plt


# ------------------------------------------------------------------------------------ colours
def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(c @ M1.T) @ M2.T


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    M2i = np.array([[1, 0.3963377774, 0.2158037573], [1, -0.1055613458, -0.0638541728], [1, -0.0894841775, -1.2914855480]])
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    c = ((lab @ M2i.T) ** 3) @ M1i.T
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(c, 0, 1)


def _ramp(v: np.ndarray) -> np.ndarray:
    xs = np.array([x for x, _ in _RAMP])
    labs = _srgb_to_oklab(np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for _, h in _RAMP]))
    v = np.clip(v, xs[0], xs[-1])
    return _oklab_to_srgb(np.stack([np.interp(v, xs, labs[:, k]) for k in range(3)], axis=-1))


def cmap(n: int = 256):
    """The page's colormap over T = seconds per passage per 30 m (green -> red); use with :func:`norm`."""
    matplotlib, _ = _plt()
    from matplotlib.colors import ListedColormap
    T = np.linspace(S.T_MIN, S.T_MAX, n)   # linear axis in T; colour = the km/h ramp at 108 / T
    cm = ListedColormap(_ramp(S.kmh(T)), name="microsegments")
    return cm.with_extremes(bad=NULL_COLOR) if hasattr(cm, "with_extremes") else cm


def norm():
    """The fixed norm of :func:`cmap`: linear axis over T from ``scale.T_MIN`` to ``scale.T_MAX``."""
    from matplotlib.colors import Normalize
    return Normalize(vmin=S.T_MIN, vmax=S.T_MAX, clip=True)


def _no_vmax(vmax):
    if vmax is not None:
        warnings.warn("vmax is ignored: the colour scale is fixed (microsegments.scale)", DeprecationWarning, stacklevel=3)


def _seconds(an: Analysis, m: str, v, len_m) -> np.ndarray:
    """T (seconds per passage per 30 m) of values ``v`` of metric ``m`` (NaN stays NaN)."""
    return S.to_seconds(m, np.asarray(v, dtype=float), np.asarray(len_m, dtype=float), an.params.tick_s)


def _colorbar(fig, ax, m: str, fraction: float):
    _, plt = _plt()
    sm = plt.cm.ScalarMappable(norm=norm(), cmap=cmap())
    cb = fig.colorbar(sm, ax=ax, fraction=fraction, pad=0.01, extend="both")
    lab = S.from_seconds(m, np.array(S.T_TICKS, dtype=float))
    cb.set_ticks(list(S.T_TICKS))
    kmh = S.kmh(np.array(S.T_TICKS, dtype=float))
    cb.set_ticklabels([("≥ " if i == len(lab) - 1 else "") + f"{x:g}"
                       + (f" s · {k:.2g} km/h" if m == "obs_per_passage" else "")   # time per vehicle: its speed
                       for i, (x, k) in enumerate(zip(np.round(lab, 1), kmh))])
    cb.minorticks_off()
    cb.set_label(METRICS[m] + (" (per 10 m)" if m == "obs_per_h_10m" else " (per 30 m)"), fontsize=8)
    return cb


# ------------------------------------------------------------------------------------ data access
def _unpack(obj) -> tuple[Analysis, pl.DataFrame | None, pl.DataFrame | None]:
    """(analysis, links, hotspots) from an Analysis or a RunResult."""
    if isinstance(obj, Analysis):
        return obj, None, None
    an = getattr(obj, "analysis", None)
    if an is None:
        raise TypeError(f"expected an Analysis or a RunResult, got {type(obj).__name__}")
    net = getattr(obj, "network", None)
    return an, (net.links if net is not None else None), getattr(obj, "hotspots", None)


def _metric(metric: str) -> str:
    m = _ALIASES.get(metric, metric)
    if m not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; one of {sorted(METRICS)}")
    return m


def _hour_code(hour: int | str | None, band: str | None = None) -> int:
    if band is not None:
        hour = band
    if hour is None:
        return BAND_HOUR["day"]
    if isinstance(hour, str):
        if hour not in BAND_HOUR:
            raise ValueError(f"unknown band {hour!r}; one of {sorted(BAND_HOUR)}")
        return BAND_HOUR[hour]
    return int(hour)


def _subset(an: Analysis, days) -> Analysis:
    """``days``: weekdays (ints 0 = Monday), or service dates."""
    if days is None:
        return an
    days = list(days)
    if days and all(isinstance(d, (int, np.integer)) for d in days):
        keep = [d for d in an.days if d.weekday() in set(int(x) for x in days)]
    else:
        keep = days
    return an.subset(keep)


def _frame(an: Analysis, direction_id: int, metric: str, pattern_uid: str | None = None) -> pl.DataFrame:
    """Result rows of one direction / pattern with a ``value`` column for ``metric``."""
    dd = an.dirs[direction_id]
    uid = pattern_uid or dd.display_uid
    r = an.result.filter((pl.col("direction_id") == direction_id) & (pl.col("pattern_uid") == uid))
    if metric == "obs_per_h_10m":
        v = pl.col("obs_per_h") * 10.0 / pl.col("len_m")
    else:
        v = pl.col(metric)
    return r.with_columns(v.cast(pl.Float64).alias("value"))


def _hatch_spans(ax, segs: pl.DataFrame, y0: float, y1: float, alpha: float = 1.0) -> None:
    """Masked stop zones: hatched rectangles over [x0, x0 + len] x [y0, y1]."""
    from matplotlib.patches import Rectangle
    for x0, ln in zip(segs["x0_m"].to_list(), segs["len_m"].to_list()):
        ax.add_patch(Rectangle((x0, min(y0, y1)), ln, abs(y1 - y0), facecolor=MASK_COLOR, edgecolor=MASK_HATCH,
                               hatch="////", lw=0, alpha=alpha, zorder=1.5))


def auto_vmax(analysis=None, metric: str = "obs_per_passage", stops: bool = True, q: float = 0.98) -> float:
    """Top of the fixed colour scale in the units of ``metric`` per 30 m (per 10 m for ``obs_per_h_10m``).
    It does not depend on the analysis (kept for compatibility; the arguments other than ``metric`` are
    ignored)."""
    return float(S.from_seconds(_metric(metric), S.T_MAX))


def _stops(an: Analysis, direction_id: int, links: pl.DataFrame | None, pattern_uid: str | None = None):
    """(x positions, names) of the stops of the pattern."""
    dd = an.dirs[direction_id]
    uid = pattern_uid or dd.display_uid
    if links is not None:
        lk = links.filter(pl.col("pattern_uid") == uid).sort("link_idx")
        if lk.height:
            s0 = lk["s0_m"].to_list()
            return (np.array(s0 + [s0[-1] + lk["len_m"][-1]]),
                    lk["from_name"].to_list() + [lk["to_name"][-1]])
    s = an.segments.filter(pl.col("pattern_uid") == uid).sort("seg_idx")
    g = s.group_by("link_idx", maintain_order=True).agg(pl.col("x0_m").min(), (pl.col("x0_m") + pl.col("len_m")).max().alias("x1"))
    xs = g["x0_m"].to_list() + [g["x1"][-1]] if g.height else [0.0]
    return np.array(xs), [f"#{i}" for i in range(len(xs))]


def _stop_ticks(ax, xs, names, max_labels: int = 40):
    keep, last = [], -np.inf
    span = (xs[-1] - xs[0]) or 1.0
    for x, n in zip(xs, names):
        if (x - last) / span >= 1.0 / max_labels:
            keep.append((x, n))
            last = x
    ax.set_xticks([x for x, _ in keep])
    ax.set_xticklabels([n for _, n in keep], rotation=40, ha="right", fontsize=8)


def _dir_label(an, direction_id, links, pattern_uid=None):
    xs, names = _stops(an, direction_id, links, pattern_uid)
    return f"{names[0]} → {names[-1]}" if names else f"direction {direction_id}"


# ------------------------------------------------------------------------------------ figures
def matrix(analysis, direction_id: int = 0, metric: str = "obs_per_passage", days=None, ax=None, *,
           vmax: float | None = None, stops: bool = True, pattern_uid: str | None = None,
           hotspots: pl.DataFrame | None = None, links: pl.DataFrame | None = None, title: str | None = None,
           colorbar: bool = True, figsize=(14, 6)):
    """Hour x micro-segment heatmap (columns as wide as the segments, stop names on x).

    ``days``: weekdays (0 = Monday) or dates to keep. Colours: the fixed scale of ``microsegments.scale``
    (``vmax`` is deprecated and ignored). ``stops=False``
    masks stop-zone segments. Hotspots (RunResult or ``hotspots=``) are outlined."""
    _, plt = _plt()
    an0, links0, hs0 = _unpack(analysis)
    links = links if links is not None else links0
    hotspots = hotspots if hotspots is not None else hs0
    m = _metric(metric)
    _no_vmax(vmax)
    an = _subset(an0, days)
    f = _frame(an, direction_id, m, pattern_uid).filter(pl.col("hour") >= 0)
    hours = sorted(an.hours)
    segs = f.select("seg_idx", "x0_m", "len_m", "zone").unique("seg_idx").sort("seg_idx")
    si = {s: i for i, s in enumerate(segs["seg_idx"].to_list())}
    Z = np.full((len(hours), segs.height), np.nan)
    hi = {h: i for i, h in enumerate(hours)}
    f = f.with_columns(pl.Series("T", _seconds(an, m, f["value"].fill_null(np.nan).to_numpy(), f["len_m"].to_numpy())))
    for h, s, v in f.select("hour", "seg_idx", "T").iter_rows():
        if h in hi and v is not None and np.isfinite(v):
            Z[hi[h], si[s]] = v
    x_edges = np.concatenate([segs["x0_m"].to_numpy(), [segs["x0_m"][-1] + segs["len_m"][-1]]]) if segs.height else np.array([0, 1])
    y_edges = np.arange(len(hours) + 1) - 0.5 + hours[0] if hours else np.array([0, 1])
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize, layout="constrained")
    ax.set_facecolor(BG)
    cm = cmap()
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(Z), cmap=cm, norm=norm(), shading="flat")
    if not stops and segs.height:
        _hatch_spans(ax, segs.filter(pl.col("zone") == "stop"), y_edges[0], y_edges[-1])
    xs, names = _stops(an, direction_id, links, pattern_uid)
    for x in xs:
        ax.axvline(x, color=BG, lw=1.2)
    if hotspots is not None and hotspots.height and (pattern_uid is None or pattern_uid == an.dirs[direction_id].display_uid):
        from matplotlib.patches import Rectangle
        for r in hotspots.filter(pl.col("direction_id") == direction_id).iter_rows(named=True):
            hs_h = sorted(int(h) for h in (r["hours"] or []))
            runs: list[list[int]] = []
            for h in hs_h:
                if runs and h == runs[-1][1] + 1:
                    runs[-1][1] = h
                else:
                    runs.append([h, h])
            for a, b in runs:
                ax.add_patch(Rectangle((r["x0_m"], a - 0.5), r["x1_m"] - r["x0_m"], b - a + 1, fill=False,
                                       ec="#ffe600", lw=1.2))
            ax.text(r["x1_m"], (hs_h[0] if hs_h else hours[0]) - 0.6, str(r["rank"]), color="#ffe600", fontsize=8,
                    ha="left", va="bottom")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[-1], y_edges[0])
    ax.set_yticks(hours)
    ax.set_yticklabels([f"{h} h" for h in hours], fontsize=8)
    _stop_ticks(ax, xs, names)
    ax.set_title(title or f"{_dir_label(an, direction_id, links, pattern_uid)} · {METRICS[m]}", fontsize=11, loc="left")
    if colorbar:
        _colorbar(fig, ax, m, 0.025)
    return fig


def profile(analysis, direction_id: int = 0, hour: int | str | None = None, *, band: str | None = None,
            metric: str = "obs_per_passage", days=None, ax=None, stops: bool = True, reference: bool = True,
            vmax: float | None = None, pattern_uid: str | None = None, links: pl.DataFrame | None = None,
            title: str | None = None, figsize=(14, 4)):
    """Bars along the line for one hour or band ("am", "pm", "day", "evening"), coloured like the matrix,
    with the evening value as a white line (``reference``)."""
    _, plt = _plt()
    an0, links0, _ = _unpack(analysis)
    links = links if links is not None else links0
    m = _metric(metric)
    _no_vmax(vmax)
    an = _subset(an0, days)
    code = _hour_code(hour, band)
    f = _frame(an, direction_id, m, pattern_uid).sort("seg_idx")
    top = auto_vmax(None, m)
    shown = lambda d: S.from_seconds(m, _seconds(an, m, d["value"].fill_null(np.nan).to_numpy(), d["len_m"].to_numpy()))  # noqa: E731
    cur = f.filter(pl.col("hour") == code)
    if not stops:
        cur = cur.filter(pl.col("zone") != "stop")
    T = _seconds(an, m, cur["value"].fill_null(np.nan).to_numpy(), cur["len_m"].to_numpy())
    v = S.from_seconds(m, T)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize, layout="constrained")
    colors = cmap()(norm()(np.nan_to_num(T, nan=S.T_MIN)))
    colors[np.isnan(T)] = (0, 0, 0, 0)
    ax.bar(cur["x0_m"].to_numpy(), np.nan_to_num(v), width=cur["len_m"].to_numpy() * 0.96, align="edge", color=colors)
    if not stops:
        _hatch_spans(ax, f.filter((pl.col("hour") == code) & (pl.col("zone") == "stop")), 0, top * 1.05, alpha=0.55)
    if reference and code != BAND_HOUR["evening"] and not m.startswith("excess") and m != "log2_ratio":
        ev = f.filter(pl.col("hour") == BAND_HOUR["evening"])
        if not stops:
            ev = ev.filter(pl.col("zone") != "stop")
        x0, ln, ve = ev["x0_m"].to_numpy(), ev["len_m"].to_numpy(), shown(ev)
        ax.hlines(ve, x0, x0 + ln, color="0.25", lw=1.3, label="evening")
    xs, names = _stops(an, direction_id, links, pattern_uid)
    for x in xs:
        ax.axvline(x, color="0.6", lw=0.6, alpha=0.6)
    _stop_ticks(ax, xs, names)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, top * 1.05)
    ax.set_ylabel(METRICS[m] + (" (per 10 m)" if m == "obs_per_h_10m" else " (per 30 m)"), fontsize=9)
    when = next((k for k, c in BAND_HOUR.items() if c == code), None) or f"{code} h"
    ax.set_title(title or f"{_dir_label(an, direction_id, links, pattern_uid)} · {when}", fontsize=11, loc="left")
    return fig


def map(analysis, direction_id: int = 0, hour: int | str | None = None, *, band: str | None = None,
        metric: str = "obs_per_passage", days=None, ax=None, stops: bool = True, vmax: float | None = None,
        hotspots: pl.DataFrame | None = None, pattern_uid: str | None = None, links: pl.DataFrame | None = None,
        title: str | None = None, figsize=(8, 8), linewidth: float = 4.0):
    """Segments coloured on lon / lat (no base map); hotspots outlined in yellow."""
    _, plt = _plt()
    from matplotlib.collections import LineCollection
    an0, links0, hs0 = _unpack(analysis)
    links = links if links is not None else links0
    hotspots = hotspots if hotspots is not None else hs0
    m = _metric(metric)
    _no_vmax(vmax)
    an = _subset(an0, days)
    code = _hour_code(hour, band)
    uid = pattern_uid or an.dirs[direction_id].display_uid
    f = _frame(an, direction_id, m, uid).filter(pl.col("hour") == code).select("seg_idx", "value", "zone")
    g = an.segments.filter(pl.col("pattern_uid") == uid).select("seg_idx", "geometry", "x0_m", "len_m").join(f, on="seg_idx", how="left").sort("seg_idx")
    lines = [np.asarray(c) if c else np.zeros((0, 2)) for c in g["geometry"].to_list()]
    if not any(len(c) for c in lines):
        raise ValueError("segments have no geometry (segment(..., geometry=True))")
    T = _seconds(an, m, g["value"].fill_null(np.nan).to_numpy(), g["len_m"].to_numpy())
    colors = cmap()(norm()(np.nan_to_num(T, nan=S.T_MIN)))
    colors[np.isnan(T)] = matplotlib_color(NULL_COLOR)
    masked = np.array(g["zone"].to_list()) == "stop" if not stops else np.zeros(len(lines), bool)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize, layout="constrained")
    if hotspots is not None and hotspots.height and uid == an.dirs[direction_id].display_uid:
        x0 = g["x0_m"].to_numpy()
        x1 = x0 + g["len_m"].to_numpy()
        for r in hotspots.filter(pl.col("direction_id") == direction_id).iter_rows(named=True):
            sel = np.flatnonzero((x1 > r["x0_m"] + 0.5) & (x0 < r["x1_m"] - 0.5))
            hl = [lines[i] for i in sel if len(lines[i])]
            if hl:
                ax.add_collection(LineCollection(hl, colors="#ffe600", linewidths=linewidth * 1.8, capstyle="butt", zorder=1))
                mid = hl[len(hl) // 2][len(hl[len(hl) // 2]) // 2]
                ax.annotate(str(r["rank"]), mid, xytext=(8, 8), textcoords="offset points", fontsize=8,
                            bbox=dict(boxstyle="circle", fc="#ffe600", ec="none"), zorder=4)
    keep = ~masked
    ax.add_collection(LineCollection([c for c, k in zip(lines, keep) if k], colors=colors[keep], linewidths=linewidth,
                                     capstyle="butt", zorder=2))
    if masked.any():   # masked stop zones: striped (dashes over a dark casing), unlike the flat no-data grey
        ml = [c for c, k in zip(lines, masked) if k]
        ax.add_collection(LineCollection(ml, colors=MASK_COLOR, linewidths=linewidth, capstyle="butt", zorder=2))
        ax.add_collection(LineCollection(ml, colors=MASK_HATCH, linewidths=linewidth, linestyles=(0, (1, 1.5)),
                                         capstyle="butt", zorder=2))
    xs, names = _stops(an, direction_id, links, uid)
    if links is not None:
        lk = links.filter(pl.col("pattern_uid") == uid).sort("link_idx")
        geos = [gm for gm in lk["geometry"].to_list() if gm]
        pts = [gm[0] for gm in geos] + ([geos[-1][-1]] if geos else [])
        if pts:
            P = np.asarray(pts)
            ax.scatter(P[:, 0], P[:, 1], s=14, c="#9aa4b8", edgecolors="k", linewidths=0.6, zorder=3)
            for (x, y), n in zip(P, names):
                ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    allc = np.concatenate([c for c in lines if len(c)])
    pad = 0.05 * max(np.ptp(allc[:, 0]), np.ptp(allc[:, 1]), 1e-3)
    ax.set_xlim(allc[:, 0].min() - pad, allc[:, 0].max() + pad)
    ax.set_ylim(allc[:, 1].min() - pad, allc[:, 1].max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(allc[:, 1].mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    when = next((k for k, c in BAND_HOUR.items() if c == code), None) or f"{code} h"
    ax.set_title(title or f"{_dir_label(an, direction_id, links, uid)} · {when} · {METRICS[m]}", fontsize=10, loc="left")
    _colorbar(fig, ax, m, 0.03)
    return fig


def matplotlib_color(c: str):
    from matplotlib.colors import to_rgba
    return to_rgba(c)


def tune_curves(tune_result, figsize=(12, 7)):
    """Criteria of ``tune.segment_length`` against L, recommended L marked."""
    _, plt = _plt()
    t = tune_result.table
    L = t["L"].to_numpy()
    panels = [("reliability_excess", "split-half reliability, time lost per vehicle (gate)"),
              ("loc_spread_m", "hotspot localisation spread (m, gate)"),
              ("cv_deviance", "CV Poisson deviance (diagnostic)"), ("reliability", "split-half reliability, obs/h"),
              ("jaccard", "split-half hotspot Jaccard (diagnostic)"), ("link_total_dev", "link total deviation")]
    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    for ax, (col, lab) in zip(axes.flat, panels):
        if col not in t.columns:
            ax.set_visible(False)
            continue
        y = t[col].to_numpy().astype(float)
        if col == "cv_deviance" and "cv_se" in t.columns:
            ax.errorbar(L, y, yerr=t["cv_se"].to_numpy().astype(float), marker="o", ms=4, capsize=3)
        else:
            ax.plot(L, y, marker="o", ms=4)
        ax.axvline(tune_result.recommended, color="#e5484d", lw=1, ls="--")
        if col in ("reliability_excess", "reliability"):
            ax.axhline(0.8, color="0.5", lw=0.8, ls=":")
        if col == "loc_spread_m":
            ax.axhline(30, color="0.5", lw=0.8, ls=":")
        ax.set_title(lab, fontsize=9, loc="left")
        ax.set_xlabel("segment length L (m)", fontsize=8)
        ax.set_xticks(L)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Recommended L = {tune_result.recommended:g} m · {tune_result.rule}", fontsize=10)
    return fig


def sensitivity(sens_result, direction_id: int | None = None, figsize=(13, 8)):
    """Left: per variant, Spearman / Jaccard / link total change vs the baseline with the stability
    thresholds. Right: day-band profile of the baseline with the min-max ribbon over variants."""
    _, plt = _plt()
    s = sens_result.summary
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    ax0 = fig.add_subplot(gs[:, 0])
    if s.height:
        names = s["variant"].to_list()
        y = np.arange(len(names))
        ax0.barh(y - 0.25, s["spearman"].fill_null(np.nan).to_numpy(), height=0.25, label="Spearman", color="#3fb950")
        ax0.barh(y, s["jaccard"].fill_null(np.nan).to_numpy(), height=0.25, label="hotspot Jaccard", color="#f0a13a")
        ax0.barh(y + 0.25, s["link_total_change"].to_numpy(), height=0.25, label="link total change", color="#e5484d")
        for x, c in ((0.9, "#3fb950"), (0.7, "#f0a13a"), (0.05, "#e5484d")):
            ax0.axvline(x, color=c, lw=0.8, ls=":")
        ax0.set_yticks(y)
        ax0.set_yticklabels([n + ("" if st else "  ✗") for n, st in zip(names, s["stable"].to_list())], fontsize=8)
        ax0.invert_yaxis()
        ax0.set_xlim(0, 1.05)
        ax0.legend(fontsize=7, loc="lower right")
    ax0.set_title(f"vs baseline · {'stable' if sens_result.stable else 'NOT stable'}", fontsize=10, loc="left")
    t = sens_result.table
    dirs = sorted(t["direction_id"].unique().to_list()) if t.height else []
    if direction_id is not None:
        dirs = [direction_id]
    for k, d in enumerate(dirs[:2]):
        ax = fig.add_subplot(gs[k, 1])
        td = t.filter(pl.col("direction_id") == d)
        base = td.filter(pl.col("variant") == "baseline").sort("x_m")
        var = td.filter(pl.col("variant") != "baseline")
        if var.height:
            g = var.group_by("x_m").agg(pl.col("obs_per_h_m").min().alias("lo"), pl.col("obs_per_h_m").max().alias("hi")).sort("x_m")
            ax.fill_between(g["x_m"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy(), color="#f0a13a", alpha=0.35,
                            step="mid", label="variants (min-max)", lw=0)
        ax.step(base["x_m"].to_numpy(), base["obs_per_h_m"].to_numpy(), where="mid", color="k", lw=0.9, label="baseline")
        ax.set_title(f"direction {d}: day profile (obs per hour per m)", fontsize=9, loc="left")
        ax.set_xlabel("position along the line (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    return fig


def save(fig, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure (format from the extension) and close it."""
    _, plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_all(analysis, out_dir: str | Path, metric: str = "obs_per_passage", stops: bool = True, dpi: int = 130) -> list[Path]:
    """One matrix PNG per direction (``matrix_dir<d>.png``) into ``out_dir``."""
    an, _, _ = _unpack(analysis)
    out = []
    for d in sorted(an.dirs):
        if len(an.dirs[d].days) == 0:
            continue
        out.append(save(matrix(analysis, d, metric, stops=stops), Path(out_dir) / f"matrix_dir{d}.png", dpi=dpi))
    return out


__all__ = ["METRICS", "auto_vmax", "cmap", "map", "norm", "matrix", "profile", "save", "save_all", "sensitivity", "tune_curves"]
