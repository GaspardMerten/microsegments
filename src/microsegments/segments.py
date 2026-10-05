"""Micro-segments: link-anchored cuts of each pattern.

Every link (stop A -> stop B) is cut on its own, so segment boundaries always fall on stops and a
segment never spans two links. Grids:

* ``"equal"`` (default): ``n = max(1, round(len / L))`` bins of ``L' = len / n``;
* ``"fixed"``: bins of exactly ``L`` from the link start, the last one shorter (the prototype's
  ``time30m_2025`` buckets);
* ``"adaptive"``: breaks given per pattern (see :func:`adaptive_breaks`).

``phase`` shifts the grid inside each link (metres, in units of the nominal ``L``: for ``"equal"`` the
shift is ``phase / L`` of a bin), leaving a remainder bin at both ends. Bins shorter than 1 m are
merged into their neighbour (the previous one, or the next one for the first bin).

``zone`` is ``"stop"`` when the segment overlaps ``[stop - up, stop + down]`` around any stop of the
pattern (``stop_zone = (up, down)``; *up* is before the stop, i.e. the end of the incoming link,
*down* after it, the start of the outgoing link), else ``"running"``.

``seg_key = f"{link_key}/{k}@{L:g}:{phase:g}"`` (+ ``"f"`` for the fixed grid, ``@a:<hash>`` for
adaptive breaks): stable for a given segmentation whatever the pattern / GTFS version the link
belongs to.
"""
from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import polars as pl

from . import schema
from .network.geometry import cut

SLIVER_M = 1.0


# ---------------------------------------------------------------- per-link grids
def link_edges(len_m: float, length: float = 30.0, phase: float = 0.0,
               grid: Literal["equal", "fixed"] = "equal") -> np.ndarray:
    """Bin edges ``[0, ..., len_m]`` of one link (see module docstring)."""
    len_m = float(len_m)
    if len_m <= 0:
        return np.array([0.0, max(len_m, 0.0)])
    if length <= 0:
        raise ValueError("length must be > 0")
    if grid == "equal":
        n = max(1, int(round(len_m / length)))
        step = len_m / n
        ph = (phase / length * step) % step
    elif grid == "fixed":
        step = float(length)
        ph = phase % step
    else:
        raise ValueError(f"unknown grid {grid!r}")
    k0 = 0 if ph > 0 else 1
    inner = ph + step * np.arange(k0, int(np.ceil(len_m / step)) + 2)
    inner = inner[(inner > 0) & (inner < len_m)]
    return _merge_slivers(np.concatenate([[0.0], inner, [len_m]]))


def _merge_slivers(e: np.ndarray, min_m: float = SLIVER_M) -> np.ndarray:
    e = list(e)
    while len(e) > 2:
        w = np.diff(e)
        small = np.flatnonzero(w < min_m)
        if not len(small):
            break
        i = int(small[-1])
        # a bin [e[i], e[i+1]] joins its previous neighbour (drop e[i]); the first one joins the next
        e.pop(i if i > 0 else 1)
    return np.asarray(e, dtype=np.float64)


def _hash_offsets(offsets) -> str:
    return hashlib.sha1(";".join(f"{x:.1f}" for x in offsets).encode()).hexdigest()[:6]


# ---------------------------------------------------------------- segmentation
def _links_table(network_or_links) -> pl.DataFrame:
    links = getattr(network_or_links, "links", network_or_links)
    if not isinstance(links, pl.DataFrame):
        raise TypeError("expected a Network or a LINK DataFrame")
    return links


def segment(network_or_links, length: float = 30.0, phase: float = 0.0,
            grid: Literal["equal", "fixed", "adaptive"] = "equal",
            stop_zone: tuple[float, float] = (30.0, 60.0), *,
            patterns: list[str] | None = None,
            breaks: dict[str, np.ndarray] | None = None,
            geometry: bool = True) -> pl.DataFrame:
    """SEGMENT table for every pattern of a Network (or a LINK table).

    ``breaks`` (grid ``"adaptive"``): ``{pattern_uid: break positions along the pattern, metres}``
    (from :func:`adaptive_breaks`); stops are always breaks. ``geometry=False`` skips cutting the
    polylines (faster, geometry column left empty)."""
    links = _links_table(network_or_links)
    if patterns is not None:
        links = links.filter(pl.col("pattern_uid").is_in(patterns))
    if grid == "adaptive" and breaks is None:
        raise ValueError("grid='adaptive' needs breaks={pattern_uid: positions}")
    up, down = float(stop_zone[0]), float(stop_zone[1])
    suffix = f"@{length:g}:{phase:g}" + ("f" if grid == "fixed" else "")
    cols = {k: [] for k in schema.SEGMENT}
    for (uid,), lk in links.sort("pattern_uid", "link_idx").group_by("pattern_uid", maintain_order=True):
        s0 = lk["s0_m"].to_numpy()
        ln = lk["len_m"].to_numpy()
        stops = np.concatenate([s0, [s0[-1] + ln[-1]]]) if len(lk) else np.zeros(0)
        bk = None if breaks is None else np.sort(np.asarray(breaks.get(uid, []), dtype=np.float64))
        seg_idx = 0
        for r, a, L_ in zip(lk.iter_rows(named=True), s0, ln):
            if grid == "adaptive":
                inner = bk[(bk > a) & (bk < a + L_)] - a
                e = _merge_slivers(np.concatenate([[0.0], inner, [L_]]))
                sfx = f"@a:{_hash_offsets(e)}"
            else:
                e = link_edges(L_, length, phase, grid)
                sfx = suffix
            lo, hi = e[:-1], e[1:]
            x0, x1 = a + lo, a + hi
            # overlap with [stop - up, stop + down] of any stop (positive length, or touching for 0-length bins)
            ov = (x0[:, None] < stops[None, :] + down) & (x1[:, None] > stops[None, :] - up)
            zone = np.where(ov.any(axis=1), "stop", "running")
            g = r["geometry"]
            for k in range(len(lo)):
                cols["pattern_uid"].append(uid)
                cols["seg_idx"].append(seg_idx)
                cols["link_idx"].append(r["link_idx"])
                cols["link_key"].append(r["link_key"])
                cols["seg_key"].append(f"{r['link_key']}/{k}{sfx}")
                cols["k"].append(k)
                cols["offset_m"].append(float(lo[k]))
                cols["x0_m"].append(float(x0[k]))
                cols["len_m"].append(float(hi[k] - lo[k]))
                cols["zone"].append(str(zone[k]))
                cols["geometry"].append(
                    cut(g, lo[k], hi[k], scale_to=L_).tolist() if geometry and g is not None and len(g) else None)
                seg_idx += 1
    return pl.DataFrame(cols, schema=schema.SEGMENT)


# ---------------------------------------------------------------- placing positions
def assign(pos_m, link_idx, segments: pl.DataFrame, pattern_uid: str | None = None) -> np.ndarray:
    """Segment index (``seg_idx``) of link-relative positions, vectorised.

    Bins are ``(lo, hi]``: a position equal to a bin's upper edge belongs to that bin, so a vehicle
    standing at stop B (pos = len of link A->B) counts in the last bin of A->B. A position of exactly
    0 (or below) belongs to the *previous* link's last segment (the vehicle is at the stop that link
    arrives at); on link 0 it goes to segment 0. Positions beyond the link length go to the link's
    last segment. NaN positions and unknown links give -1.

    ``segments``: SEGMENT rows of one pattern (or pass ``pattern_uid`` to pick one)."""
    seg = segments if pattern_uid is None else segments.filter(pl.col("pattern_uid") == pattern_uid)
    if seg["pattern_uid"].n_unique() > 1:
        raise ValueError("segments hold several patterns: pass pattern_uid")
    seg = seg.sort("seg_idx")
    pos = np.asarray(pos_m, dtype=np.float64)
    li = np.asarray(link_idx).astype(np.int64)
    pos, li = np.broadcast_arrays(pos, li)
    out = np.full(pos.shape, -1, dtype=np.int32)
    s_link = seg["link_idx"].to_numpy().astype(np.int64)
    s_idx = seg["seg_idx"].to_numpy().astype(np.int64)
    hi_all = (seg["offset_m"] + seg["len_m"]).to_numpy()
    starts = {}
    for L_ in np.unique(s_link):
        m = np.flatnonzero(s_link == L_)
        starts[int(L_)] = (m[0], m[-1])
    for L_ in np.unique(li):
        if int(L_) not in starts:
            continue
        a, b = starts[int(L_)]
        sel = (li == L_) & ~np.isnan(pos)
        p = pos[sel]
        hi = hi_all[a:b + 1].copy()
        hi[-1] = np.inf
        j = np.searchsorted(hi, p, side="left")
        res = s_idx[a + j]
        at0 = p <= 0
        if a > 0:
            res[at0] = s_idx[a - 1]
        else:
            res[at0] = s_idx[0]
        out[sel] = res
    return out


def assign_frame(df: pl.DataFrame, segments: pl.DataFrame, pos_col: str = "pos_m") -> pl.DataFrame:
    """Add ``seg_idx`` and ``seg_key`` to a frame with ``pattern_uid``, ``link_idx`` and ``pos_col``."""
    parts = []
    df = df.with_row_index("__row")
    for (uid,), g in df.group_by("pattern_uid"):
        s = segments.filter(pl.col("pattern_uid") == uid)
        idx = assign(g[pos_col].to_numpy(), g["link_idx"].to_numpy(), s) if len(s) else np.full(len(g), -1)
        parts.append(g.with_columns(seg_idx=pl.Series(idx, dtype=pl.Int32)))
    out = pl.concat(parts) if parts else df.with_columns(seg_idx=pl.lit(None, pl.Int32))
    keys = segments.select("pattern_uid", "seg_idx", "seg_key")
    return out.join(keys, on=["pattern_uid", "seg_idx"], how="left").sort("__row").drop("__row")


# ---------------------------------------------------------------- adaptive
def _poisson_cost(c: float, expo: float) -> float:
    return 0.0 if c <= 0 or expo <= 0 else -(c * np.log(c / expo) - c)


def adaptive_breaks(counts, link_bounds, penalty: float | None = None, *, bin_m: float = 10.0,
                    min_m: float = 20.0, max_m: float = 200.0) -> np.ndarray:
    """Data-driven break positions (metres along the pattern), stops forced as breaks.

    ``counts``: observations per ``bin_m`` bin along the pattern (bin i covers
    ``[i * bin_m, (i + 1) * bin_m)``); ``link_bounds``: stop positions (0 ... pattern length).
    Within each link, the optimal partition (optimal partitioning / PELT objective, exact O(n^2))
    minimises the Poisson negative log-likelihood of piecewise-constant rates plus ``penalty`` per
    segment (default ``log(total bins)``), with segment lengths in ``[min_m, max_m]`` (a link
    shorter than ``min_m`` stays whole). Returns sorted breaks including the stops."""
    c = np.asarray(counts, dtype=np.float64)
    S = np.asarray(link_bounds, dtype=np.float64)
    pen = float(np.log(max(len(c), 2))) if penalty is None else float(penalty)
    out = [float(S[0])]
    centres = (np.arange(len(c)) + 0.5) * bin_m
    for a, b in zip(S[:-1], S[1:]):
        if b - a <= min_m:
            out.append(float(b))
            continue
        # candidate cut positions: bin edges strictly inside the link, plus both stops
        inner = np.arange(np.floor(a / bin_m) + 1, np.ceil(b / bin_m)) * bin_m
        pos = np.concatenate([[a], inner[(inner > a) & (inner < b)], [b]])
        # counts between consecutive candidate positions (bins assigned by their centre)
        cc = np.array([c[(centres >= lo) & (centres < hi)].sum() for lo, hi in zip(pos[:-1], pos[1:])])
        cum = np.concatenate([[0.0], np.cumsum(cc)])
        n = len(pos)
        F = np.full(n, np.inf)
        F[0] = 0.0
        prev = np.zeros(n, dtype=np.int64)
        for j in range(1, n):
            for i in range(j - 1, -1, -1):
                w = pos[j] - pos[i]
                if w > max_m + 1e-9:
                    break
                if w < min_m - 1e-9 or not np.isfinite(F[i]):
                    continue
                v = F[i] + _poisson_cost(cum[j] - cum[i], w / bin_m) + pen
                if v < F[j]:
                    F[j], prev[j] = v, i
        if not np.isfinite(F[-1]):   # cannot satisfy the bounds: equal parts of at most max_m
            k = int(np.ceil((b - a) / max_m))
            out += list(a + (b - a) * np.arange(1, k + 1) / k)
            continue
        cuts, j = [], n - 1
        while j > 0:
            cuts.append(float(pos[j]))
            j = prev[j]
        out += cuts[::-1]
    return np.asarray(out)
