"""Polyline geometry in lon/lat with metre arithmetic.

Port of ``speed55_2025.shape_slices`` (BrusselsTransitStuck), made robust:

* stops are projected *onto the segments* of the shape (not snapped to the nearest vertex), in a
  local equirectangular frame (metres);
* the projection is monotone along the shape and globally optimal (dynamic programme minimising the
  summed lateral distance), so loops, self-overlapping shapes (terminus loops, out-and-back
  branches) and stops far off the shape cannot make the order go backwards;
* a missing (or hopeless) shape falls back to the polyline through the stops.

Along-distances (``cum``, ``s``) are geodesic-accurate: segment lengths use haversine, the
projection fraction comes from the local frame (error well under a millimetre at city scale).
Coordinates are always ``(lon, lat)`` arrays of shape (n, 2).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

R_EARTH = 6371000.0
FALLBACK_MEDIAN_OFF_M = 200.0   # median stop-to-shape distance above which the shape is ignored
ROBUST_OFF_M = 30.0             # projection cost grows 100x slower beyond this lateral distance


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lon1, lat1, lon2, lat2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


class LocalFrame:
    """Equirectangular projection around a reference point: metres east / north."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.radians(1.0) * R_EARTH * np.cos(np.radians(self.lat0))
        self.ky = np.radians(1.0) * R_EARTH

    @classmethod
    def around(cls, *arrays) -> LocalFrame:
        pts = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1, 2) for a in arrays if len(a)])
        return cls(*pts.mean(axis=0))

    def xy(self, ll) -> np.ndarray:
        ll = np.asarray(ll, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([(ll[:, 0] - self.lon0) * self.kx, (ll[:, 1] - self.lat0) * self.ky])

    def ll(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def as_coords(coords) -> np.ndarray:
    a = np.asarray(coords, dtype=np.float64)
    return a.reshape(-1, 2)


def dedupe(coords) -> np.ndarray:
    """Drop consecutive duplicate vertices."""
    c = as_coords(coords)
    if len(c) < 2:
        return c
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(np.diff(c, axis=0) != 0, axis=1)
    return c[keep]


def cumlen(coords) -> np.ndarray:
    """Cumulative haversine length at each vertex (starts at 0)."""
    c = as_coords(coords)
    if len(c) < 2:
        return np.zeros(len(c))
    seg = haversine(c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1])
    return np.concatenate([[0.0], np.cumsum(seg)])


def length(coords) -> float:
    c = cumlen(coords)
    return float(c[-1]) if len(c) else 0.0


def interpolate(coords, s, cum: np.ndarray | None = None) -> np.ndarray:
    """Point(s) at along-distance ``s`` (metres, clipped to the polyline). Returns (k, 2)."""
    c = as_coords(coords)
    cum = cumlen(c) if cum is None else cum
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    if len(c) == 1:
        return np.repeat(c, len(s), axis=0)
    s = np.clip(s, 0.0, cum[-1])
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(c) - 2)
    seg = cum[i + 1] - cum[i]
    f = np.where(seg > 0, (s - cum[i]) / np.where(seg > 0, seg, 1.0), 0.0)
    f = np.clip(f, 0.0, 1.0)[:, None]
    return c[i] + f * (c[i + 1] - c[i])


def cut(coords, s0: float, s1: float, cum: np.ndarray | None = None, scale_to: float | None = None) -> np.ndarray:
    """Sub-polyline between along-distances ``s0 <= s1`` (metres).

    ``scale_to``: the metres ``s0``/``s1`` refer to a polyline of that nominal length (the polyline
    is stretched to it, as the prototype's ``cut``), e.g. a feed-rescaled link length."""
    c = as_coords(coords)
    cum = cumlen(c) if cum is None else cum
    if scale_to is not None and cum[-1] > 0:
        cum = cum * (scale_to / cum[-1])
    s0, s1 = float(s0), float(s1)
    if s1 < s0:
        s0, s1 = s1, s0
    a, b = interpolate(c, [s0, s1], cum)
    inner = c[(cum > s0) & (cum < s1)]
    return np.vstack([a[None], inner, b[None]])


def hausdorff_m(a, b) -> float:
    """Symmetric Hausdorff distance in metres between two lon/lat polylines (densified)."""
    import shapely

    a, b = as_coords(a), as_coords(b)
    fr = LocalFrame.around(a, b)
    ga, gb = (shapely.LineString(fr.xy(x)) if len(x) > 1 else shapely.Point(fr.xy(x)[0]) for x in (a, b))
    return float(shapely.hausdorff_distance(ga, gb, densify=0.25))


@dataclass
class Slices:
    """Result of :func:`shape_slices`."""
    s: np.ndarray              # along-shape position of each stop (metres, non-decreasing)
    off_m: np.ndarray          # lateral distance of each stop to the shape
    coords: list[np.ndarray]   # per link (n_stops - 1) polyline, endpoints = projected stops
    lengths: np.ndarray        # per link metres (= diff(s))
    shape: np.ndarray          # the polyline used (shape, or stops when it fell back)
    from_shape: bool


def project_monotone(stops_ll, shape_ll) -> tuple[np.ndarray, np.ndarray]:
    """Project stops onto a polyline keeping their order: returns (s, lateral distance), metres.

    Globally optimal monotone assignment: for stop j and shape segment i, cost d[j, i] (lateral
    distance of the projection); best[j, i] = d[j, i] + min_{i' <= i} best[j-1, i'] (prefix min),
    then backtrack. Stops projecting inside the same segment out of order are clamped so ``s`` is
    non-decreasing."""
    stops = as_coords(stops_ll)
    shape = as_coords(shape_ll)
    n = len(stops)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    cum = cumlen(shape)
    if len(shape) < 2:
        d = haversine(stops[:, 0], stops[:, 1], shape[0, 0], shape[0, 1]) if len(shape) else np.zeros(n)
        return np.zeros(n), d
    fr = LocalFrame.around(stops, shape)
    p, q = fr.xy(stops), fr.xy(shape)
    a, v = q[:-1], np.diff(q, axis=0)                       # (m, 2)
    vv = (v * v).sum(axis=1)
    rel = p[:, None, :] - a[None, :, :]                     # (n, m, 2)
    t = np.where(vv > 0, (rel * v[None]).sum(axis=2) / np.where(vv > 0, vv, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = np.hypot(*(rel - t[..., None] * v[None]).transpose(2, 0, 1))   # (n, m)
    seg = np.diff(cum)
    s_all = cum[:-1][None, :] + t * seg[None, :]

    # robust cost: a stop far off the shape barely pulls on the solution, so it cannot drag its
    # neighbours along; it still lands at its nearest feasible point
    cost = np.where(d < ROBUST_OFF_M, d, ROBUST_OFF_M + 0.01 * (d - ROBUST_OFF_M))
    m = d.shape[1]
    back = np.zeros((n, m), dtype=np.int64)
    best = cost[0].copy()
    for j in range(1, n):
        # running prefix minimum of best over i' <= i, with its argmin (earliest on ties)
        pm = np.minimum.accumulate(best)
        changed = np.concatenate([[True], best[1:] < pm[:-1]])
        arg = np.maximum.accumulate(np.where(changed, np.arange(m), 0))
        back[j] = arg
        best = cost[j] + pm
    i = np.empty(n, dtype=np.int64)
    # last stop: on ties prefer the latest segment (end of a loop shape)
    last = best.min()
    i[-1] = int(np.flatnonzero(best <= last + 1e-9)[-1])
    for j in range(n - 1, 0, -1):
        i[j - 1] = back[j, i[j]]
    s = s_all[np.arange(n), i].copy()
    off = d[np.arange(n), i]
    # order violations can only happen between stops sharing a segment: the stop closer to the
    # shape keeps its projection, the other one is moved onto it
    for j in range(1, n):
        if s[j] < s[j - 1]:
            if off[j] <= off[j - 1]:
                k = j - 1
                while k >= 0 and s[k] > s[j]:
                    s[k] = s[j]
                    k -= 1
            else:
                s[j] = s[j - 1]
    return s, off


def shape_slices(stops_ll, shape_ll=None) -> Slices:
    """Cut a shape at each stop (monotone projection); per-link coordinates and metres.

    ``shape_ll`` None / empty / too far from the stops (median offset > 200 m): the polyline
    through the stops is used instead."""
    stops = as_coords(stops_ll)
    shape = dedupe(shape_ll) if shape_ll is not None and len(shape_ll) else np.zeros((0, 2))
    from_shape = len(shape) >= 2
    if from_shape:
        s, off = project_monotone(stops, shape)
        if len(off) and float(np.median(off)) > FALLBACK_MEDIAN_OFF_M:
            warnings.warn(f"shape is {np.median(off):.0f} m from the stops (median); using stop polyline",
                          stacklevel=2)
            from_shape = False
    if not from_shape:
        shape = dedupe(stops) if len(stops) else stops
        cum = cumlen(shape)
        # stops are the vertices: their s is the cumulative length (repeated stops keep their s)
        s = cumlen(stops)
        off = np.zeros(len(stops))
    else:
        cum = cumlen(shape)
    coords = [cut(shape, s[k], s[k + 1], cum) for k in range(len(stops) - 1)]
    return Slices(s=s, off_m=off, coords=coords, lengths=np.diff(s), shape=shape, from_shape=from_shape)
