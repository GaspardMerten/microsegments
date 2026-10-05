"""Per-vehicle resampling of placed fixes onto a global time grid (one virtual poll every ``tick_s``).

A snapshot feed (STIB) sees every vehicle at each poll; a per-vehicle feed (GTFS-RT, AVL) reports
fixes at its own pace. To count comparably, each track is sampled at the grid instants
``k * tick_s`` lying between two of its consecutive fixes on the same pattern that are at most
``max_gap_s`` apart (or within half an interval before / after a run's first / last fix), taking
the nearer fix (position, status, flags). Interpolating positions would
spread dwells: between the last standing fix and the first moving one the vehicle mostly still
stands. Unlike keeping one fix per tick, this does not over-count where a track ends (a vehicle unseen
after its arrival) nor under-count where it starts.
"""
from __future__ import annotations

import numpy as np
import polars as pl



def _within(n: np.ndarray) -> np.ndarray:
    """0..n_i-1 for each group, concatenated."""
    first = np.repeat(np.cumsum(n) - n, n)
    return np.arange(n.sum()) - first


def resample(placed, tick_s: float = 20.0, max_gap_s: float = 90.0) -> pl.DataFrame:
    """Resample a PLACED frame (or Placed) with ``track_id`` onto the ``tick_s`` grid. Rows without
    ``track_id`` are returned unchanged. Extra columns are taken from the nearer fix."""
    frame = getattr(placed, "frame", placed)
    none = frame.filter(pl.col("track_id").is_null() | pl.col("s_m").is_null())
    d = (frame.filter(pl.col("track_id").is_not_null() & pl.col("s_m").is_not_null())
         .with_columns((pl.col("ts").dt.epoch("us") / 1e6).alias("_t"))
         .sort("track_id", "_t", maintain_order=True))
    if not d.height:
        return frame
    t = d["_t"].to_numpy()
    same = np.zeros(len(t), bool)
    same[:-1] = ((d["track_id"].to_numpy()[1:] == d["track_id"].to_numpy()[:-1])
                 & (d["pattern_uid"].to_numpy()[1:] == d["pattern_uid"].to_numpy()[:-1])
                 & (np.diff(t) <= max_gap_s) & (np.diff(t) > 0))
    i = np.flatnonzero(same)
    k0 = np.floor(t[i] / tick_s).astype(np.int64)           # grid instants in (t_i, t_i+1]
    k1 = np.floor(t[i + 1] / tick_s).astype(np.int64)
    n = k1 - k0
    pair = np.repeat(i, n)
    first = np.repeat(np.cumsum(n) - n, n)
    g = (np.repeat(k0, n) + 1 + (np.arange(n.sum()) - first)) * tick_s
    a, b = t[pair], t[pair + 1]
    w = (g - a) / (b - a)
    near = np.where(w < 0.5, pair, pair + 1)
    # run ends / starts: the end fix is also the nearest fix of the instants up to half its last
    # interval after it (and the start fix of those half its first interval before it)
    prev_ok = np.zeros(len(t), bool)
    prev_ok[1:] = same[:-1]
    e = np.flatnonzero(~same & prev_ok)
    he = (t[e] - t[e - 1]) / 2
    ne = np.floor((t[e] + he) / tick_s).astype(np.int64) - np.floor(t[e] / tick_s).astype(np.int64)
    st = np.flatnonzero(same & ~prev_ok)
    hs = (t[st + 1] - t[st]) / 2
    ns = np.ceil(t[st] / tick_s).astype(np.int64) - np.ceil((t[st] - hs) / tick_s).astype(np.int64)
    ge = (np.repeat(np.floor(t[e] / tick_s), ne) + 1 + _within(ne)) * tick_s
    gs = (np.repeat(np.ceil(t[st] / tick_s), ns) - 1 - _within(ns)) * tick_s
    near = np.concatenate([near, np.repeat(e, ne), np.repeat(st, ns)])
    g = np.concatenate([g, ge, gs])
    out = d[near].with_columns(
        pl.Series("_g", np.round(g * 1e6).astype(np.int64)).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
        .cast(d.schema["ts"]).alias("ts"))
    # local time of the grid instant: shift hour by whole hours crossed (service_date kept)
    out = out.with_columns(
        (pl.col("hour").cast(pl.Int32) + ((pl.col("ts").dt.epoch("s") // 3600) - (pl.col("_t") // 3600)).cast(pl.Int32))
        .cast(pl.Int8).alias("hour")).drop("_t")
    return pl.concat([out.select(frame.columns), none.select(frame.columns)], how="vertical_relaxed")
