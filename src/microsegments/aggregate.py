"""Placed observations -> Cube: how many times a vehicle was observed in each micro-segment.

Only rows with ``count`` True (or null) are counted. A row is assigned to the segment whose
link-relative interval ``(offset_m, offset_m + len_m]`` contains its ``pos_m`` (lower bound open,
upper bound closed, as in the prototype: a vehicle standing at stop B is at the end of link A->B).
A row at ``pos_m <= 0`` of link i > 0 goes to the last segment of link i - 1, i.e. the end of the
link arriving at that stop. Positions beyond the link end are clipped to its last segment.

Per-vehicle feeds (GTFS-RT with irregular / dense fixes) are put on a ``tick_s`` grid so counts stay
comparable with a snapshot feed polled every ``tick_s``: by default ``locate.resample.resample`` (each
track sampled at the grid instants between its fixes, nearer fix; unbiased at track ends and for
sparse fixes), or ``method="thin"``: at most one observation per ``(track_id, floor(ts / tick_s))``
(the last fix of each tick; under-counts when fixes are sparser than the tick).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .schema import CUBE, conform

_SCALE = 1e7  # metres; composite sort key = group * _SCALE + position (links are far shorter)


def thin(placed: pl.DataFrame, tick_s: float = 20.0) -> pl.DataFrame:
    """Keep the last fix per (track_id, tick bucket). Rows without track_id are kept as they are."""
    tick_ms = max(int(round(tick_s * 1000)), 1)
    df = placed.with_columns((pl.col("ts").dt.epoch("ms") // tick_ms).alias("_tick"))
    has = df.filter(pl.col("track_id").is_not_null())
    none = df.filter(pl.col("track_id").is_null())
    has = has.sort("ts", maintain_order=True).unique(subset=["track_id", "_tick"], keep="last", maintain_order=True)
    return pl.concat([has, none], how="vertical").drop("_tick")


def assign(placed: pl.DataFrame, segments: pl.DataFrame) -> pl.DataFrame:
    """Add ``seg_idx`` and ``seg_key`` to placed rows (null when the row's link has no segment).

    Vectorised: one searchsorted over all (pattern, link) groups at once."""
    seg = (segments.select("pattern_uid", "link_idx", "seg_idx", "seg_key", "offset_m", "len_m")
           .with_columns(pl.col("link_idx").cast(pl.Int16))
           .sort("pattern_uid", "link_idx", "offset_m"))
    groups = (seg.group_by("pattern_uid", "link_idx", maintain_order=True)
              .agg(pl.len().alias("_n"))
              .with_row_index("_gid"))
    gn = groups["_n"].to_numpy().astype(np.int64)
    g_last = np.cumsum(gn) - 1
    g_first = g_last - gn + 1
    seg = seg.join(groups.select("pattern_uid", "link_idx", "_gid"), on=["pattern_uid", "link_idx"], how="left")
    seg_comp = seg["_gid"].to_numpy().astype(np.float64) * _SCALE + (seg["offset_m"] + seg["len_m"]).to_numpy()

    # pos <= 0 on link i > 0 belongs to the end of link i - 1.
    back = (pl.col("pos_m") <= 0) & (pl.col("link_idx") > 0)
    df = placed.with_columns(
        pl.when(back).then(pl.col("link_idx") - 1).otherwise(pl.col("link_idx")).cast(pl.Int16).alias("_li"),
        pl.when(back).then(pl.lit(np.inf)).otherwise(pl.col("pos_m").cast(pl.Float64)).alias("_pos"),
    )
    df = df.join(groups.select(pl.col("pattern_uid"), pl.col("link_idx").alias("_li"), "_gid"),
                 on=["pattern_uid", "_li"], how="left", maintain_order="left")
    ok = df["_gid"].is_not_null().to_numpy()
    gid = df["_gid"].fill_null(0).to_numpy().astype(np.int64)
    pos = df["_pos"].fill_null(0.0).to_numpy().astype(np.float64)
    pos = np.clip(np.nan_to_num(pos, nan=0.0, posinf=_SCALE / 2), -1.0, _SCALE / 2)
    n = len(df)
    if len(seg_comp) == 0:
        ok = np.zeros(n, bool)
        idx = np.zeros(n, np.int64)
    else:
        idx = np.searchsorted(seg_comp, gid * _SCALE + pos, side="left")
        idx = np.clip(idx, g_first[gid], g_last[gid])
    seg_idx = seg["seg_idx"].to_numpy()[idx] if len(seg_comp) else np.zeros(n, np.int32)
    seg_key = seg["seg_key"].gather(idx) if len(seg_comp) else pl.Series([None] * n, dtype=pl.Utf8)
    ok_s = pl.Series(ok)
    return df.drop("_li", "_pos", "_gid").with_columns(
        pl.when(ok_s).then(pl.Series(seg_idx, dtype=pl.Int32)).otherwise(None).alias("seg_idx"),
        pl.when(ok_s).then(seg_key).otherwise(None).alias("seg_key"),
    )


def per_vehicle_grid(placed: pl.DataFrame, tick_s: float = 20.0, method: str = "resample") -> pl.DataFrame:
    """Counted rows of a per-vehicle feed on the ``tick_s`` grid (``method`` "resample" or "thin")."""
    if method == "thin":
        return thin(placed.filter(pl.col("count").fill_null(True)), tick_s)
    if method != "resample":
        raise ValueError(f"unknown per-vehicle method {method!r}")
    from .locate.resample import resample
    df = placed
    if "s_m" not in df.columns:
        df = df.with_columns(pl.col("pos_m").alias("s_m"))
    elif "pos_m" in df.columns and df["s_m"].null_count():
        df = df.with_columns(pl.coalesce("s_m", pl.col("pos_m").cast(df.schema["s_m"])).alias("s_m"))
    if "count" not in df.columns:
        df = df.with_columns(pl.lit(True).alias("count"))
    # resample first (non-counted fixes still bound the intervals), then keep counted instants
    return resample(df, tick_s).filter(pl.col("count").fill_null(True))


def count(placed: pl.DataFrame, segments: pl.DataFrame, tick_s: float = 20.0,
          per_vehicle: bool = False, method: str = "resample") -> pl.DataFrame:
    """Placed -> Cube (service_date, hour, pattern_uid, seg_key, obs).

    Σ obs equals the number of counted (for per-vehicle feeds: resampled, or thinned with
    ``method="thin"``) rows whose link has segments."""
    if per_vehicle:
        df = per_vehicle_grid(placed, tick_s, method)
    else:
        df = placed.filter(pl.col("count").fill_null(True))
    df = assign(df.select("ts", "service_date", "hour", "pattern_uid", "link_idx", "pos_m", "track_id"), segments)
    cube = (df.filter(pl.col("seg_key").is_not_null())
            .group_by("service_date", "hour", "pattern_uid", "seg_key")
            .agg(pl.len().alias("obs"))
            .sort("service_date", "hour", "pattern_uid", "seg_key"))
    return conform(cube, CUBE)


def link_totals(cube: pl.DataFrame, segments: pl.DataFrame,
                by: tuple[str, ...] = ("service_date", "hour")) -> pl.DataFrame:
    """Sum the cube per link: (pattern_uid, link_key, *by, obs). Invariant to segment length and phase."""
    keys = segments.select("pattern_uid", "seg_key", "link_key").unique(["pattern_uid", "seg_key"])
    return (cube.join(keys, on=["pattern_uid", "seg_key"], how="left")
            .group_by("pattern_uid", "link_key", *by)
            .agg(pl.col("obs").sum().cast(pl.UInt64))
            .sort("pattern_uid", "link_key", *by))
