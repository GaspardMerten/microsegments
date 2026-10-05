"""Canonical tables exchanged between the pipeline stages.

Every stage takes and returns polars frames with these columns (extra columns are allowed
and carried along). Adapters in ``microsegments.io`` turn any source into ``OBSERVATION``;
everything downstream only relies on the schemas below.

Pipeline::

    source --io--> Observation --locate--> Placed --aggregate--> Cube --metrics--> Result
                   Snapshot ----coverage-> Coverage --------------------^
    GTFS --network--> Pattern / PatternDay / Link --segments--> Segment
    Placed (+ StopEvent) --tracks--> Passages

Times: ``ts`` is always UTC. ``service_date`` / ``hour`` / ``dow`` are local (config timezone),
with a configurable service-day start (default 04:00, so 01:30 belongs to the previous day).
"""
from __future__ import annotations

import polars as pl

UTC = pl.Datetime("us", "UTC")

# ---------------------------------------------------------------- input
# One row per position fix. Either (lat, lon) or (stop_id, dist_m) must be filled.
OBSERVATION = {
    "ts": UTC,                   # snapshot or fix time
    "route_id": pl.Utf8,         # line, as it should match the GTFS (route_id or short name, see config)
    "vehicle_id": pl.Utf8,       # null when the feed is anonymous (STIB vehicle-distance)
    "trip_id": pl.Utf8,          # GTFS trip_id when known
    "direction_id": pl.Int8,     # GTFS 0/1 when known
    "terminus_id": pl.Utf8,      # destination stop / code (STIB directionId), normalised
    "stop_id": pl.Utf8,          # linear referencing: last stop passed, normalised
    "dist_m": pl.Float32,        # linear referencing: metres travelled since stop_id
    "lat": pl.Float64,
    "lon": pl.Float64,
    "bearing": pl.Float32,       # degrees, 0 = north
}
OBSERVATION_REQUIRED = ("ts", "route_id")

# One row per feed poll (snapshot feeds). Inferred from distinct observation ts if absent.
SNAPSHOT = {
    "ts": UTC,
    "n_rows": pl.UInt32,         # rows of the whole feed in that snapshot (all lines)
    "frozen": pl.Boolean,        # identical to the previous snapshot: feed stalled, not a new poll
}

# Optional stop-call events (STIB punctuality, GTFS-RT TripUpdates history, AVL stop log).
STOP_EVENT = {
    "service_date": pl.Date,
    "route_id": pl.Utf8,
    "trip_key": pl.Utf8,         # any id unique per trip run within the day
    "stop_id": pl.Utf8,
    "stop_sequence": pl.Int32,
    "arrival": UTC,
    "departure": UTC,
    "observed": pl.Boolean,      # measured (True) or interpolated (False)
}

# ---------------------------------------------------------------- coverage
# Feed health per local service date x hour. One slot = one nominal poll interval (tick_s).
COVERAGE = {
    "service_date": pl.Date,
    "hour": pl.Int8,             # 0..27 allowed: hours after midnight belong to the service day (24 = 00h next day)
    "n_snapshots": pl.UInt32,    # distinct, non-frozen polls
    "covered_s": pl.Float32,     # sum over polls of min(gap to next poll, gap_cap_s)
    "coverage": pl.Float32,      # covered_s / 3600 (share of the hour the feed was alive), 0..1
    "frozen_s": pl.Float32,      # seconds during which the feed repeated itself
    "flags": pl.UInt16,          # Flag bitmask
}

# ---------------------------------------------------------------- network
PATTERN = {
    "pattern_uid": pl.Utf8,      # sha1(stop sequence + rounded shape)[:10], stable across feeds
    "route_id": pl.Utf8,
    "route_short_name": pl.Utf8,
    "direction_id": pl.Int8,
    "stop_ids": pl.List(pl.Utf8),
    "stop_names": pl.List(pl.Utf8),
    "kind": pl.Utf8,             # "main" | "short" | "detour" | "other"
    "parent_uid": pl.Utf8,       # for "short": the main pattern it is a contiguous part of
    "parent_offset": pl.Int16,   # index of this pattern's first stop in the parent
    "length_m": pl.Float64,
    "geometry": pl.List(pl.List(pl.Float64)),   # [[lon, lat], ...]
}

# Which pattern ran on which date (from the GTFS valid that day).
PATTERN_DAY = {
    "service_date": pl.Date,
    "route_id": pl.Utf8,
    "direction_id": pl.Int8,
    "pattern_uid": pl.Utf8,
    "n_trips": pl.UInt32,        # scheduled trips that day
    "is_main": pl.Boolean,       # covers the most scheduled link traversals that day (network.patterns.choose_main)
}

LINK = {
    "pattern_uid": pl.Utf8,
    "link_idx": pl.Int16,        # 0 = first stop -> second stop
    "link_key": pl.Utf8,         # stable physical id across GTFS versions, see network.keys
    "from_stop": pl.Utf8,
    "to_stop": pl.Utf8,
    "from_name": pl.Utf8,
    "to_name": pl.Utf8,
    "s0_m": pl.Float64,          # along-pattern start
    "len_m": pl.Float64,
    "geometry": pl.List(pl.List(pl.Float64)),
}

SEGMENT = {
    "pattern_uid": pl.Utf8,
    "seg_idx": pl.Int32,         # along the pattern, 0..n-1
    "link_idx": pl.Int16,
    "link_key": pl.Utf8,
    "seg_key": pl.Utf8,          # f"{link_key}/{k}@{length}:{phase}" , stable for a given segmentation
    "k": pl.Int16,               # index within the link
    "offset_m": pl.Float64,      # start, from link start
    "x0_m": pl.Float64,          # start, from pattern start
    "len_m": pl.Float64,
    "zone": pl.Utf8,             # "stop" (within the stop zone around a stop) | "running"
    "geometry": pl.List(pl.List(pl.Float64)),
}

# ---------------------------------------------------------------- located observations
PLACED = {
    "ts": UTC,
    "service_date": pl.Date,
    "hour": pl.Int8,
    "dow": pl.Int8,              # 0 = Monday
    "route_id": pl.Utf8,
    "pattern_uid": pl.Utf8,      # main pattern the fix was placed on (short workings map to parent)
    "link_idx": pl.Int16,
    "link_key": pl.Utf8,
    "pos_m": pl.Float32,         # within the link, (0, len]; a vehicle standing at stop B is at len of link A->B
    "s_m": pl.Float32,           # along the pattern
    "raw_dist_m": pl.Float32,    # linear input only, before rescaling
    "match_err_m": pl.Float32,   # lat/lon input only, lateral distance to the shape
    "track_id": pl.Utf8,         # vehicle_id, or id reconstructed by tracks.track_anonymous
    "count": pl.Boolean,         # False: kept for tracking only (terminus pseudo rows), never counted
    "flags": pl.UInt16,
}

PASSAGES = {
    "service_date": pl.Date,
    "hour": pl.Int8,
    "pattern_uid": pl.Utf8,
    "link_key": pl.Utf8,
    "n_feed": pl.Float32,        # from position tracks
    "n_events": pl.Float32,      # from stop events (null if none)
    "n": pl.Float32,             # max(n_feed, n_events): both sources only ever miss vehicles
}

# ---------------------------------------------------------------- counts
# The core product: how many times a vehicle was observed in each micro-segment.
CUBE = {
    "service_date": pl.Date,
    "hour": pl.Int8,
    "pattern_uid": pl.Utf8,
    "seg_key": pl.Utf8,
    "obs": pl.UInt32,
}

# Aggregate over the selected days, per segment x hour (or band). See metrics.py for formulas.
RESULT = {
    "pattern_uid": pl.Utf8,
    "direction_id": pl.Int8,
    "seg_key": pl.Utf8,
    "seg_idx": pl.Int32,
    "hour": pl.Int8,
    "obs": pl.UInt32,            # raw total over included day-hours
    "n_days": pl.UInt16,         # included days with this segment on the main pattern
    "covered_h": pl.Float32,     # sum of coverage over included day-hours (exposure)
    "obs_per_h": pl.Float32,     # obs / covered_h : mean vehicles seen in the segment per full hour of polling
    "passages": pl.Float32,
    "obs_per_passage": pl.Float32,
    "ref_obs_per_passage": pl.Float32,
    "excess_obs_per_h": pl.Float32,        # (obs - ref * passages) / covered_h
    "excess_per_passage": pl.Float32,      # obs_per_passage - ref
    "log2_ratio": pl.Float32,
    "flags": pl.UInt16,
}

HOTSPOT = {
    "pattern_uid": pl.Utf8,
    "direction_id": pl.Int8,
    "rank": pl.Int16,
    "from_seg": pl.Int32,
    "to_seg": pl.Int32,          # inclusive
    "x0_m": pl.Float64,
    "x1_m": pl.Float64,
    "from_stop_name": pl.Utf8,   # link the stretch starts in
    "to_stop_name": pl.Utf8,
    "zone": pl.Utf8,             # "stop" | "running"
    "kind": pl.Utf8,             # "infrastructure" | "congestion" | "mixed"
    "hours": pl.List(pl.Int8),
    "peak_hour": pl.Int8,
    "excess_per_passage": pl.Float32,      # summed over the stretch, in observations per vehicle
    "ci_lo": pl.Float32,
    "ci_hi": pl.Float32,
    "persistence": pl.Float32,   # share of days with positive excess
    "n_days": pl.UInt16,
}


class Flag:
    """Bitmask values used in ``flags`` columns."""
    LOW_COVERAGE = 1          # coverage below quality.min_hour_coverage
    FROZEN = 2                # feed stalled > quality.max_frozen_s
    LINE_ABSENT = 4           # feed alive, trips scheduled, no row for the line
    DEVIATION = 8             # share of off-pattern rows above quality.max_off_pattern
    FEW_DAYS = 16             # n_days below quality.min_days
    PARTIAL_VERSIONS = 32     # segment absent from the main pattern on some selected days
    PASSAGES_ESTIMATED = 64   # passages from schedule, not observed
    LOW_DETECTION = 128       # vehicles often unseen in this segment (tunnel, GPS loss)
    VEHICLE_RATIO = 256       # vehicles seen / scheduled outside quality bounds
    # placement flags (Placed.flags)
    LAYOVER = 512
    SHORT_WORKING = 1024
    OFF_PATTERN = 2048
    AMBIGUOUS = 4096
    AT_STOP = 8192

    @classmethod
    def names(cls, value: int) -> list[str]:
        return [k.lower() for k, v in vars(cls).items() if isinstance(v, int) and value & v]


def empty(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def conform(df: pl.DataFrame | pl.LazyFrame, schema: dict, required: tuple[str, ...] = ()):
    """Add missing schema columns as nulls, cast present ones, keep extra columns.

    Raises ValueError if a required column is missing."""
    cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    miss = [c for c in required if c not in cols]
    if miss:
        raise ValueError(f"missing required columns: {miss}")
    exprs = []
    for name, dtype in schema.items():
        if name in cols:
            exprs.append(pl.col(name).cast(dtype, strict=False))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))
    extra = [pl.col(c) for c in cols if c not in schema]
    return df.select(exprs + extra)


def validate(df: pl.DataFrame, schema: dict, required: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Check columns and dtypes; required defaults to all schema columns."""
    required = tuple(schema) if required is None else required
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing column {c!r}")
        if df.schema[c] != schema[c]:
            raise TypeError(f"column {c!r} is {df.schema[c]}, expected {schema[c]}")
    return df
