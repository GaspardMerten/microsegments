"""GTFS-RT VehiclePositions -> ``schema.OBSERVATION``.

Flattened tables (CSV / parquet) are read with their column names auto-detected: protobuf paths
(``vehicle.position.latitude``, ``vehicle.trip.trip_id``, ``vehicle.vehicle.id``,
``vehicle.timestamp`` ...), snake / camel variants, or plain ``latitude`` / ``lon`` / ``timestamp``
/ ``trip_id`` / ``route_id`` / ``vehicle_id`` / ``direction_id`` / ``bearing``.
Raw ``.pb`` FeedMessages need the ``gtfsrt`` extra (``gtfs-realtime-bindings``).
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import polars as pl

from .. import schema
from ..schema import OBSERVATION

try:  # optional extra
    from google.transit import gtfs_realtime_pb2  # type: ignore
    HAS_BINDINGS = True
except Exception:  # pragma: no cover - depends on the environment
    gtfs_realtime_pb2 = None
    HAS_BINDINGS = False

# canonical -> candidate suffixes of the normalised column name (dot separated, lower case)
_CANDIDATES: dict[str, list[str]] = {
    "ts": ["vehicle.timestamp", "position.timestamp", "timestamp", "ts", "time", "datetime", "fix.time"],
    "lat": ["position.latitude", "latitude", "lat"],
    "lon": ["position.longitude", "longitude", "lon", "lng", "long"],
    "bearing": ["position.bearing", "bearing", "heading"],
    "trip_id": ["trip.trip.id", "trip.id"],
    "route_id": ["trip.route.id", "route.id", "route"],
    "direction_id": ["trip.direction.id", "direction.id", "direction"],
    "vehicle_id": ["vehicle.vehicle.id", "vehicle.id", "vehicle.label", "vehicle"],
}


def _norm(name: str) -> str:
    n = re.sub(r"([a-z0-9])([A-Z])", r"\1.\2", name)       # camelCase -> camel.Case
    return re.sub(r"[^a-z0-9]+", ".", n.lower()).strip(".")


def detect_columns(names: list[str]) -> dict[str, str]:
    """Map canonical OBSERVATION names to source columns (canonical -> source)."""
    normed = {c: _norm(c) for c in names if not _norm(c).startswith("header.")}
    out: dict[str, str] = {}
    used: set[str] = set()
    for can, cands in _CANDIDATES.items():
        for cand in cands:
            hits = [c for c, n in normed.items() if c not in used and (n == cand or n.endswith("." + cand))]
            if hits:
                best = min(hits, key=lambda c: len(normed[c]))
                out[can] = best
                used.add(best)
                break
    return out


def _guess_unit(lf: pl.LazyFrame, col: str) -> str:
    dtype = lf.collect_schema()[col]
    if dtype == pl.Utf8:
        v = lf.select(pl.col(col)).drop_nulls().head(1).collect()[col]
        if len(v) and not re.fullmatch(r"\s*\d+(\.\d+)?\s*", v[0]):
            return "iso"
        mx = lf.select(pl.col(col).cast(pl.Float64, strict=False).max()).collect().item()
    elif dtype.is_numeric():
        mx = lf.select(pl.col(col).max()).collect().item()
    else:
        return "s"
    if mx is None:
        return "s"
    return "us" if mx > 1e14 else "ms" if mx > 1e11 else "s"


def read_flat(paths, route: str | None = None, tz: str = "Europe/Brussels",
              columns: dict[str, str] | None = None, ts_unit: str | None = None) -> pl.LazyFrame:
    """Flattened VehiclePositions (CSV / parquet files or globs) -> OBSERVATION LazyFrame.
    ``columns`` overrides the detection (canonical -> source); ``ts_unit`` None = guessed."""
    from .tabular import expand_paths, normalise_frame, scan_table
    lf = scan_table(expand_paths(paths))
    cols = detect_columns(lf.collect_schema().names())
    cols.update(columns or {})
    if "ts" not in cols:
        raise ValueError("no timestamp column detected")
    unit = ts_unit or _guess_unit(lf, cols["ts"])
    return normalise_frame(lf, cols, unit, tz, "none", route)


def parse_feed(content: bytes, route: str | None = None) -> pl.DataFrame:
    """One serialized FeedMessage -> OBSERVATION rows (vehicle entities only)."""
    if not HAS_BINDINGS:
        raise ImportError("parsing .pb needs the 'gtfsrt' extra: pip install 'microsegments[gtfsrt]'")
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.ParseFromString(content)
    head_ts = msg.header.timestamp if msg.header.HasField("timestamp") else None
    rows = []
    for ent in msg.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        if not v.HasField("position"):
            continue
        ts = v.timestamp if v.HasField("timestamp") else head_ts
        if ts is None:
            continue
        tr = v.trip if v.HasField("trip") else None
        rid = tr.route_id if tr is not None and tr.route_id else None
        if route is not None and rid != str(route):
            continue
        p = v.position
        rows.append({
            "ts": int(ts) * 1_000_000,
            "route_id": rid,
            "vehicle_id": (v.vehicle.id or v.vehicle.label or ent.id) if v.HasField("vehicle") else ent.id,
            "trip_id": (tr.trip_id or None) if tr is not None else None,
            "direction_id": tr.direction_id if tr is not None and tr.HasField("direction_id") else None,
            "lat": p.latitude, "lon": p.longitude,
            "bearing": p.bearing if p.HasField("bearing") else None,
        })
    df = pl.DataFrame(rows, schema={"ts": pl.Int64, "route_id": pl.Utf8, "vehicle_id": pl.Utf8,
                                    "trip_id": pl.Utf8, "direction_id": pl.Int8, "lat": pl.Float64,
                                    "lon": pl.Float64, "bearing": pl.Float32})
    df = df.with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    return schema.conform(df, OBSERVATION, schema.OBSERVATION_REQUIRED)


def read_pb(paths, route: str | None = None) -> pl.DataFrame:
    """``.pb`` / ``.pb.gz`` FeedMessage files -> OBSERVATION, duplicates (same vehicle and ts) dropped."""
    from .tabular import expand_paths
    frames = []
    for f in expand_paths(paths):
        b = Path(f).read_bytes()
        if f.endswith(".gz"):
            b = gzip.decompress(b)
        frames.append(parse_feed(b, route))
    if not frames:
        return schema.empty(OBSERVATION)
    return pl.concat(frames).unique(["ts", "vehicle_id", "lat", "lon"], maintain_order=True).sort("ts")


def build_feed(obs: pl.DataFrame) -> bytes:
    """OBSERVATION (lat/lon) -> serialized FeedMessage (for tests and examples)."""
    if not HAS_BINDINGS:
        raise ImportError("needs the 'gtfsrt' extra")
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    if len(obs):
        msg.header.timestamp = int(obs["ts"].dt.epoch("s").max())
    for i, r in enumerate(obs.iter_rows(named=True)):
        e = msg.entity.add()
        e.id = str(i)
        v = e.vehicle
        v.timestamp = int(r["ts"].timestamp())
        v.position.latitude = r["lat"]
        v.position.longitude = r["lon"]
        if r.get("bearing") is not None:
            v.position.bearing = r["bearing"]
        if r.get("vehicle_id") is not None:
            v.vehicle.id = r["vehicle_id"]
        if r.get("trip_id") is not None:
            v.trip.trip_id = r["trip_id"]
        if r.get("route_id") is not None:
            v.trip.route_id = r["route_id"]
        if r.get("direction_id") is not None:
            v.trip.direction_id = int(r["direction_id"])
    return msg.SerializeToString()
