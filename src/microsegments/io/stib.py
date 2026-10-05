"""STIB ``vehicle-distance`` feed (MobilityTwin ``stib/vehicle-distance``).

Raw snapshot (one every ~20 s, all lines)::

    [{"lineId": "55", "directionId": "9943", "pointId": "5766", "distanceFromPoint": 95}, ...]

(or the same list under a ``data`` / ``results`` key). A row is a vehicle: line, terminus code
(``directionId``, a stop code, sometimes not a GTFS stop such as 9943), last stop passed and metres
travelled since it. No vehicle id.

Compact per-day parquet (prototype ``data2025/stib/vd55`` and the platform's ``raw/vd``)::

    <day>.parquet        ts:int64 epoch s, [line], dir, point, dist
    <day>.snaps.parquet  ts:int64 epoch s, same|frozen:bool, [n_rows]

Rows of frozen snapshots (content identical to the previous one) are not stored: the feed did not
move, so they carry no new observation. Snapshots keep them, flagged ``frozen``.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from .. import schema
from ..schema import OBSERVATION, SNAPSHOT, UTC
from .ids import normalise_expr

_FNAME_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-?:?(\d{2})-?:?(\d{2})")


def _records(data) -> list[dict]:
    if isinstance(data, (bytes, str)):
        data = json.loads(data)
    if isinstance(data, dict):
        data = data.get("data") or data.get("results") or []
    return list(data or [])


def _epoch_us(ts) -> int:
    """int/float epoch seconds, datetime or ISO string -> epoch microseconds."""
    import datetime as dt
    if isinstance(ts, (int, float)):
        return int(round(float(ts) * 1_000_000))
    if isinstance(ts, str):
        ts = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        return int(round(ts.timestamp() * 1_000_000))
    raise TypeError(f"unsupported timestamp {ts!r}")


def _rows_frame(recs: list[dict], ts_us: int) -> pl.DataFrame:
    return pl.DataFrame({
        "ts": [ts_us] * len(recs),
        "line": [None if r.get("lineId") is None else str(r.get("lineId")) for r in recs],
        "dir": [None if r.get("directionId") is None else str(r.get("directionId")) for r in recs],
        "point": [None if r.get("pointId") is None else str(r.get("pointId")) for r in recs],
        "dist": [r.get("distanceFromPoint") for r in recs],
    }, schema={"ts": pl.Int64, "line": pl.Utf8, "dir": pl.Utf8, "point": pl.Utf8, "dist": pl.Float64})


def parse_snapshots(snapshots: Iterable[tuple[object, object]], route: str | list[str] | None = None,
                    id_normaliser: str = "leading_digits") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse raw snapshots ``(ts, data)`` in time order -> (OBSERVATION, SNAPSHOT) frames.

    ``ts``: epoch seconds, datetime (naive = UTC) or ISO string; ``data``: decoded JSON, bytes or str.
    ``route``: keep these lineIds only (None: all lines). ``n_rows`` counts every row of the snapshot.
    A snapshot whose content equals the previous one is ``frozen`` and contributes no rows."""
    routes = None if route is None else {str(route)} if isinstance(route, (str, int)) else {str(r) for r in route}
    frames, snaps = [], []
    prev = None
    for ts, data in snapshots:
        raw = data if isinstance(data, (bytes, str)) else json.dumps(data, sort_keys=True)
        digest = hashlib.sha1(raw.encode() if isinstance(raw, str) else raw).digest()
        recs = _records(data)
        ts_us = _epoch_us(ts)
        frozen = digest == prev
        prev = digest
        snaps.append((ts_us, len(recs), frozen))
        if frozen:
            continue
        if routes is not None:
            recs = [r for r in recs if str(r.get("lineId")) in routes]
        if recs:
            frames.append(_rows_frame(recs, ts_us))
    rows = pl.concat(frames) if frames else _rows_frame([], 0)
    obs = compact_to_observations(rows.lazy(), id_normaliser=id_normaliser, ts_unit="us").collect()
    sn = pl.DataFrame(snaps, schema={"ts": pl.Int64, "n_rows": pl.UInt32, "frozen": pl.Boolean}, orient="row")
    sn = sn.with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")).cast(SNAPSHOT)
    return obs, sn


def ts_from_filename(path: str | Path) -> float | None:
    """MobilityTwin download name ``2025-02-17T04-00-00Z.json`` (UTC) -> epoch seconds."""
    import datetime as dt
    m = _FNAME_TS.search(Path(path).name)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()


def read_json_snapshots(paths: str | Path | list, route: str | list[str] | None = None,
                        id_normaliser: str = "leading_digits") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Raw JSON snapshot files (``.json`` / ``.json.gz``, globs allowed), time from the file name
    (MobilityTwin layout) -> (OBSERVATION, SNAPSHOT)."""
    files = _expand(paths)
    stamped = []
    for f in files:
        t = ts_from_filename(f)
        if t is None:
            raise ValueError(f"no UTC timestamp in file name {f} (expected e.g. 2025-02-17T04-00-00Z.json)")
        stamped.append((t, f))
    stamped.sort()

    def gen():
        for t, f in stamped:
            b = Path(f).read_bytes()
            if str(f).endswith(".gz"):
                b = gzip.decompress(b)
            yield t, b
    return parse_snapshots(gen(), route=route, id_normaliser=id_normaliser)


def compact_to_observations(lf: pl.LazyFrame, route_id: str | None = None,
                            id_normaliser: str = "leading_digits", ts_unit: str = "s") -> pl.LazyFrame:
    """Compact rows (ts, [line], dir, point, dist) -> OBSERVATION. ``route_id`` fills a missing
    ``line`` column (prototype files hold one line) or filters an existing one."""
    cols = lf.collect_schema().names()
    factor = {"s": 1_000_000, "ms": 1_000, "us": 1}[ts_unit]
    if "line" in cols:
        route = pl.col("line").cast(pl.Utf8)
        if route_id is not None:
            lf = lf.filter(pl.col("line").cast(pl.Utf8) == str(route_id))
    elif route_id is not None:
        route = pl.lit(str(route_id), dtype=pl.Utf8)
    else:
        raise ValueError("compact STIB rows have no 'line' column: pass route_id")
    out = lf.select(
        (pl.col("ts").cast(pl.Int64) * factor).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("ts"),
        route.alias("route_id"),
        normalise_expr(pl.col("dir"), id_normaliser).alias("terminus_id"),
        normalise_expr(pl.col("point"), id_normaliser).alias("stop_id"),
        pl.col("dist").cast(pl.Float32).alias("dist_m"),
    )
    return schema.conform(out, OBSERVATION, schema.OBSERVATION_REQUIRED)


def observations_to_compact(obs: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """OBSERVATION (linear) -> compact columns ts:int64 s, line, dir, point, dist:int64."""
    return obs.select(
        pl.col("ts").dt.epoch("s").alias("ts"),
        pl.col("route_id").alias("line"),
        pl.col("terminus_id").alias("dir"),
        pl.col("stop_id").alias("point"),
        pl.col("dist_m").round(0).cast(pl.Int64).alias("dist"),
    )


def snapshots_to_compact(snaps: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return snaps.select(pl.col("ts").dt.epoch("s").alias("ts"), pl.col("frozen"), pl.col("n_rows"))


def _expand(paths) -> list[str]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[str] = []
    for p in paths:
        p = str(p)
        if Path(p).is_dir():
            out += sorted(str(x) for x in Path(p).rglob("*") if x.is_file())
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    return out


def split_compact_paths(paths) -> tuple[list[str], list[str]]:
    """Separate row files from snapshot files (``*.snaps.parquet`` or ``snaps.parquet``)."""
    files = [f for f in _expand(paths) if f.endswith(".parquet")]
    snaps = [f for f in files if Path(f).name.endswith("snaps.parquet")]
    rows = [f for f in files if f not in set(snaps)]
    return rows, snaps


def read_compact(paths, route_id: str | None = None, id_normaliser: str = "leading_digits",
                 ) -> tuple[pl.LazyFrame, pl.LazyFrame | None]:
    """Compact per-day parquet (files, directories or globs; row and ``snaps`` files may be mixed)
    -> (OBSERVATION LazyFrame, SNAPSHOT LazyFrame or None if no snaps file)."""
    rows, snaps = split_compact_paths(paths)
    if not rows:
        raise FileNotFoundError(f"no STIB compact parquet in {paths}")
    lf = pl.concat([pl.scan_parquet(f) for f in rows], how="diagonal_relaxed")
    obs = compact_to_observations(lf, route_id=route_id, id_normaliser=id_normaliser)
    return obs, (read_compact_snapshots(snaps) if snaps else None)


def read_compact_snapshots(paths) -> pl.LazyFrame:
    files = _expand(paths)
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")
    cols = lf.collect_schema().names()
    frozen = pl.col("frozen") if "frozen" in cols else pl.col("same") if "same" in cols else pl.lit(False)
    n_rows = pl.col("n_rows") if "n_rows" in cols else pl.lit(None)
    return lf.select(
        (pl.col("ts").cast(pl.Int64) * 1_000_000).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").cast(UTC).alias("ts"),
        n_rows.cast(pl.UInt32).alias("n_rows"),
        frozen.cast(pl.Boolean).fill_null(False).alias("frozen"),
    ).unique("ts", keep="first").sort("ts")
