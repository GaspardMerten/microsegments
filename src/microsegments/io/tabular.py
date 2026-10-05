"""Generic CSV / parquet positions -> ``schema.OBSERVATION``.

``cfg.input.columns`` maps canonical names to source columns (only those that differ). Times:
``cfg.input.ts_unit`` s / ms / us epoch, or ``iso`` strings (with offset or ``Z``: converted to
UTC; naive: local time in ``cfg.input.timezone``). Datetime columns are handled by dtype.
Stop / terminus ids go through ``cfg.input.linear.id_normaliser``.

``read_observations`` also dispatches the other adapters by ``cfg.input.format`` / ``kind`` /
extension: STIB raw JSON (``.json``), STIB compact parquet (``kind = "stib"``), GTFS-RT ``.pb``.
"""
from __future__ import annotations

import glob
from pathlib import Path

import polars as pl

from .. import schema
from ..config import Config
from ..schema import OBSERVATION
from .ids import normalise_expr
from .timeutil import add_local_time, service_day_bounds, to_utc_expr

_ISO_OFFSET = r"(Z|[+-]\d{2}:?\d{2})$"


def expand_paths(paths) -> list[str]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[str] = []
    for p in paths:
        p = str(p)
        if any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        elif Path(p).is_dir():
            out += sorted(str(x) for x in Path(p).rglob("*") if x.is_file() and not x.name.startswith("."))
        else:
            out.append(p)
    return out


def _fmt(path: str, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    n = path.lower()
    if n.endswith((".parquet", ".pq")):
        return "parquet"
    if n.endswith((".json", ".json.gz")):
        return "stib_json"
    if n.endswith((".pb", ".pbf", ".pb.gz")):
        return "gtfsrt_pb"
    return "csv"


def scan_table(files: list[str], fmt: str = "auto") -> pl.LazyFrame:
    frames = []
    for f in files:
        k = _fmt(f, fmt)
        if k == "parquet":
            frames.append(pl.scan_parquet(f))
        elif k == "csv":
            frames.append(pl.scan_csv(f, infer_schema_length=10000, try_parse_dates=False))
        else:
            raise ValueError(f"{f}: format {k} is not tabular")
    if not frames:
        raise FileNotFoundError("no input files")
    return pl.concat(frames, how="diagonal_relaxed")


def parse_ts(lf: pl.LazyFrame, col: str = "ts", unit: str = "s", tz: str = "Europe/Brussels") -> pl.LazyFrame:
    """Turn ``col`` into ``schema.UTC`` whatever its dtype (see module doc)."""
    dtype = lf.collect_schema()[col]
    if dtype == pl.Utf8:
        import re
        first = lf.select(pl.col(col)).drop_nulls().head(1).collect()
        sample = (first[col][0] if len(first) else "").strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", sample):   # numbers stored as text
            u = "s" if unit == "iso" else unit
            return lf.with_columns(to_utc_expr(pl.col(col).cast(pl.Float64, strict=False), pl.Float64, tz, u).alias(col))
        s = pl.col(col).str.strip_chars().str.replace(" ", "T")
        if re.search(_ISO_OFFSET, sample):
            e = s.str.replace(r"Z$", "+00:00").str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
        else:
            e = (s.str.to_datetime(time_unit="us", strict=False)
                 .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null").dt.convert_time_zone("UTC"))
        return lf.with_columns(e.cast(schema.UTC).alias(col))
    return lf.with_columns(to_utc_expr(col, dtype, tz, unit).alias(col))


def normalise_frame(lf: pl.LazyFrame, columns: dict[str, str] | None = None, ts_unit: str = "s",
                    tz: str = "Europe/Brussels", id_normaliser: str = "leading_digits",
                    route: str | None = None) -> pl.LazyFrame:
    """Rename (canonical -> source mapping), parse time, normalise ids, filter/fill route, conform."""
    columns = dict(columns or {})
    names = lf.collect_schema().names()
    ren = {src: can for can, src in columns.items() if src in names and src != can}
    # a canonical column shadowed by a renamed source would clash: drop it first
    clash = [c for c in ren.values() if c in names and c not in ren]
    if clash:
        lf = lf.drop(clash)
    lf = lf.rename(ren)
    names = lf.collect_schema().names()
    if "ts" not in names:
        raise ValueError(f"no time column: map it with [input.columns] ts = '<name>' (have {names})")
    lf = parse_ts(lf, "ts", ts_unit, tz)
    ids = [normalise_expr(pl.col(c), id_normaliser).alias(c) for c in ("stop_id", "terminus_id") if c in names]
    if ids:
        lf = lf.with_columns(ids)
    for c in ("route_id", "vehicle_id", "trip_id"):
        if c in names:
            lf = lf.with_columns(pl.col(c).cast(pl.Utf8))
    if "route_id" not in names:
        if route is None:
            raise ValueError("no route_id column and no select.route to fill it")
        lf = lf.with_columns(pl.lit(str(route)).alias("route_id"))
    elif route is not None:
        lf = lf.filter(pl.col("route_id") == str(route))
    return schema.conform(lf, OBSERVATION, schema.OBSERVATION_REQUIRED)


def _date_filter(lf: pl.LazyFrame, cfg: Config) -> pl.LazyFrame:
    rng = cfg.select.date_range()
    if not rng:
        return lf
    a, _ = service_day_bounds(rng[0], cfg.input.timezone, cfg.input.service_day_start)
    _, b = service_day_bounds(rng[1], cfg.input.timezone, cfg.input.service_day_start)
    return lf.filter((pl.col("ts") >= a) & (pl.col("ts") < b))


def read_observations(cfg: Config) -> pl.LazyFrame:
    """All input files of ``cfg`` -> OBSERVATION LazyFrame (route and date range of ``cfg.select``
    applied when given)."""
    files = expand_paths(cfg.input.paths)
    if not files:
        raise FileNotFoundError(f"no files match {cfg.input.paths}")
    inp = cfg.input
    route = cfg.select.route
    norm = inp.linear.id_normaliser
    kinds = {_fmt(f, inp.format) for f in files}
    if kinds == {"stib_json"}:
        from .stib import read_json_snapshots
        obs, _ = read_json_snapshots(files, route=route, id_normaliser=norm)
        lf = obs.lazy()
    elif kinds == {"gtfsrt_pb"}:
        from .gtfsrt import read_pb
        lf = read_pb(files, route=route).lazy()
    elif inp.kind == "stib" and kinds == {"parquet"} and not inp.columns:
        from .stib import read_compact
        lf, _ = read_compact(files, route_id=route, id_normaliser=norm)
    elif kinds <= {"csv", "parquet"}:
        lf = scan_table(files, inp.format)
        cols = dict(inp.columns)
        if inp.kind == "latlon" and not cols:
            from .gtfsrt import detect_columns
            cols = detect_columns(lf.collect_schema().names())
        lf = normalise_frame(lf, cols, inp.ts_unit, inp.timezone, norm, route)
    else:
        raise ValueError(f"mixed input formats {kinds}")
    return _date_filter(lf, cfg)


def read_snapshots(cfg: Config) -> pl.LazyFrame | None:
    """Snapshot table from ``cfg.input.snapshots`` (parquet/CSV with ts [, n_rows, frozen|same]),
    or the ``snaps`` files mixed in ``cfg.input.paths`` (STIB compact). None if neither."""
    from .stib import read_compact_snapshots, split_compact_paths
    if inp_s := cfg.input.snapshots:
        files = expand_paths(inp_s)
        if all(f.endswith(".parquet") for f in files):
            return read_compact_snapshots(files) if _is_epoch(files[0]) else _generic_snaps(files, cfg)
        return _generic_snaps(files, cfg)
    _, snaps = split_compact_paths(expand_paths(cfg.input.paths))
    return read_compact_snapshots(snaps) if snaps else None


def _is_epoch(f: str) -> bool:
    return pl.read_parquet_schema(f).get("ts") in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64)


def _generic_snaps(files, cfg: Config) -> pl.LazyFrame:
    lf = parse_ts(scan_table(files), "ts", cfg.input.ts_unit, cfg.input.timezone)
    names = lf.collect_schema().names()
    if "frozen" not in names:
        lf = lf.with_columns((pl.col("same") if "same" in names else pl.lit(False)).alias("frozen"))
    return schema.conform(lf, schema.SNAPSHOT).select(list(schema.SNAPSHOT))


__all__ = ["read_observations", "read_snapshots", "normalise_frame", "parse_ts", "scan_table",
           "expand_paths", "add_local_time"]
