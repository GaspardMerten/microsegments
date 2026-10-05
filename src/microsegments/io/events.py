"""Stop-call events -> ``schema.STOP_EVENT``.

STIB punctuality parquet (MobilityTwin ``stib/punctuality``): one row per trip x stop with
``journey_id, trip_id, start_date (YYYYMMDD), stop_sequence, stop_id ("5272F"), route_id ("55"),
arrival_time, departure_time`` (naive UTC), ``observed`` (else inferred).
Generic CSV / parquet: any column names, mapped canonical -> source.
"""
from __future__ import annotations

import polars as pl

from .. import schema
from ..schema import STOP_EVENT, UTC
from .ids import normalise_expr
from .tabular import expand_paths, parse_ts, scan_table
from .timeutil import add_local_time

_PUNCT_COLS = ["journey_id", "trip_id", "start_date", "stop_sequence", "stop_id", "route_id",
               "arrival_time", "departure_time", "observed"]


def read_stib_punctuality(paths, route: str | None = None, id_normaliser: str = "leading_digits",
                          only_observed: bool = False) -> pl.LazyFrame:
    """STIB punctuality parquet files -> STOP_EVENT. ``trip_key`` = ``journey_id:trip_id``."""
    files = [f for f in expand_paths(paths) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"no punctuality parquet in {paths}")
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")
    names = lf.collect_schema().names()
    if route is not None:
        lf = lf.filter(pl.col("route_id").cast(pl.Utf8) == str(route))
    if only_observed:
        lf = lf.filter(pl.col("observed"))

    def utc(c):
        dtype = lf.collect_schema()[c]
        e = pl.col(c)
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            e = e.dt.replace_time_zone("UTC")
        return e.dt.convert_time_zone("UTC").cast(UTC)

    jid = pl.col("journey_id").cast(pl.Utf8) if "journey_id" in names else pl.lit("")
    out = lf.select(
        pl.col("start_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("service_date"),
        pl.col("route_id").cast(pl.Utf8),
        pl.concat_str([jid.fill_null(""), pl.col("trip_id").cast(pl.Utf8)], separator=":").alias("trip_key"),
        normalise_expr(pl.col("stop_id"), id_normaliser).alias("stop_id"),
        pl.col("stop_sequence").cast(pl.Int32),
        utc("arrival_time").alias("arrival"),
        utc("departure_time").alias("departure"),
        (pl.col("observed") if "observed" in names else pl.lit(True)).cast(pl.Boolean).alias("observed"),
    )
    return out.select(list(STOP_EVENT))


def read_events(paths, columns: dict[str, str] | None = None, ts_unit: str = "s",
                tz: str = "Europe/Brussels", service_day_start: str = "04:00",
                id_normaliser: str = "none", route: str | None = None) -> pl.LazyFrame:
    """Generic CSV / parquet stop events -> STOP_EVENT.

    ``columns``: canonical -> source names. Needs ``trip_key`` (or ``trip_id``), ``stop_id`` and
    ``arrival`` and/or ``departure``. Missing ``service_date``: the service date of the trip's first
    event; missing ``stop_sequence``: rank by time within the trip; missing ``observed``: True."""
    lf = scan_table(expand_paths(paths))
    names = lf.collect_schema().names()
    ren = {src: can for can, src in (columns or {}).items() if src in names and src != can}
    lf = lf.rename(ren)
    names = lf.collect_schema().names()
    if "trip_key" not in names:
        if "trip_id" not in names:
            raise ValueError("stop events need a trip_key or trip_id column")
        lf = lf.with_columns(pl.col("trip_id").cast(pl.Utf8).alias("trip_key"))
    for c in ("arrival", "departure"):
        if c in names:
            lf = parse_ts(lf, c, ts_unit, tz)
        else:
            lf = lf.with_columns(pl.lit(None, dtype=UTC).alias(c))
    if "arrival" not in names and "departure" not in names:
        raise ValueError("stop events need arrival and/or departure")
    t = pl.coalesce("departure", "arrival")
    if "service_date" not in names:
        lf = add_local_time(lf.with_columns(t.min().over("trip_key").alias("_t0")), tz, service_day_start,
                            ts_col="_t0").drop("_t0", "hour", "dow")
    elif lf.collect_schema()["service_date"] == pl.Utf8:
        sd = pl.col("service_date").str.replace_all("-", "")
        lf = lf.with_columns(sd.str.strptime(pl.Date, "%Y%m%d", strict=False).alias("service_date"))
    if "stop_sequence" not in names:
        lf = lf.with_columns(t.rank("ordinal").over("trip_key").cast(pl.Int32).alias("stop_sequence"))
    if "observed" not in names:
        lf = lf.with_columns(pl.lit(True).alias("observed"))
    lf = lf.with_columns(normalise_expr(pl.col("stop_id"), id_normaliser).alias("stop_id"))
    if "route_id" in names:
        lf = lf.with_columns(pl.col("route_id").cast(pl.Utf8))
        if route is not None:
            lf = lf.filter(pl.col("route_id") == str(route))
    elif route is not None:
        lf = lf.with_columns(pl.lit(str(route)).alias("route_id"))
    return schema.conform(lf, STOP_EVENT).select(list(STOP_EVENT))
