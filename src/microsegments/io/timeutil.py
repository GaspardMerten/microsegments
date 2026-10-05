"""Local service time helpers shared by every stage.

A service day D runs from D ``service_day_start`` (default 04:00) to D+1 at the same time, local
clock. Fixes between midnight and the service-day start belong to the previous date, with hours
24, 25, ... (01:30 on the 18th -> service_date 17th, hour 25), as in GTFS stop_times.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from ..schema import UTC


def parse_hhmm(value: str | int | float | dt.time) -> int:
    """'04:00' / '4:30:15' / 4 / time(4) -> seconds after midnight."""
    if isinstance(value, dt.time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, (int, float)):
        return int(round(float(value) * 3600))
    parts = [int(p) for p in str(value).split(":")]
    parts += [0] * (3 - len(parts))
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def to_utc_expr(col: str | pl.Expr, dtype: pl.DataType, tz: str, unit: str = "s") -> pl.Expr:
    """Expression turning a time column of any dtype into ``schema.UTC``.

    Numbers are epoch in ``unit`` (s / ms / us / ns); naive datetimes are local time in ``tz``;
    aware datetimes are converted. Strings must be parsed beforehand (see tabular)."""
    e = pl.col(col) if isinstance(col, str) else col
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            e = e.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
        return e.dt.convert_time_zone("UTC").cast(UTC)
    if dtype == pl.Date:
        return e.cast(pl.Datetime("us")).dt.replace_time_zone(tz, ambiguous="earliest").dt.convert_time_zone("UTC").cast(UTC)
    if dtype.is_numeric():
        factor = {"s": 1_000_000, "ms": 1_000, "us": 1, "ns": 0.001}[unit]
        return (e.cast(pl.Float64) * factor).round(0).cast(pl.Int64).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    raise TypeError(f"cannot convert {dtype} to a UTC timestamp")


def add_local_time(lf: pl.LazyFrame | pl.DataFrame, tz: str = "Europe/Brussels",
                   service_day_start: str = "04:00", ts_col: str = "ts"):
    """Add ``service_date`` (Date), ``hour`` (Int8, 24+ after midnight) and ``dow`` (Int8, Monday=0
    of the service date) from the UTC ``ts_col``. Hours are wall-clock hours (DST aware)."""
    start_s = parse_hhmm(service_day_start)
    local = pl.col(ts_col).dt.convert_time_zone(tz).dt.replace_time_zone(None)
    sd = (local - pl.duration(seconds=start_s)).dt.date()
    return lf.with_columns(
        sd.alias("service_date"),
        ((local - sd.cast(pl.Datetime("us"))).dt.total_seconds() // 3600).cast(pl.Int8).alias("hour"),
        (sd.dt.weekday() - 1).cast(pl.Int8).alias("dow"),
    )


def utc_offset_s(epoch_s: np.ndarray, tz: str) -> np.ndarray:
    """UTC offset (seconds, local - UTC) at each epoch second."""
    epoch_s = np.asarray(epoch_s, dtype=np.float64)
    if epoch_s.size == 0:
        return np.zeros(0)
    s = pl.Series("t", np.floor(epoch_s).astype(np.int64) * 1_000_000).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = s.dt.convert_time_zone(tz).dt.replace_time_zone(None)
    return ((local - s.dt.replace_time_zone(None)).dt.total_seconds()).to_numpy().astype(np.float64)


def service_day_bounds(date: dt.date, tz: str = "Europe/Brussels", service_day_start: str = "04:00") -> tuple[dt.datetime, dt.datetime]:
    """UTC [start, end) of a service date."""
    z = ZoneInfo(tz)
    s = parse_hhmm(service_day_start)
    a = dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=s)
    b = dt.datetime.combine(date + dt.timedelta(days=1), dt.time()) + dt.timedelta(seconds=s)
    return (a.replace(tzinfo=z).astimezone(dt.timezone.utc), b.replace(tzinfo=z).astimezone(dt.timezone.utc))


def local_epoch(date: dt.date, seconds: float, tz: str = "Europe/Brussels") -> float:
    """Epoch seconds of ``seconds`` after local midnight of ``date`` (seconds may exceed 86400)."""
    z = ZoneInfo(tz)
    base = dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=float(seconds))
    return base.replace(tzinfo=z).timestamp()
