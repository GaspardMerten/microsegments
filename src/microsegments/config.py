"""Run configuration: where the data is, how its columns map to ``schema.OBSERVATION``, what to select
and the method parameters. Load from TOML (``Config.from_toml``) or build in Python.

Example (lat/lon GTFS-RT positions exported to CSV)::

    [input]
    paths = ["positions/*.csv"]
    kind = "latlon"
    [input.columns]
    ts = "timestamp"
    route_id = "route_id"
    vehicle_id = "vehicle_id"
    lat = "latitude"
    lon = "longitude"
    [gtfs]
    path = "gtfs.zip"
    [select]
    route = "55"
    dates = "2025-02-17..2025-05-16"
    weekdays = [0, 1, 2, 3, 4]
"""
from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass
class LinearInput:
    """Linear-referenced feeds: 'last stop passed + metres since' (STIB vehicle-distance)."""
    direction_kind: Literal["terminus_stop", "direction_id"] = "terminus_stop"
    id_normaliser: str = "leading_digits"     # "none" | "leading_digits" | a regex with one group
    feed_length: Literal["auto", "gtfs"] = "auto"   # auto: rescale feed metres to GTFS link length (p99.5)
    terminus_aliases: dict[str, str] = field(default_factory=dict)   # code -> stop_id, else learned


@dataclass
class Input:
    paths: list[str] = field(default_factory=list)   # csv / parquet / json globs
    kind: Literal["latlon", "linear", "stib"] = "latlon"
    format: Literal["auto", "csv", "parquet", "stib_json", "gtfsrt_pb"] = "auto"
    timezone: str = "Europe/Brussels"
    service_day_start: str = "04:00"
    ts_unit: Literal["s", "ms", "us", "iso"] = "s"
    # canonical name -> source column name (only the ones that differ / exist)
    columns: dict[str, str] = field(default_factory=dict)
    linear: LinearInput = field(default_factory=LinearInput)
    snapshots: str | None = None              # optional path(s) to a Snapshot table
    events: str | None = None                 # optional stop events (StopEvent schema or STIB punctuality)


@dataclass
class Gtfs:
    path: str | None = None                   # zip / directory / gtfs-parquet directory
    dated: str | None = None                  # template with {date} (YYYY-MM-DD) for one feed per day
    route_key: Literal["route_short_name", "route_id"] = "route_short_name"


@dataclass
class Select:
    route: str | None = None
    dates: str | None = None                  # "YYYY-MM-DD..YYYY-MM-DD"
    weekdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    exclude_dates: list[str] = field(default_factory=list)
    hours: tuple[int, int] = (5, 24)          # [start, end)
    directions: list[int] | None = None

    def date_range(self) -> tuple[dt.date, dt.date] | None:
        if not self.dates:
            return None
        a, b = self.dates.split("..")
        return dt.date.fromisoformat(a), dt.date.fromisoformat(b)


@dataclass
class Params:
    segment_m: float = 30.0
    phase_m: float = 0.0
    grid: Literal["equal", "fixed"] = "equal"  # equal: L' = len/round(len/L) per link; fixed: L with a short last bin
    tick_s: float = 20.0                     # nominal poll interval; per-vehicle feeds are put on this grid
    per_vehicle: Literal["resample", "thin"] = "resample"   # resample: locate.resample (unbiased); thin: 1 fix / tick
    gap_cap_s: float = 40.0                  # a poll covers at most this long
    stop_zone: tuple[float, float] = (30.0, 60.0)   # metres before / after a stop masked as "stop" zone
    reference_hours: tuple[int, int] = (20, 23)     # [start, end) "normal" evening reference
    reference: Literal["evening", "freeflow"] = "evening"
    layover_m: float = 10.0                  # at the first stop of a pattern closer than this = terminus layover
    max_lateral_m: float = 40.0              # map matching: reject fixes farther from the shape
    bootstrap: int = 200


@dataclass
class Quality:
    min_hour_coverage: float = 0.8
    min_day_coverage: float = 0.75           # share of 6-21 h day-hours that must pass
    max_frozen_s: float = 300.0
    vehicle_ratio: tuple[float, float] = (0.7, 1.3)
    max_off_pattern: float = 0.2
    min_days: int = 5
    min_passages_per_day: float = 0.5


@dataclass
class Report:
    names: str | dict[str, str] | None = None   # "title": title-case all-caps stop names; or {gtfs: shown}


@dataclass
class Config:
    input: Input = field(default_factory=Input)
    gtfs: Gtfs = field(default_factory=Gtfs)
    select: Select = field(default_factory=Select)
    params: Params = field(default_factory=Params)
    quality: Quality = field(default_factory=Quality)
    report: Report = field(default_factory=Report)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _build(cls, d)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        path = Path(path)
        cfg = cls.from_dict(tomllib.loads(path.read_text()))
        base = path.parent
        cfg.input.paths = [p if Path(p).is_absolute() else str(base / p) for p in cfg.input.paths]
        for obj, attr in ((cfg.gtfs, "path"), (cfg.gtfs, "dated"), (cfg.input, "snapshots"), (cfg.input, "events")):
            v = getattr(obj, attr)
            if v and not Path(v).is_absolute():
                setattr(obj, attr, str(base / v))
        return cfg


def _build(cls, d: dict[str, Any]):
    kw = {}
    names = {f.name: f for f in fields(cls)}
    for k, v in (d or {}).items():
        if k not in names:
            raise ValueError(f"unknown config key {cls.__name__}.{k}")
        f = names[k]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(v, dict):
            v = _build(type(default), v)
        elif isinstance(default, tuple) and isinstance(v, list):
            v = tuple(v)
        kw[k] = v
    return cls(**kw)
