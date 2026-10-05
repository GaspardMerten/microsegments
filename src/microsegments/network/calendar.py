"""Which GTFS feed applies on a date, and which trips of a route run that day.

A :class:`GtfsSource` is either

* one feed (``path``): a GTFS zip, a directory of ``.txt`` files, or a gtfs-parquet directory / zip
  of ``.parquet`` tables;
* daily snapshots (``dated``): a path template containing ``{date}`` (``YYYY-MM-DD``) or ``{yyyymmdd}``;
  a date without a snapshot uses the latest earlier one (up to ``max_back_days``);
* a mapping ``{date: feed path}``: each feed is valid from its date until the next one.

Feeds are loaded restricted to one route (gtfs-parquet tables are scanned lazily and filtered
before reading, so a whole-network feed costs only the route's rows) and cached (LRU).
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl
from gtfs_parquet import Feed
from gtfs_parquet.parse import parse_gtfs_file

TABLES = ("routes", "trips", "stop_times", "stops", "shapes", "calendar", "calendar_dates")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _as_date(d) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


# ---------------------------------------------------------------- raw table access
class _Reader:
    """Lazy access to the tables of one feed, whatever its format."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        p = self.path
        self.zip = p.is_file() and zipfile.is_zipfile(p)
        if p.is_dir():
            self.parquet = any(p.glob("*.parquet"))
        elif self.zip:
            with zipfile.ZipFile(p) as zf:
                self.parquet = any(n.endswith(".parquet") for n in zf.namelist())
        else:
            raise FileNotFoundError(f"GTFS feed not found or not a zip / directory: {p}")

    def scan(self, name: str) -> pl.LazyFrame | None:
        p = self.path
        if self.parquet and not self.zip:
            f = p / f"{name}.parquet"
            return pl.scan_parquet(f) if f.exists() else None
        if self.zip:
            with zipfile.ZipFile(p) as zf:
                names = {Path(n).name: n for n in zf.namelist()}
                key = f"{name}.parquet" if self.parquet else f"{name}.txt"
                if key not in names:
                    return None
                raw = zf.read(names[key])
            df = pl.read_parquet(io.BytesIO(raw)) if self.parquet else parse_gtfs_file(raw, key)
            return df.lazy()
        f = p / f"{name}.txt"
        return parse_gtfs_file(f, f"{name}.txt").lazy() if f.exists() else None


def _route_ids(routes: pl.DataFrame, route: str, route_key: str) -> list[str]:
    """route_ids matching ``route`` on ``route_key``; falls back to the other key."""
    route = str(route)
    keys = [route_key] + [k for k in ("route_short_name", "route_id") if k != route_key]
    for k in keys:
        if k in routes.columns:
            ids = routes.filter(pl.col(k).cast(pl.Utf8) == route)["route_id"].to_list()
            if ids:
                return ids
    return []


def load_route_feed(path: str | Path, route: str | None = None, route_key: str = "route_short_name") -> Feed:
    """Read a feed, keeping only what ``route`` needs (all routes when None)."""
    r = _Reader(path)
    lz = {n: r.scan(n) for n in TABLES}
    if lz["routes"] is None or lz["trips"] is None or lz["stop_times"] is None or lz["stops"] is None:
        raise ValueError(f"{path}: routes, trips, stop_times and stops are required")
    routes = lz["routes"].collect()
    if route is not None:
        rids = _route_ids(routes, route, route_key)
        routes = routes.filter(pl.col("route_id").is_in(rids))
    trips = lz["trips"].filter(pl.col("route_id").is_in(routes["route_id"].implode())).collect()
    st = (lz["stop_times"].filter(pl.col("trip_id").is_in(trips["trip_id"].implode()))
          .select([c for c in ("trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time",
                               "shape_dist_traveled") if c in lz["stop_times"].collect_schema().names()])
          .collect())
    stops = lz["stops"].filter(pl.col("stop_id").is_in(st["stop_id"].unique().implode())).collect()
    feed = Feed(routes=routes, trips=trips, stop_times=st, stops=stops)
    if lz["shapes"] is not None and "shape_id" in trips.columns:
        feed.shapes = lz["shapes"].filter(pl.col("shape_id").is_in(trips["shape_id"].drop_nulls().unique().implode())).collect()
    sids = trips["service_id"].unique().implode()
    for n in ("calendar", "calendar_dates"):
        if lz[n] is not None:
            setattr(feed, n, lz[n].filter(pl.col("service_id").is_in(sids)).collect())
    return feed


def active_services(feed: Feed, date) -> set[str]:
    """service_ids running on ``date`` (calendar + calendar_dates; either may be missing)."""
    d = _as_date(date)
    on: set[str] = set()
    cal = feed.calendar
    if cal is not None and len(cal):
        wd = WEEKDAYS[d.weekday()]
        on = set(cal.filter((pl.col("start_date") <= d) & (pl.col("end_date") >= d) & (pl.col(wd) == 1))
                 ["service_id"].to_list())
    cd = feed.calendar_dates
    if cd is not None and len(cd):
        x = cd.filter(pl.col("date") == d)
        on |= set(x.filter(pl.col("exception_type") == 1)["service_id"].to_list())
        on -= set(x.filter(pl.col("exception_type") == 2)["service_id"].to_list())
    return on


def service_dates(feed: Feed) -> tuple[dt.date, dt.date] | None:
    """First and last date any service of the feed runs on (None if no calendar at all)."""
    lo, hi = [], []
    if feed.calendar is not None and len(feed.calendar):
        lo.append(feed.calendar["start_date"].min())
        hi.append(feed.calendar["end_date"].max())
    if feed.calendar_dates is not None and len(feed.calendar_dates):
        add = feed.calendar_dates.filter(pl.col("exception_type") == 1)
        if len(add):
            lo.append(add["date"].min())
            hi.append(add["date"].max())
    return (min(lo), max(hi)) if lo else None


# ---------------------------------------------------------------- source
class GtfsSource:
    """One GTFS feed, or one feed per date (template or mapping). See module docstring."""

    def __init__(self, path: str | Path | None = None, *, dated: str | None = None,
                 feeds: Mapping | None = None, route_key: str = "route_short_name",
                 max_back_days: int = 14, cache_size: int = 8):
        if sum(x is not None for x in (path, dated, feeds)) != 1:
            raise ValueError("give exactly one of path, dated, feeds")
        self.path = str(path) if path is not None else None
        self.dated = dated
        self.feeds = {_as_date(k): str(v) for k, v in sorted((feeds or {}).items(), key=lambda kv: _as_date(kv[0]))}
        self.route_key = route_key
        self.max_back_days = max_back_days
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple, Feed] = OrderedDict()

    @classmethod
    def from_config(cls, gtfs) -> GtfsSource:
        """From a ``config.Gtfs`` (``path`` or ``dated``)."""
        if gtfs.dated:
            return cls(dated=gtfs.dated, route_key=gtfs.route_key)
        return cls(gtfs.path, route_key=gtfs.route_key)

    @staticmethod
    def _fmt(template: str, d: dt.date) -> str:
        return template.format(date=d.isoformat(), yyyymmdd=d.strftime("%Y%m%d"))

    def feed_path(self, date) -> str | None:
        """Path of the feed valid on ``date`` (None if none)."""
        d = _as_date(date)
        if self.path is not None:
            return self.path
        if self.dated is not None:
            for back in range(self.max_back_days + 1):
                p = self._fmt(self.dated, d - dt.timedelta(days=back))
                if Path(p).exists():
                    return p
            return None
        keys = [k for k in self.feeds if k <= d]
        return self.feeds[keys[-1]] if keys else None

    def feed_for(self, date, route: str | None = None) -> Feed | None:
        """The feed valid on ``date``, restricted to ``route`` when given (cached)."""
        p = self.feed_path(date)
        if p is None:
            return None
        return self.load(p, route)

    def load(self, path: str, route: str | None = None) -> Feed:
        key = (str(path), None if route is None else str(route), self.route_key)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        feed = load_route_feed(path, route, self.route_key)
        self._cache[key] = feed
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return feed

    def active_trips(self, date, route: str) -> pl.DataFrame:
        """Trips of ``route`` running on ``date`` (trips.txt columns); empty if no feed."""
        feed = self.feed_for(date, route)
        if feed is None:
            return pl.DataFrame(schema={"route_id": pl.Utf8, "service_id": pl.Utf8, "trip_id": pl.Utf8,
                                        "direction_id": pl.Int8, "shape_id": pl.Utf8})
        on = active_services(feed, date)
        return feed.trips.filter(pl.col("service_id").is_in(list(on)))

    def dates_by_feed(self, dates: Iterable) -> dict[str, list[dt.date]]:
        """Group dates by the feed path valid on each (dates without a feed are dropped)."""
        out: dict[str, list[dt.date]] = {}
        for d in sorted({_as_date(x) for x in dates}):
            p = self.feed_path(d)
            if p is not None:
                out.setdefault(p, []).append(d)
        return out
