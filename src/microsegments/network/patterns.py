"""Stop patterns of a route per service date, and their inter-stop links.

A *pattern* is a distinct stop sequence of one direction (its geometry is the shape most of its
trips use, cut at the stops by :func:`geometry.shape_slices`). ``pattern_uid`` hashes the stop
sequence and the rounded geometry, so it is stable across feeds that describe the same path.

For every service date and direction, the pattern with the most trips that day is the *main*
pattern. The others are classified against it:

* ``short``  : a contiguous sub-sequence of the main pattern (short working; ``parent_uid`` /
  ``parent_offset`` locate it on the main);
* ``detour`` : not contiguous, but at least half of its links are links of the main pattern
  (deviation in the middle, other first / last stop, skipped stops);
* ``other``  : anything else (depot runs, unrelated branches).

``Network.pattern_days`` keeps the per-date classification (extra columns ``kind``,
``parent_uid``, ``parent_offset`` beyond ``schema.PATTERN_DAY``); ``Network.patterns.kind`` is the
class a pattern has on most of its days. ``Network.versions`` lists runs of identical main patterns
(e.g. before / after a line is cut), with links added / removed against the display version (the
main pattern with the most days).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .. import schema
from .calendar import GtfsSource, _as_date, active_services
from .geometry import shape_slices
from .keys import KeyRegistry

KIND_ORDER = {"main": 0, "short": 1, "detour": 2, "other": 3}
_SEP = "\x1f"

PATTERN_DAY_EXTRA = {"kind": pl.Utf8, "parent_uid": pl.Utf8, "parent_offset": pl.Int16}
VERSIONS_SCHEMA = {
    "direction_id": pl.Int8,
    "version": pl.Int16,          # 0.. in date order, per direction
    "pattern_uid": pl.Utf8,
    "first": pl.Date,
    "last": pl.Date,
    "n_days": pl.UInt32,          # dates of the run
    "display": pl.Boolean,        # the main pattern with the most days over all runs
    "links_added": pl.List(pl.Utf8),     # link_keys of this version missing from the display version
    "links_removed": pl.List(pl.Utf8),   # link_keys of the display version missing here
}


def pattern_uid(stop_ids: list[str], coords) -> str:
    c = np.round(np.asarray(coords, dtype=np.float64).reshape(-1, 2), 4)
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(np.diff(c, axis=0) != 0, axis=1)
    txt = "|".join(stop_ids) + "#" + ";".join(f"{x:.4f},{y:.4f}" for x, y in c[keep])
    return hashlib.sha1(txt.encode()).hexdigest()[:10]


def _sub_offset(sub: list[str], main: list[str]) -> int | None:
    """Index of ``sub`` as a contiguous run inside ``main`` (None if not)."""
    n, m = len(sub), len(main)
    if n == 0 or n > m:
        return None
    for i in range(m - n + 1):
        if main[i] == sub[0] and main[i:i + n] == sub:
            return i
    return None


def classify(seq: list[str], main: list[str]) -> tuple[str, int | None]:
    """(kind, parent_offset) of a pattern relative to the day's main pattern."""
    if seq == main:
        return "main", None
    off = _sub_offset(seq, main)
    if off is not None:
        return "short", off
    main_links = set(zip(main[:-1], main[1:]))
    own = list(zip(seq[:-1], seq[1:]))
    if own and sum(lk in main_links for lk in own) >= 0.5 * len(own):
        return "detour", None
    return "other", None


@dataclass
class _FeedPatterns:
    trips: pl.DataFrame                    # trip_id, service_id, route_id, direction_id, pkey
    info: dict[str, dict]                  # pkey -> pattern info (uid, seq, names, slices, ...)


def _feed_patterns(feed, route_short_name: str | None) -> _FeedPatterns:
    st = feed.stop_times.sort("trip_id", "stop_sequence")
    seqs = st.group_by("trip_id", maintain_order=True).agg(pl.col("stop_id"))
    trips = feed.trips
    if "direction_id" not in trips.columns:
        trips = trips.with_columns(direction_id=pl.lit(0, pl.Int8))
    if "shape_id" not in trips.columns:
        trips = trips.with_columns(shape_id=pl.lit(None, pl.Utf8))
    trips = (trips.with_columns(pl.col("direction_id").fill_null(0).cast(pl.Int8))
             .join(seqs, on="trip_id", how="inner")
             # drop consecutive repeats of a stop (zero-length links)
             .with_columns(stop_id=pl.col("stop_id").list.eval(
                 pl.element().filter(pl.element() != pl.element().shift(1).fill_null(""))))
             .filter(pl.col("stop_id").list.len() >= 2)
             .with_columns(pkey=pl.col("direction_id").cast(pl.Utf8) + "#" + pl.col("stop_id").list.join(_SEP)))
    stops = feed.stops
    ll = {r[0]: (float(r[1]), float(r[2])) for r in stops.select("stop_id", "stop_lon", "stop_lat").iter_rows()}
    names = dict(zip(stops["stop_id"].to_list(), stops["stop_name"].to_list()))
    shapes = {}
    if feed.shapes is not None and len(feed.shapes):
        for (sid,), g in feed.shapes.sort("shape_id", "shape_pt_sequence").group_by("shape_id", maintain_order=True):
            shapes[sid] = g.select("shape_pt_lon", "shape_pt_lat").to_numpy().astype(np.float64)
    rsn = {}
    if "route_short_name" in feed.routes.columns:
        rsn = dict(zip(feed.routes["route_id"].to_list(), feed.routes["route_short_name"].to_list()))

    info: dict[str, dict] = {}
    agg = (trips.group_by("pkey").agg(
        pl.col("direction_id").first(), pl.col("stop_id").first().alias("seq"),
        pl.col("route_id").mode().sort().first(),
        pl.col("shape_id").drop_nulls().mode().sort().first(),
    ).sort("pkey"))
    for r in agg.iter_rows(named=True):
        seq = r["seq"]
        missing = [s for s in seq if s not in ll]
        if missing:
            raise ValueError(f"stops missing from stops.txt: {missing[:5]}")
        sl = shape_slices(np.array([ll[s] for s in seq]), shapes.get(r["shape_id"]))
        geom = np.vstack([c if k == 0 else c[1:] for k, c in enumerate(sl.coords)]) if sl.coords else np.zeros((0, 2))
        info[r["pkey"]] = {
            "uid": pattern_uid(seq, geom), "seq": seq, "names": [names.get(s) for s in seq],
            "direction_id": r["direction_id"], "route_id": r["route_id"],
            "route_short_name": rsn.get(r["route_id"], route_short_name), "slices": sl, "geometry": geom,
        }
    return _FeedPatterns(trips.select("trip_id", "service_id", "route_id", "direction_id", "pkey"), info)


def _ll_list(c) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in np.asarray(c).reshape(-1, 2)]


@dataclass
class Network:
    """Patterns, per-date usage and links of one route over a set of service dates."""
    route: str
    patterns: pl.DataFrame
    pattern_days: pl.DataFrame
    links: pl.DataFrame
    stops: pl.DataFrame                      # stop_id, stop_name, stop_lon, stop_lat
    registry: KeyRegistry
    missing_dates: list[dt.date] = field(default_factory=list)   # no feed or no trip that day
    # trip_id -> pattern_uid (+ direction_id, service_id) of every trip running on some date; lets
    # map matching use a fix's trip_id. Same trip_id in several feeds: one row per (trip, pattern).
    trips: pl.DataFrame | None = None

    # ------------------------------------------------------------ lookups
    @property
    def dates(self) -> list[dt.date]:
        return sorted(self.pattern_days["service_date"].unique().to_list())

    @property
    def directions(self) -> list[int]:
        return sorted(self.pattern_days["direction_id"].unique().to_list())

    def pattern(self, uid: str) -> dict:
        rows = self.patterns.filter(pl.col("pattern_uid") == uid)
        if not len(rows):
            raise KeyError(uid)
        return rows.row(0, named=True)

    def patterns_on(self, date) -> pl.DataFrame:
        """Patterns running on ``date`` with that day's n_trips / is_main / kind / parent."""
        d = _as_date(date)
        day = self.pattern_days.filter(pl.col("service_date") == d).drop("route_id")
        return (self.patterns.drop("kind", "parent_uid", "parent_offset", "direction_id")
                .join(day, on="pattern_uid", how="inner").sort("direction_id", "n_trips", descending=[False, True]))

    def mains(self) -> pl.DataFrame:
        """service_date, direction_id, pattern_uid of the main pattern per date and direction."""
        return (self.pattern_days.filter(pl.col("is_main"))
                .select("service_date", "direction_id", "pattern_uid").sort("direction_id", "service_date"))

    def main(self, date, direction: int) -> str | None:
        d = _as_date(date)
        x = self.pattern_days.filter((pl.col("service_date") == d) & (pl.col("direction_id") == direction)
                                     & pl.col("is_main"))
        return x["pattern_uid"][0] if len(x) else None

    def links_of(self, uid: str) -> pl.DataFrame:
        return self.links.filter(pl.col("pattern_uid") == uid).sort("link_idx")

    def link_days(self) -> pl.DataFrame:
        """Link x date validity: (service_date, direction_id, pattern_uid, link_key) for the links of
        each day's main pattern. A segment is aggregated only over the days its link appears here."""
        return self.mains().join(self.links.select("pattern_uid", "link_key", "link_idx"), on="pattern_uid") \
            .sort("direction_id", "service_date", "link_idx").drop("link_idx")

    def display_pattern(self, direction: int) -> str | None:
        """Main pattern with the most days (ties: the latest)."""
        m = self.mains().filter(pl.col("direction_id") == direction)
        if not len(m):
            return None
        c = (m.group_by("pattern_uid").agg(n=pl.len(), last=pl.col("service_date").max())
             .sort("n", "last", "pattern_uid", descending=True))
        return c["pattern_uid"][0]

    def versions(self, direction: int | None = None) -> pl.DataFrame:
        """Runs of consecutive dates (among the network's dates) sharing the same main pattern."""
        rows = []
        dirs = self.directions if direction is None else [direction]
        for d in dirs:
            m = self.mains().filter(pl.col("direction_id") == d).sort("service_date")
            if not len(m):
                continue
            disp = self.display_pattern(d)
            disp_keys = set(self.links_of(disp)["link_key"].to_list())
            runs = (m.with_columns(run=(pl.col("pattern_uid") != pl.col("pattern_uid").shift(1)).fill_null(True).cum_sum())
                    .group_by("run", maintain_order=True)
                    .agg(pl.col("pattern_uid").first(), first=pl.col("service_date").min(),
                         last=pl.col("service_date").max(), n_days=pl.len()))
            for k, r in enumerate(runs.iter_rows(named=True)):
                keys = self.links_of(r["pattern_uid"])["link_key"].to_list()
                rows.append({"direction_id": d, "version": k, "pattern_uid": r["pattern_uid"],
                             "first": r["first"], "last": r["last"], "n_days": r["n_days"],
                             "display": r["pattern_uid"] == disp,
                             "links_added": [x for x in keys if x not in disp_keys],
                             "links_removed": [x for x in self.links_of(disp)["link_key"].to_list() if x not in set(keys)]})
        return pl.DataFrame(rows, schema=VERSIONS_SCHEMA)


def build_network(source: GtfsSource | str, route: str, dates: Iterable, *,
                  registry: KeyRegistry | None = None, route_key: str = "route_short_name") -> Network:
    """Patterns, pattern-days and links of ``route`` on ``dates`` (see module docstring).

    ``source``: a :class:`GtfsSource` or a single feed path. ``registry``: link-key registry to
    reuse / extend (a new one otherwise); it is returned as ``Network.registry``."""
    if not isinstance(source, GtfsSource):
        source = GtfsSource(source, route_key=route_key)
    registry = registry if registry is not None else KeyRegistry()
    route = str(route)
    dates = sorted({_as_date(d) for d in dates})
    by_feed = source.dates_by_feed(dates)
    missing = sorted(set(dates) - {d for ds in by_feed.values() for d in ds})

    pat_rows: dict[str, dict] = {}
    day_rows: list[dict] = []
    link_rows: list[dict] = []
    stops_seen: dict[str, tuple] = {}
    trip_rows: list[pl.DataFrame] = []

    for path in sorted(by_feed, key=lambda p: by_feed[p][0]):
        feed = source.load(path, route)
        if not len(feed.trips):
            missing += by_feed[path]
            continue
        fp = _feed_patterns(feed, route)
        for r in feed.stops.select("stop_id", "stop_name", "stop_lon", "stop_lat").iter_rows():
            stops_seen.setdefault(r[0], r)
        used: set[str] = set()
        for d in by_feed[path]:
            on = active_services(feed, d)
            counts = (fp.trips.filter(pl.col("service_id").is_in(list(on)))
                      .group_by("pkey").agg(n=pl.len()))
            if not len(counts):
                missing.append(d)
                continue
            cnt = dict(zip(counts["pkey"].to_list(), counts["n"].to_list()))
            for direction in sorted({fp.info[k]["direction_id"] for k in cnt}):
                keys = [k for k in cnt if fp.info[k]["direction_id"] == direction]
                keys.sort(key=lambda k: (-cnt[k], -len(fp.info[k]["seq"]), fp.info[k]["uid"]))
                main = fp.info[keys[0]]
                for k in keys:
                    p = fp.info[k]
                    kind, off = classify(p["seq"], main["seq"])
                    day_rows.append({"service_date": d, "route_id": p["route_id"], "direction_id": direction,
                                     "pattern_uid": p["uid"], "n_trips": cnt[k], "is_main": k == keys[0],
                                     "kind": kind, "parent_uid": main["uid"] if kind in ("short", "detour") else None,
                                     "parent_offset": off})
                    used.add(k)
        uid_of = pl.DataFrame({"pkey": list(used), "pattern_uid": [fp.info[k]["uid"] for k in used]},
                              schema={"pkey": pl.Utf8, "pattern_uid": pl.Utf8})
        trip_rows.append(fp.trips.join(uid_of, on="pkey", how="inner")
                         .select(pl.col("trip_id").cast(pl.Utf8), pl.col("service_id").cast(pl.Utf8),
                                 "direction_id", "pattern_uid"))
        # register links of the busiest patterns first (their geometry seeds new keys)
        tot = Counter()
        for r in day_rows:
            tot[r["pattern_uid"]] += r["n_trips"]
        for k in sorted(used, key=lambda k: (-tot[fp.info[k]["uid"]], fp.info[k]["uid"])):
            p = fp.info[k]
            if p["uid"] in pat_rows:
                continue
            sl = p["slices"]
            s0 = np.concatenate([[0.0], np.cumsum(sl.lengths)])
            for i in range(len(p["seq"]) - 1):
                key = registry.key_for(p["seq"][i], p["seq"][i + 1], sl.coords[i], float(sl.lengths[i]))
                link_rows.append({"pattern_uid": p["uid"], "link_idx": i, "link_key": key,
                                  "from_stop": p["seq"][i], "to_stop": p["seq"][i + 1],
                                  "from_name": p["names"][i], "to_name": p["names"][i + 1],
                                  "s0_m": float(s0[i]), "len_m": float(sl.lengths[i]),
                                  "geometry": _ll_list(sl.coords[i])})
            pat_rows[p["uid"]] = {"pattern_uid": p["uid"], "route_id": p["route_id"],
                                  "route_short_name": p["route_short_name"], "direction_id": p["direction_id"],
                                  "stop_ids": p["seq"], "stop_names": p["names"],
                                  "length_m": float(sl.lengths.sum()), "geometry": _ll_list(p["geometry"])}

    days = pl.DataFrame(day_rows, schema={**schema.PATTERN_DAY, **PATTERN_DAY_EXTRA})
    # pattern-level kind: the class it has on most of its days (ties: main < short < detour < other)
    for uid, row in pat_rows.items():
        mine = [r for r in day_rows if r["pattern_uid"] == uid]
        kinds = Counter(r["kind"] for r in mine)
        kind = min(kinds, key=lambda k: (-kinds[k], KIND_ORDER[k]))
        parents = Counter((r["parent_uid"], r["parent_offset"]) for r in mine if r["kind"] == kind)
        parent, off = max(parents, key=lambda po: (parents[po], str(po[0]))) if kind in ("short", "detour") else (None, None)
        row.update(kind=kind, parent_uid=parent, parent_offset=off)
    patterns = pl.DataFrame(list(pat_rows.values()), schema=schema.PATTERN).sort("direction_id", "pattern_uid")
    links = pl.DataFrame(link_rows, schema=schema.LINK).sort("pattern_uid", "link_idx")
    stops = pl.DataFrame([dict(zip(("stop_id", "stop_name", "stop_lon", "stop_lat"), r)) for r in stops_seen.values()],
                         schema={"stop_id": pl.Utf8, "stop_name": pl.Utf8, "stop_lon": pl.Float64, "stop_lat": pl.Float64})
    return Network(route=route, patterns=patterns, pattern_days=days.sort("service_date", "direction_id", "n_trips",
                                                                          descending=[False, False, True]),
                   links=links, stops=stops, registry=registry, missing_dates=sorted(set(missing)),
                   trips=(pl.concat(trip_rows).unique(["trip_id", "pattern_uid"], keep="first") if trip_rows else
                          pl.DataFrame(schema={"trip_id": pl.Utf8, "service_id": pl.Utf8, "direction_id": pl.Int8,
                                               "pattern_uid": pl.Utf8})))
