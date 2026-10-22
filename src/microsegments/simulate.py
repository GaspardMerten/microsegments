"""Seeded synthetic line + vehicles with ground truth, for tests and examples.

One route ``S1`` (route_id = route_short_name = "S1"), two directions, ``n_links`` links of
300-600 m on a gently curved polyline near Brussels. Stop positions are whole metres along the line,
so 1-m truth bins never straddle a stop.

Coordinates. Everything is expressed per direction in *full-line metres* ``s`` (0 at the first stop
of the direction's full pattern, all stops). Short workings and the cut pattern are contiguous parts
of the full line, listed with their ``offset_m`` in ``SimResult.patterns``.

Motion. Rest-to-rest legs (cruise 30 km/h, accel/decel 1 m/s^2) between stops, dwell
Gamma(mean 20 s, CV 0.5) at intermediate stops. Injected hotspots (``SimResult.hotspots``):

* ``signal``: fixed signal (60 s cycle, red when epoch % 60 < 30) at the same physical point in both
  directions, all day: a vehicle that would cross it in red stops there until green.
* ``queue``: direction 0, hours 7-8 and 16-17: the vehicle stops at the queue tail and crawls 80 m,
  taking 30 s more than at cruise speed.
* ``holding``: direction 1, one stop, hours 19-22: +60 s dwell.

Ground truth. ``truth_seconds``: seconds spent per (direction_id, s_bin, service_date, hour), bin k
covering s in (k, k+1] (a vehicle standing at a stop at s=k is in bin k-1, the end of the link that
arrives there). Counted from departure at the first stop to arrival at the last (no layover);
hour = local hour (service-day convention, 24+ after midnight) of the middle of each 1-m step.
``passages``: trips per (direction_id, link_idx (full line), service_date, hour of departure from the
link's first stop). ``stop_events``: exact STOP_EVENT rows.

Outputs. ``mode="stib"``: polls every 20 +- 2 s, 2 % dropped, outages (no poll), frozen windows
(polls flagged frozen, no rows); rows = last stop passed + feed metres since (x1.03, sigma 5 m,
rounded, 0 when standing), terminus code, no vehicle id; feed ids zero-padded ("01003").
``mode="gtfsrt"``: per-vehicle fixes every U(15, 60) s (``gtfsrt_interval``), lat/lon with sigma noise,
vehicle_id, trip_id, direction_id, bearing; none during outages / frozen windows.
Observations carry extra ``true_*`` columns (true direction, s, trip key, vehicle) for tests.
"""
from __future__ import annotations

import datetime as dt
import functools
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from . import schema
from .io.timeutil import local_epoch

ROUTE = "S1"
VMAX = 30 / 3.6
ACC = 1.0
SIGNAL_CYCLE, SIGNAL_RED = 60.0, 30.0
QUEUE_LEN, QUEUE_EXTRA = 80, 30.0
QUEUE_HOURS = (7, 8, 16, 17)
HOLD_HOURS = (19, 20, 21, 22)
HOLD_EXTRA = 60.0
OTHER_LINES_ROWS = 700
STOP_NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India",
              "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa"]
HEADWAY_MIN = {5: 10, 6: 8, 7: 6, 8: 6, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 6, 17: 6,
               18: 8, 19: 10, 20: 10, 21: 10, 22: 10, 23: 10}
R_EARTH = 6_371_008.8


@dataclass
class SimResult:
    mode: str
    gtfs_dir: Path
    dates: list[dt.date]
    tz: str
    observations: pl.DataFrame          # OBSERVATION + true_direction_id, true_s_m, true_trip_key, true_vehicle_id
    snapshots: pl.DataFrame             # SNAPSHOT (feed polls)
    raw: pl.DataFrame | None            # stib mode: compact rows ts, line, dir, point, dist (raw feed ids)
    stop_events: pl.DataFrame           # STOP_EVENT, exact
    truth_seconds: pl.DataFrame         # direction_id, s_bin, service_date, hour, seconds
    passages: pl.DataFrame              # direction_id, link_idx, service_date, hour, n
    trips: pl.DataFrame                 # one row per trip run
    stops: pl.DataFrame                 # direction_id, idx, stop_id, stop_name, s_m (full line)
    patterns: pl.DataFrame              # name, direction_id, kind, stop_ids, offset_m, length_m, first_date, last_date
    hotspots: list[dict]                # injected hotspots (direction_id, kind, s0_m, s1_m, link_idx, hours, extra_s)
    outages: list[tuple[float, float]]  # [start, end) epoch s, no poll
    frozen: list[tuple[float, float]]   # [start, end) epoch s, frozen polls
    line_length_m: int
    params: dict = field(default_factory=dict)

    def link_bounds(self, direction_id: int) -> list[tuple[int, int]]:
        s = self.stops.filter(pl.col("direction_id") == direction_id).sort("idx")["s_m"].to_list()
        return list(zip(s[:-1], s[1:]))

    def truth_link_seconds(self) -> pl.DataFrame:
        """Truth seconds per (direction_id, link_idx (full line), service_date, hour)."""
        frames = []
        for d in (0, 1):
            b = self.link_bounds(d)
            starts = np.array([x for x, _ in b])
            t = self.truth_seconds.filter(pl.col("direction_id") == d)
            li = np.searchsorted(starts, t["s_bin"].to_numpy(), side="right") - 1
            frames.append(t.with_columns(pl.Series("link_idx", li, dtype=pl.Int16)))
        return (pl.concat(frames).group_by("direction_id", "link_idx", "service_date", "hour")
                .agg(pl.col("seconds").sum()).sort("direction_id", "link_idx", "service_date", "hour"))

    def write_stib(self, out_dir: str | Path) -> list[Path]:
        """Write the compact per-day STIB parquet (``<YYYYMMDD>.parquet`` + ``.snaps.parquet``)."""
        from .io.stib import snapshots_to_compact
        from .io.timeutil import add_local_time
        if self.raw is None:
            raise ValueError("write_stib needs mode='stib'")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        raw = add_local_time(self.raw.with_columns(
            pl.col("ts").cast(pl.Int64).mul(1_000_000).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("_t")),
            self.tz, ts_col="_t")
        sn = add_local_time(self.snapshots, self.tz)
        for d in self.dates:
            p = out / f"{d:%Y%m%d}.parquet"
            raw.filter(pl.col("service_date") == d).select("ts", "line", "dir", "point", "dist").write_parquet(p)
            q = out / f"{d:%Y%m%d}.snaps.parquet"
            snapshots_to_compact(sn.filter(pl.col("service_date") == d)).write_parquet(q)
            paths += [p, q]
        return paths


# ------------------------------------------------------------------ geometry
def _curve(n_links: int, rng: np.random.Generator):
    """Dense polyline (lon, lat), its cumulative haversine metres, integer stop positions."""
    lens = rng.integers(300, 601, n_links)
    total = int(lens.sum())
    x = np.linspace(0, total * 1.02, int(total * 1.02 / 5) + 1)
    y = 120.0 * np.sin(x / 900.0) + 0.15 * x
    lat0, lon0 = 50.8466, 4.3528
    lat = lat0 + np.degrees(y / R_EARTH)
    lon = lon0 + np.degrees(x / (R_EARTH * math.cos(math.radians(lat0))))
    cum = np.concatenate([[0.0], np.cumsum(_hav(lon[:-1], lat[:-1], lon[1:], lat[1:]))])
    stops = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    keep = cum <= stops[-1] + 5
    return lon[keep], lat[keep], cum[keep], stops


def _hav(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


class _Line:
    def __init__(self, lon, lat, cum, stops_phys):
        self.lon, self.lat, self.cum = lon, lat, cum
        self.L = int(stops_phys[-1])
        self.stops_phys = stops_phys

    def phys(self, d: int, s):
        return np.asarray(s, dtype=float) if d == 0 else self.L - np.asarray(s, dtype=float)

    def lonlat(self, d: int, s):
        p = self.phys(d, s)
        return np.interp(p, self.cum, self.lon), np.interp(p, self.cum, self.lat)

    def bearing(self, d: int, s):
        p = self.phys(d, s)
        lo1, la1 = np.interp(p - 2, self.cum, self.lon), np.interp(p - 2, self.cum, self.lat)
        lo2, la2 = np.interp(p + 2, self.cum, self.lon), np.interp(p + 2, self.cum, self.lat)
        b = np.degrees(np.arctan2((lo2 - lo1) * math.cos(math.radians(50.85)), la2 - la1)) % 360
        return b if d == 0 else (b + 180) % 360

    def polyline(self, d: int, s0: float, s1: float) -> list[tuple[float, float]]:
        """Shape points (lon, lat) from s0 to s1 in direction d."""
        p0, p1 = sorted((float(self.phys(d, s0)), float(self.phys(d, s1))))
        inner = (self.cum > p0) & (self.cum < p1)
        lo = np.concatenate([[np.interp(p0, self.cum, self.lon)], self.lon[inner], [np.interp(p1, self.cum, self.lon)]])
        la = np.concatenate([[np.interp(p0, self.cum, self.lat)], self.lat[inner], [np.interp(p1, self.cum, self.lat)]])
        pts = list(zip(lo.tolist(), la.tolist()))
        return pts if d == 0 else pts[::-1]


# ------------------------------------------------------------------ kinematics
@functools.lru_cache(maxsize=4096)
def _rest(D: int) -> np.ndarray:
    """Time (s) at each whole metre 0..D of a rest-to-rest move over D metres."""
    d = np.arange(D + 1, dtype=float)
    da = VMAX * VMAX / (2 * ACC)
    if D >= 2 * da:
        ta = VMAX / ACC
        T = 2 * ta + (D - 2 * da) / VMAX
        t = np.where(d <= da, np.sqrt(2 * d / ACC),
                     np.where(d <= D - da, ta + (d - da) / VMAX, T - np.sqrt(np.maximum(2 * (D - d) / ACC, 0))))
    else:
        h = D / 2
        T = 2 * math.sqrt(2 * h / ACC) if D else 0.0
        t = np.where(d <= h, np.sqrt(2 * d / ACC), T - np.sqrt(np.maximum(2 * (D - d) / ACC, 0)))
    t.setflags(write=False)
    return t


def _hhmmss(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _windows(spec, dates, tz, rng, n_default, dur_range, kind):
    """Outage / frozen windows: explicit [(day_idx, 'HH:MM' | hours, minutes)] or n random."""
    out = []
    if spec is None:
        spec = []
        for _ in range(n_default):
            day = int(rng.integers(0, len(dates)))
            h = float(rng.uniform(6, 21))
            spec.append((day, h, float(rng.uniform(*dur_range))))
    for day, start, minutes in spec:
        if isinstance(start, str):
            hh, mm = start.split(":")[:2]
            start = int(hh) + int(mm) / 60
        a = local_epoch(dates[int(day)], float(start) * 3600, tz)
        out.append((a, a + float(minutes) * 60))
    return out


def _in_windows(t: np.ndarray, wins) -> np.ndarray:
    m = np.zeros(t.shape, bool)
    for a, b in wins:
        m |= (t >= a) & (t < b)
    return m


# ------------------------------------------------------------------ line, patterns, timetable
@dataclass
class _Net:
    """The simulated line: geometry, stops per direction (full-line metres), hotspots, patterns, timetable."""
    line: _Line
    n_links: int
    stop_s: dict             # direction -> full-line stop positions
    phys_of: dict            # direction -> physical stop index of each stop
    stop_id: dict
    stop_name: dict
    pat_defs: dict           # (direction, kind) -> (first stop idx, last stop idx) in direction order
    services: list           # (service_id, first day idx, last day idx)
    deps: list               # (departure local s, direction, kind), sorted
    hs: list = field(default_factory=list)
    sig: dict = field(default_factory=dict)      # direction -> signal position
    queue: dict = field(default_factory=dict)    # direction -> (queue start, queue end)
    hold: dict = field(default_factory=dict)     # direction -> holding stop idx

    @property
    def n_stops(self) -> int:
        return self.n_links + 1

    def pattern(self, sid: str, d: int, kind: str) -> tuple[str, int, int]:
        """Pattern kind and stop span of a departure on service ``sid`` (CUT: the main trips are cut)."""
        pk = "cut" if (sid == "CUT" and kind == "main") else kind
        return (pk, *self.pat_defs[(d, pk)])


def _net(n_links: int, rng, days: int, change_day, short_working: bool, hotspots: bool) -> _Net:
    lon, lat, cum, sp = _curve(n_links, rng)
    line = _Line(lon, lat, cum, sp)
    n_stops = n_links + 1
    # physical stop j is index j in dir 0, n-1-j in dir 1
    phys_of = {0: list(range(n_stops)), 1: list(range(n_stops))[::-1]}
    cut, short_end = 2, min(5, n_links - 1)
    services = [("BASE", 0, days - 1)]
    if change_day is not None and 0 < change_day < days:
        services = [("BASE", 0, change_day - 1), ("CUT", change_day, days - 1)]
    net = _Net(line=line, n_links=n_links, stop_s={0: sp.copy(), 1: (line.L - sp[::-1]).astype(int)}, phys_of=phys_of,
               stop_id={d: [f"{1000 * (d + 1) + j + 1}" for j in phys_of[d]] for d in (0, 1)},
               stop_name={d: [STOP_NAMES[j % len(STOP_NAMES)] for j in phys_of[d]] for d in (0, 1)},
               pat_defs={(0, "main"): (0, n_stops - 1), (1, "main"): (0, n_stops - 1),
                         (0, "short"): (0, short_end), (1, "short"): (n_stops - 1 - short_end, n_stops - 1),
                         (0, "cut"): (0, n_stops - 1 - cut), (1, "cut"): (cut, n_stops - 1)},
               services=services, deps=_departures(short_working))
    if hotspots:
        _add_hotspots(net, sp)
    return net


def _add_hotspots(net: _Net, sp) -> None:
    """The injected hotspots (full-line coordinates): a signal, a peak queue, an evening holding."""
    n_links, L, stop_s = net.n_links, net.line.L, net.stop_s
    k = min(2, n_links - 1)
    s_sig0 = int(sp[k] + round(0.6 * (sp[k + 1] - sp[k])))
    net.sig = {0: s_sig0, 1: L - s_sig0}
    for d in (0, 1):
        li = int(np.searchsorted(stop_s[d], net.sig[d], side="right") - 1)
        net.hs.append({"kind": "signal", "direction_id": d, "s0_m": float(net.sig[d]), "s1_m": float(net.sig[d]),
                       "link_idx": li, "hours": list(range(5, 28)), "extra_s": SIGNAL_RED / 2,
                       "class": "infrastructure", "zone": "running"})
    kq = min(5, n_links - 1)
    q1 = int(sp[kq] + round(0.7 * (sp[kq + 1] - sp[kq])))
    net.queue = {0: (q1 - QUEUE_LEN, q1)}
    net.hs.append({"kind": "queue", "direction_id": 0, "s0_m": float(q1 - QUEUE_LEN), "s1_m": float(q1),
                   "link_idx": kq, "hours": list(QUEUE_HOURS), "extra_s": QUEUE_EXTRA,
                   "class": "congestion", "zone": "running"})
    kh = min(4, n_links - 1)
    net.hold = {1: kh}
    net.hs.append({"kind": "holding", "direction_id": 1, "s0_m": float(stop_s[1][kh]), "s1_m": float(stop_s[1][kh]),
                   "link_idx": kh - 1, "stop_idx": kh, "stop_id": net.stop_id[1][kh], "hours": list(HOLD_HOURS),
                   "extra_s": HOLD_EXTRA, "class": "holding", "zone": "stop"})


def _departures(short_working: bool) -> list[tuple[float, int, str]]:
    """Departures of a service day (local s, direction, kind): every 4th daytime trip is a short working."""
    deps = []
    for d in (0, 1):
        t = 5 * 3600 + (180 if d else 0)
        n = 0
        while t < 24 * 3600:
            kind = "short" if (short_working and 6 <= t / 3600 < 20 and n % 4 == 3) else "main"
            deps.append((float(t), d, kind))
            n += 1
            t += HEADWAY_MIN[int(t // 3600)] * 60
    deps.sort()
    return deps


def _nominal(stop_s, d: int, a: int, b: int) -> list[tuple[float, float]]:
    """Nominal (arrival, departure) offsets at stops a..b (mean dwell 20 s)."""
    t, out_t = 0.0, [(0.0, 0.0)]
    for i in range(a, b):
        t += float(_rest(int(stop_s[d][i + 1] - stop_s[d][i]))[-1])
        arr = t
        if i + 1 < b:
            t += 20.0
        out_t.append((arr, t))
    return out_t


def _write_gtfs(net: _Net, out: Path, dates, tz: str) -> dict[str, set[int]]:
    """The static GTFS of the line; returns the patterns used, with the day indices they run on."""
    trips_rows, st_rows, shape_rows, used_pats = [], [], [], {}
    stop_s, stop_id = net.stop_s, net.stop_id
    for sid, d0, d1 in net.services:
        for j, (tdep, d, kind) in enumerate(net.deps):
            pk, a, b = net.pattern(sid, d, kind)
            name = f"d{d}_{pk}"
            used_pats.setdefault(name, set()).update(range(d0, d1 + 1))
            trip_id = f"{sid}_{d}_{j:04d}"
            trips_rows.append({"route_id": ROUTE, "service_id": sid, "trip_id": trip_id, "direction_id": d,
                               "shape_id": name, "trip_headsign": net.stop_name[d][b]})
            for q, (i, (arr, dep)) in enumerate(zip(range(a, b + 1), _nominal(stop_s, d, a, b))):
                st_rows.append({"trip_id": trip_id, "arrival_time": _hhmmss(tdep + arr),
                                "departure_time": _hhmmss(tdep + dep), "stop_id": stop_id[d][i],
                                "stop_sequence": q + 1,
                                "shape_dist_traveled": float(stop_s[d][i] - stop_s[d][a])})
    for name in used_pats:
        d, pk = int(name[1]), name.split("_")[1]
        a, b = net.pat_defs[(d, pk)]
        pts = net.line.polyline(d, stop_s[d][a], stop_s[d][b])
        cd = np.concatenate([[0.0], np.cumsum(_hav(*np.array(pts[:-1]).T, *np.array(pts[1:]).T))])
        for q, ((x, y), c) in enumerate(zip(pts, cd)):
            shape_rows.append({"shape_id": name, "shape_pt_lat": y, "shape_pt_lon": x,
                               "shape_pt_sequence": q + 1, "shape_dist_traveled": float(c)})
    stops_rows = []
    for d in (0, 1):
        for i in range(net.n_stops):
            x, y = net.line.lonlat(d, stop_s[d][i])
            stops_rows.append({"stop_id": stop_id[d][i], "stop_name": net.stop_name[d][i],
                               "stop_lat": float(y), "stop_lon": float(x)})
    pl.DataFrame(stops_rows).write_csv(out / "stops.txt")
    pl.DataFrame([{"agency_id": "SIM", "agency_name": "Simulated Transit", "agency_url": "https://example.org",
                   "agency_timezone": tz}]).write_csv(out / "agency.txt")
    pl.DataFrame([{"route_id": ROUTE, "agency_id": "SIM", "route_short_name": ROUTE, "route_long_name": "Simulated line",
                   "route_type": 0}]).write_csv(out / "routes.txt")
    pl.DataFrame(trips_rows).write_csv(out / "trips.txt")
    pl.DataFrame(st_rows).write_csv(out / "stop_times.txt")
    pl.DataFrame(shape_rows).write_csv(out / "shapes.txt")
    pl.DataFrame([{"service_id": sid, "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 1,
                   "saturday": 1, "sunday": 1, "start_date": dates[a].strftime("%Y%m%d"),
                   "end_date": dates[b].strftime("%Y%m%d")} for sid, a, b in net.services]).write_csv(out / "calendar.txt")
    return used_pats


def _polls(dates, tz: str, rng, poll_s: float, poll_jitter_s: float, drop_rate: float, outage_w):
    """Feed poll times (epoch s) of every service day, after drops and outages."""
    polls = []
    for day in dates:
        a = local_epoch(day, 4 * 3600, tz)
        b = local_epoch(day + dt.timedelta(days=1), 4 * 3600, tz)
        n = int((b - a) / poll_s) + 10
        t = a + np.round(np.arange(n) * poll_s + rng.uniform(-poll_jitter_s, poll_jitter_s, n), 0)
        polls.append(t[(t >= a) & (t < b)])
    poll_t = np.unique(np.concatenate(polls))
    keep = (rng.random(poll_t.size) >= drop_rate) & ~_in_windows(poll_t, outage_w)
    return poll_t[keep]


# ------------------------------------------------------------------ vehicles
def _obstacles(net: _Net, d: int, sA: int, sB: int) -> list[tuple[str, int]]:
    """The signal and the queue tail inside the link sA -> sB, in order."""
    obstacles = []
    if d in net.sig and sA < net.sig[d] < sB:
        obstacles.append(("signal", net.sig[d]))
    if d in net.queue and sA < net.queue[d][0] and net.queue[d][1] < sB:
        obstacles.append(("queue", net.queue[d][0]))
    return sorted(obstacles, key=lambda o: o[1])


def _pass_obstacle(net: _Net, d: int, kind: str, pos: int, cur: int, sB: int, t: float, day0: float, S, T):
    """Run up to an obstacle when it stops the vehicle (red signal, queue hours); appends the knots to S, T and
    returns the new (position, time)."""
    t_through = t + float(_rest(sB - cur)[pos - cur])
    if kind == "signal":
        if (t_through % SIGNAL_CYCLE) < SIGNAL_RED:
            leg = _rest(pos - cur)
            S.append(np.arange(cur, pos + 1))
            T.append(t + leg)
            t_arr = t + float(leg[-1])
            green = math.floor(t_arr / SIGNAL_CYCLE) * SIGNAL_CYCLE + SIGNAL_RED
            t = max(t_arr, green if (t_arr % SIGNAL_CYCLE) < SIGNAL_RED else t_arr)
            S.append(np.array([pos]))
            T.append(np.array([t]))
            cur = pos
    elif int((t_through - day0) // 3600) + 4 in QUEUE_HOURS:
        q0, q1 = net.queue[d]
        leg = _rest(q0 - cur)
        S.append(np.arange(cur, q0 + 1))
        T.append(t + leg)
        t = t + float(leg[-1])
        crawl = QUEUE_LEN / VMAX + QUEUE_EXTRA
        S.append(np.arange(q0, q1 + 1))
        T.append(t + np.arange(QUEUE_LEN + 1) * (crawl / QUEUE_LEN))
        t += crawl
        cur = q1
    return cur, t


def _trajectory(net: _Net, rng, d: int, a_i: int, b_i: int, dep: float, day0: float):
    """Knots (S whole metres, T epoch s) of one trip from stop a_i to b_i, and its stop events
    [(s, arrival, departure)]; draws one Gamma dwell per intermediate stop."""
    S_parts, T_parts = [], []
    t = dep
    ev = [(net.stop_s[d][a_i], None, dep)]
    for i in range(a_i, b_i):
        sA, sB = int(net.stop_s[d][i]), int(net.stop_s[d][i + 1])
        cur = sA
        for kind, pos in _obstacles(net, d, sA, sB):
            cur, t = _pass_obstacle(net, d, kind, pos, cur, sB, t, day0, S_parts, T_parts)
        leg = _rest(sB - cur)
        S_parts.append(np.arange(cur, sB + 1))
        T_parts.append(t + leg)
        t = t + float(leg[-1])
        arr = t
        if i + 1 < b_i:
            dwell = float(rng.gamma(4.0, 5.0))
            if d in net.hold and i + 1 == net.hold[d] and int((t - day0) // 3600) + 4 in HOLD_HOURS:
                dwell += HOLD_EXTRA
            t += dwell
            S_parts.append(np.array([sB]))
            T_parts.append(np.array([t]))
        ev.append((sB, arr, t if i + 1 < b_i else None))
    S = np.concatenate(S_parts).astype(np.int64)
    T = np.concatenate(T_parts)
    dup = np.concatenate([[False], (np.diff(S) == 0) & (np.diff(T) == 0)])
    return S[~dup], T[~dup], ev


def _add_truth(truth, pas, S, T, ev, d: int, di: int, a_i: int, b_i: int, day0: float) -> None:
    """Seconds per 1-m bin and hour, and the passage of each link (hour of departure from its first stop)."""
    hours_n = truth.shape[2]
    ds, dtt = np.diff(S), np.diff(T)
    bins = np.where(ds > 0, S[:-1], S[:-1] - 1)
    mid = (T[:-1] + T[1:]) / 2
    hrs = np.clip(((mid - day0) // 3600).astype(int) + 4, 0, hours_n - 1)
    ok = bins >= 0
    np.add.at(truth[d, di], (hrs[ok], bins[ok]), dtt[ok])
    for q, i in enumerate(range(a_i, b_i)):
        pas[d, di, min(int((ev[q][2] - day0) // 3600) + 4, hours_n - 1), i] += 1


class _Fleet:
    """Vehicles waiting at each physical stop: a trip takes one that is free 30 s before its departure, else a new one."""

    def __init__(self):
        self.n = 0
        self.pools: dict[int, list[list]] = {}   # physical stop -> [[available_t, vid]]

    def new_day(self):
        self.pools = {}

    def take(self, p_start: int, p_end: int, dep: float, arr_end: float, lay: float) -> tuple[str, float]:
        """The vehicle and the start of its layover before ``dep``; it is free again at p_end after arr_end + 60 s."""
        pool = self.pools.setdefault(p_start, [])
        pool.sort()
        if pool and pool[0][0] <= dep - 30:
            av, vid = pool.pop(0)
            t_lay0 = max(av, dep - lay)
        else:
            self.n += 1
            vid = f"V{self.n:03d}"
            t_lay0 = dep - lay
        self.pools.setdefault(p_end, []).append([arr_end + 60.0, vid])
        return vid, t_lay0


def _obs_times(mode: str, rng, live_t, t0: float, t1: float, interval, wins) -> np.ndarray:
    """Observation times of a vehicle over [t0, t1): the live polls (stib) or its own GTFS-RT fixes."""
    if mode == "stib":
        lo, hi = np.searchsorted(live_t, [t0, t1])
        return live_t[lo:hi]
    ilo, ihi = interval
    n = int((t1 - t0) / ilo) + 2
    ts = t0 + rng.uniform(0, ihi) + np.concatenate([[0], np.cumsum(rng.uniform(ilo, ihi, n))])
    ts = np.round(ts[ts < t1], 0)
    return ts[~_in_windows(ts, wins[0]) & ~_in_windows(ts, wins[1])]


def _run_vehicles(net: _Net, rng, dates, tz: str, mode: str, live_t, interval, wins):
    """Every trip of every day: truth (2, D, 28, L + 1) seconds, passages (2, D, 28, links), stop events, trips,
    and the true observations (dicts of arrays per trip)."""
    truth = np.zeros((2, len(dates), 28, net.line.L + 1))
    pas = np.zeros((2, len(dates), 28, net.n_links), dtype=np.int64)
    ev_rows, trip_rows, obs_parts = [], [], []
    fleet = _Fleet()
    for di, day in enumerate(dates):
        sid = next(s for s, a, b in net.services if a <= di <= b)
        day0 = local_epoch(day, 4 * 3600, tz)
        fleet.new_day()
        for j, (tdep, d, kind) in enumerate(net.deps):
            pk, a_i, b_i = net.pattern(sid, d, kind)
            trip_id = f"{sid}_{d}_{j:04d}"
            tkey = f"{day:%Y%m%d}:{trip_id}"
            dep = local_epoch(day, tdep, tz) + float(rng.uniform(0, 45))
            S, T, ev = _trajectory(net, rng, d, a_i, b_i, dep, day0)
            arr_end = float(T[-1])
            _add_truth(truth, pas, S, T, ev, d, di, a_i, b_i, day0)
            for q, (_, arr, dpt) in enumerate(ev):
                ev_rows.append((day, tkey, net.stop_id[d][a_i + q], a_i + q + 1,
                                None if arr is None else int(round(arr * 1e6)),
                                None if dpt is None else int(round(dpt * 1e6))))
            vid, t_lay0 = fleet.take(net.phys_of[d][a_i], net.phys_of[d][b_i], dep, arr_end, float(rng.uniform(60, 240)))
            trip_rows.append((day, tkey, trip_id, vid, d, f"d{d}_{pk}", dep, arr_end))
            ts = _obs_times(mode, rng, live_t, t_lay0, arr_end, interval, wins)
            if ts.size == 0:
                continue
            pos = np.where(ts < dep, float(net.stop_s[d][a_i]), np.interp(ts, T, S.astype(float)))
            obs_parts.append({"ts": ts, "s": pos, "d": np.full(ts.size, d), "a": np.full(ts.size, a_i),
                              "b": np.full(ts.size, b_i), "trip": np.full(ts.size, tkey, dtype=object),
                              "trip_id": np.full(ts.size, trip_id, dtype=object),
                              "vid": np.full(ts.size, vid, dtype=object)})
    return truth, pas, ev_rows, trip_rows, obs_parts


# ------------------------------------------------------------------ feeds
def _stib_feed(net: _Net, cat: dict, rng, terminus_alias: dict, feed_scale: float, noise_m: float):
    """The compact STIB rows (last stop passed + noisy feed metres since it, terminus code) of the true
    positions, and the observations parsed from them; ``cat`` reordered by time."""
    from .io.stib import compact_to_observations
    dd, s = cat["d"], cat["s"]
    base = np.empty(s.size)
    term = np.empty(s.size, dtype=object)
    point = np.empty(s.size, dtype=object)
    for d in (0, 1):
        m = dd == d
        k = np.searchsorted(net.stop_s[d], s[m] + 1e-9, side="right") - 1
        k = np.minimum(np.maximum(k, cat["a"][m]), cat["b"][m])
        base[m] = net.stop_s[d][k]
        ids = np.array(["0" + x for x in net.stop_id[d]], dtype=object)
        point[m] = ids[k]
        tb = ids[cat["b"][m]]
        if d in terminus_alias:
            tb = np.full(tb.size, str(terminus_alias[d]), dtype=object)
        term[m] = tb
    since = s - base
    dist = np.where(since > 1e-6, np.maximum(since * feed_scale + rng.normal(0, noise_m, s.size), 0), 0.0)
    raw = pl.DataFrame({"ts": cat["ts"].astype(np.int64), "line": np.full(s.size, ROUTE, dtype=object),
                        "dir": term, "point": point, "dist": np.round(dist).astype(np.int64)},
                       schema={"ts": pl.Int64, "line": pl.Utf8, "dir": pl.Utf8, "point": pl.Utf8, "dist": pl.Int64})
    order = np.argsort(cat["ts"], kind="stable")
    raw = raw[order]
    return compact_to_observations(raw.lazy()).collect(), raw, {k: v[order] for k, v in cat.items()}


def _gtfsrt_feed(net: _Net, cat: dict, rng, noise_m: float):
    """GTFS-RT vehicle positions (noisy lat/lon, bearing) of the true positions; ``cat`` reordered by time, vehicle."""
    n = cat["s"].size
    lo_, la_, br = np.empty(n), np.empty(n), np.empty(n)
    for d in (0, 1):
        m = cat["d"] == d
        lo_[m], la_[m] = net.line.lonlat(d, cat["s"][m])
        br[m] = net.line.bearing(d, cat["s"][m])
    la_ = la_ + np.degrees(rng.normal(0, noise_m, la_.size) / R_EARTH)
    lo_ = lo_ + np.degrees(rng.normal(0, noise_m, lo_.size) / (R_EARTH * math.cos(math.radians(50.85))))
    order = np.lexsort((cat["vid"].astype(str), cat["ts"]))
    cat = {k: v[order] for k, v in cat.items()}
    obs = pl.DataFrame({
        "ts": (cat["ts"] * 1e6).astype(np.int64), "route_id": np.full(order.size, ROUTE, dtype=object),
        "vehicle_id": cat["vid"], "trip_id": cat["trip_id"], "direction_id": cat["d"].astype(np.int8),
        "lat": la_[order], "lon": lo_[order], "bearing": br[order].astype(np.float32),
    }).with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    return schema.conform(obs, schema.OBSERVATION, schema.OBSERVATION_REQUIRED), cat


def _observations(net: _Net, obs_parts, mode: str, rng, terminus_alias, feed_scale, noise_m):
    """The feed's observations (with the true_* columns) and, in stib mode, its raw compact rows."""
    if not obs_parts:
        return schema.empty(schema.OBSERVATION), None
    cat = {k: np.concatenate([p[k] for p in obs_parts]) for k in obs_parts[0]}
    raw = None
    if mode == "stib":
        obs, raw, cat = _stib_feed(net, cat, rng, terminus_alias, feed_scale, noise_m)
    else:
        obs, cat = _gtfsrt_feed(net, cat, rng, noise_m)
    return obs.with_columns(
        pl.Series("true_direction_id", cat["d"].astype(np.int8)),
        pl.Series("true_s_m", cat["s"].astype(np.float32)),
        pl.Series("true_trip_key", cat["trip"].astype(str)),
        pl.Series("true_vehicle_id", cat["vid"].astype(str)),
    ), raw


# ------------------------------------------------------------------ tables
def _snapshots(poll_t, poll_frozen, raw) -> pl.DataFrame:
    n_rows = np.full(poll_t.size, OTHER_LINES_ROWS, dtype=np.int64)
    if raw is not None:
        u, c = np.unique(raw["ts"].to_numpy(), return_counts=True)
        n_rows[np.searchsorted(poll_t, u)] += c
    return pl.DataFrame({"ts": (poll_t * 1e6).astype(np.int64), "n_rows": n_rows, "frozen": poll_frozen}).with_columns(
        pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")).cast(schema.SNAPSHOT)


def _stop_events(ev_rows) -> pl.DataFrame:
    ev = pl.DataFrame(ev_rows, schema={"service_date": pl.Date, "trip_key": pl.Utf8, "stop_id": pl.Utf8,
                                       "stop_sequence": pl.Int32, "arrival": pl.Int64, "departure": pl.Int64},
                      orient="row").with_columns(
        *[pl.col(c).cast(pl.Datetime("us")).dt.replace_time_zone("UTC") for c in ("arrival", "departure")],
        pl.lit(ROUTE).alias("route_id"), pl.lit(True).alias("observed"))
    return ev.select([pl.col(k).cast(v) for k, v in schema.STOP_EVENT.items()])


def _nonzero(a: np.ndarray, dates, last: str, last_t, val: str, val_t) -> pl.DataFrame:
    """The non-zero cells of a (direction, day, hour, last) array as rows."""
    nz = np.nonzero(a)
    return pl.DataFrame({"direction_id": nz[0].astype(np.int8), last: nz[3].astype(last_t),
                         "service_date": np.array(dates, dtype="datetime64[D]")[nz[1]],
                         "hour": nz[2].astype(np.int8), val: a[nz].astype(val_t)}
                        ).with_columns(pl.col("service_date").cast(pl.Date))


def _patterns_frame(net: _Net, used_pats, dates) -> pl.DataFrame:
    pats = []
    for name, dset in sorted(used_pats.items()):
        d, pk = int(name[1]), name.split("_")[1]
        a, b = net.pat_defs[(d, pk)]
        pats.append({"name": name, "direction_id": d, "kind": "short" if pk == "short" else "main",
                     "stop_ids": net.stop_id[d][a:b + 1], "offset_m": float(net.stop_s[d][a]),
                     "length_m": float(net.stop_s[d][b] - net.stop_s[d][a]),
                     "first_date": dates[min(dset)], "last_date": dates[max(dset)]})
    return pl.DataFrame(pats)


# ------------------------------------------------------------------ main
def simulate(seed: int = 0, days: int = 10, start: dt.date | str = "2025-03-03", mode: str = "stib",
             out_dir: str | Path | None = None, n_links: int = 8, tz: str = "Europe/Brussels",
             short_working: bool = True, change_day: int | None = None,
             outages=None, frozen=None, drop_rate: float = 0.02, poll_s: float = 20.0, poll_jitter_s: float = 2.0,
             noise_m: float = 5.0, feed_scale: float = 1.03, gtfsrt_interval: tuple[float, float] = (15.0, 60.0),
             terminus_alias: dict[int, str] | None = None, hotspots: bool = True) -> SimResult:
    """Simulate ``days`` service days from ``start``. See the module doc.

    ``change_day``: from this day index on, the last 2 stops of direction 0 (first 2 of direction 1)
    are cut (a second GTFS service). ``outages`` / ``frozen``: None = random (3 outages of 5-30 min,
    one 8-min freeze), [] = none, or [(day_idx, "HH:MM", minutes), ...].
    ``terminus_alias``: {direction_id: code} reported instead of the terminus stop id (STIB 9943 case)."""
    rng = np.random.default_rng(seed)
    if isinstance(start, str):
        start = dt.date.fromisoformat(start)
    dates = [start + dt.timedelta(days=i) for i in range(days)]
    out = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="ms_sim_"))
    out.mkdir(parents=True, exist_ok=True)
    # the draw order of rng is part of the result: geometry, windows, polls, trips, feed noise
    net = _net(n_links, rng, days, change_day, short_working, hotspots)
    used_pats = _write_gtfs(net, out, dates, tz)
    outage_w = _windows(outages, dates, tz, rng, 3, (5, 30), "outage")
    frozen_w = _windows(frozen, dates, tz, rng, 1, (8, 8), "frozen")
    poll_t = _polls(dates, tz, rng, poll_s, poll_jitter_s, drop_rate, outage_w)
    poll_frozen = _in_windows(poll_t, frozen_w)
    truth, pas, ev_rows, trip_rows, obs_parts = _run_vehicles(net, rng, dates, tz, mode, poll_t[~poll_frozen],
                                                              gtfsrt_interval, (outage_w, frozen_w))
    obs, raw = _observations(net, obs_parts, mode, rng, terminus_alias or {}, feed_scale, noise_m)
    trips_df = pl.DataFrame(trip_rows, orient="row", schema={
        "service_date": pl.Date, "trip_key": pl.Utf8, "trip_id": pl.Utf8, "vehicle_id": pl.Utf8,
        "direction_id": pl.Int8, "pattern": pl.Utf8, "dep": pl.Float64, "arr": pl.Float64})
    stops_df = pl.DataFrame([{"direction_id": d, "idx": i, "stop_id": net.stop_id[d][i], "stop_name": net.stop_name[d][i],
                              "s_m": int(net.stop_s[d][i])} for d in (0, 1) for i in range(net.n_stops)],
                            schema={"direction_id": pl.Int8, "idx": pl.Int16, "stop_id": pl.Utf8,
                                    "stop_name": pl.Utf8, "s_m": pl.Int32})
    return SimResult(
        mode=mode, gtfs_dir=out, dates=dates, tz=tz, observations=obs, snapshots=_snapshots(poll_t, poll_frozen, raw),
        raw=raw, stop_events=_stop_events(ev_rows),
        truth_seconds=_nonzero(truth, dates, "s_bin", np.int32, "seconds", np.float64)
        .sort("direction_id", "service_date", "hour", "s_bin"),
        passages=_nonzero(pas, dates, "link_idx", np.int16, "n", np.int64), trips=trips_df, stops=stops_df,
        patterns=_patterns_frame(net, used_pats, dates), hotspots=net.hs, outages=outage_w, frozen=frozen_w,
        line_length_m=net.line.L,
        params={"seed": seed, "days": days, "mode": mode, "change_day": change_day, "poll_s": poll_s,
                "noise_m": noise_m, "feed_scale": feed_scale, "gtfsrt_interval": gtfsrt_interval,
                "signal": {"cycle_s": SIGNAL_CYCLE, "red_s": SIGNAL_RED}, "short_working": short_working},
    )


__all__ = ["simulate", "SimResult"]
