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
    terminus_alias = terminus_alias or {}

    lon, lat, cum, sp = _curve(n_links, rng)
    line = _Line(lon, lat, cum, sp)
    L = line.L
    n_stops = n_links + 1
    # per direction: full-line stop positions and ids (physical stop j is index j in dir 0, n-1-j in dir 1)
    stop_s = {0: sp.copy(), 1: (L - sp[::-1]).astype(int)}
    phys_of = {0: list(range(n_stops)), 1: list(range(n_stops))[::-1]}
    stop_id = {d: [f"{1000 * (d + 1) + j + 1}" for j in phys_of[d]] for d in (0, 1)}
    stop_name = {d: [STOP_NAMES[j % len(STOP_NAMES)] for j in phys_of[d]] for d in (0, 1)}

    # hotspots (full-line coordinates)
    hs: list[dict] = []
    sig = {}
    queue = {}
    hold = {}
    if hotspots:
        k = min(2, n_links - 1)
        s_sig0 = int(sp[k] + round(0.6 * (sp[k + 1] - sp[k])))
        sig = {0: s_sig0, 1: L - s_sig0}
        for d in (0, 1):
            li = int(np.searchsorted(stop_s[d], sig[d], side="right") - 1)
            hs.append({"kind": "signal", "direction_id": d, "s0_m": float(sig[d]), "s1_m": float(sig[d]),
                       "link_idx": li, "hours": list(range(5, 28)), "extra_s": SIGNAL_RED / 2,
                       "class": "infrastructure", "zone": "running"})
        kq = min(5, n_links - 1)
        q1 = int(sp[kq] + round(0.7 * (sp[kq + 1] - sp[kq])))
        queue = {0: (q1 - QUEUE_LEN, q1)}
        hs.append({"kind": "queue", "direction_id": 0, "s0_m": float(q1 - QUEUE_LEN), "s1_m": float(q1),
                   "link_idx": kq, "hours": list(QUEUE_HOURS), "extra_s": QUEUE_EXTRA,
                   "class": "congestion", "zone": "running"})
        kh = min(4, n_links - 1)
        hold = {1: kh}
        hs.append({"kind": "holding", "direction_id": 1, "s0_m": float(stop_s[1][kh]), "s1_m": float(stop_s[1][kh]),
                   "link_idx": kh - 1, "stop_idx": kh, "stop_id": stop_id[1][kh], "hours": list(HOLD_HOURS),
                   "extra_s": HOLD_EXTRA, "class": "holding", "zone": "stop"})

    # patterns: (first stop idx, last stop idx) in direction order
    cut = 2
    short_end = min(5, n_links - 1)
    pat_defs = {
        (0, "main"): (0, n_stops - 1), (1, "main"): (0, n_stops - 1),
        (0, "short"): (0, short_end), (1, "short"): (n_stops - 1 - short_end, n_stops - 1),
        (0, "cut"): (0, n_stops - 1 - cut), (1, "cut"): (cut, n_stops - 1),
    }
    services = [("BASE", 0, days - 1)]
    if change_day is not None and 0 < change_day < days:
        services = [("BASE", 0, change_day - 1), ("CUT", change_day, days - 1)]

    # nominal schedule (mean dwell 20 s)
    def nominal(d, a, b):
        t, out_t = 0.0, [(0.0, 0.0)]
        for i in range(a, b):
            t += float(_rest(int(stop_s[d][i + 1] - stop_s[d][i]))[-1])
            arr = t
            if i + 1 < b:
                t += 20.0
            out_t.append((arr, t))
        return out_t

    # departures per service day
    deps = []   # (dep_local_s, direction, kind)
    for d in (0, 1):
        t = 5 * 3600 + (180 if d else 0)
        n = 0
        while t < 24 * 3600:
            kind = "short" if (short_working and 6 <= t / 3600 < 20 and n % 4 == 3) else "main"
            deps.append((float(t), d, kind))
            n += 1
            t += HEADWAY_MIN[int(t // 3600)] * 60
    deps.sort()

    # ---------------- GTFS
    trips_rows, st_rows, shape_rows, used_pats = [], [], [], {}
    for sid, d0, d1 in services:
        for j, (tdep, d, kind) in enumerate(deps):
            pk = "cut" if (sid == "CUT" and kind == "main") else kind
            a, b = pat_defs[(d, pk)]
            name = f"d{d}_{pk}"
            used_pats.setdefault(name, set()).update(range(d0, d1 + 1))
            trip_id = f"{sid}_{d}_{j:04d}"
            trips_rows.append({"route_id": ROUTE, "service_id": sid, "trip_id": trip_id, "direction_id": d,
                               "shape_id": name, "trip_headsign": stop_name[d][b]})
            for q, (i, (arr, dep)) in enumerate(zip(range(a, b + 1), nominal(d, a, b))):
                st_rows.append({"trip_id": trip_id, "arrival_time": _hhmmss(tdep + arr),
                                "departure_time": _hhmmss(tdep + dep), "stop_id": stop_id[d][i],
                                "stop_sequence": q + 1,
                                "shape_dist_traveled": float(stop_s[d][i] - stop_s[d][a])})
    for name in used_pats:
        d, pk = int(name[1]), name.split("_")[1]
        a, b = pat_defs[(d, pk)]
        pts = line.polyline(d, stop_s[d][a], stop_s[d][b])
        cd = np.concatenate([[0.0], np.cumsum(_hav(*np.array(pts[:-1]).T, *np.array(pts[1:]).T))])
        for q, ((x, y), c) in enumerate(zip(pts, cd)):
            shape_rows.append({"shape_id": name, "shape_pt_lat": y, "shape_pt_lon": x,
                               "shape_pt_sequence": q + 1, "shape_dist_traveled": float(c)})
    stops_rows = []
    for d in (0, 1):
        for i in range(n_stops):
            x, y = line.lonlat(d, stop_s[d][i])
            stops_rows.append({"stop_id": stop_id[d][i], "stop_name": stop_name[d][i],
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
                   "end_date": dates[b].strftime("%Y%m%d")} for sid, a, b in services]).write_csv(out / "calendar.txt")

    # ---------------- polls
    outage_w = _windows(outages, dates, tz, rng, 3, (5, 30), "outage")
    frozen_w = _windows(frozen, dates, tz, rng, 1, (8, 8), "frozen")
    polls = []
    for day in dates:
        a = local_epoch(day, 4 * 3600, tz)
        b = local_epoch(day + dt.timedelta(days=1), 4 * 3600, tz)
        n = int((b - a) / poll_s) + 10
        t = a + np.round(np.arange(n) * poll_s + rng.uniform(-poll_jitter_s, poll_jitter_s, n), 0)
        t = t[(t >= a) & (t < b)]
        polls.append(t)
    poll_t = np.unique(np.concatenate(polls))
    keep = (rng.random(poll_t.size) >= drop_rate) & ~_in_windows(poll_t, outage_w)
    poll_t = poll_t[keep]
    poll_frozen = _in_windows(poll_t, frozen_w)
    live_t = poll_t[~poll_frozen]

    # ---------------- vehicles
    n_bins = L + 1
    hours_n = 28
    truth = np.zeros((2, days, hours_n, n_bins))
    pas = np.zeros((2, days, hours_n, n_links), dtype=np.int64)
    ev_rows, trip_rows = [], []
    obs_parts: list[dict[str, np.ndarray]] = []
    veh_count = 0
    shape_k, scale_k = 4.0, 5.0
    for di, day in enumerate(dates):
        sid = next(s for s, a, b in services if a <= di <= b)
        day0 = local_epoch(day, 4 * 3600, tz)
        pools: dict[int, list[list]] = {}   # physical stop -> [[available_t, vid]]
        for j, (tdep, d, kind) in enumerate(deps):
            pk = "cut" if (sid == "CUT" and kind == "main") else kind
            a_i, b_i = pat_defs[(d, pk)]
            trip_id = f"{sid}_{d}_{j:04d}"
            tkey = f"{day:%Y%m%d}:{trip_id}"
            dep = local_epoch(day, tdep, tz) + float(rng.uniform(0, 45))
            # --- trajectory knots (s, t)
            S_parts, T_parts = [], []
            t = dep
            ev = [(stop_s[d][a_i], None, dep)]
            for i in range(a_i, b_i):
                sA, sB = int(stop_s[d][i]), int(stop_s[d][i + 1])
                cur = sA
                obstacles = []
                if d in sig and sA < sig[d] < sB:
                    obstacles.append(("signal", sig[d]))
                if d in queue and sA < queue[d][0] and queue[d][1] < sB:
                    obstacles.append(("queue", queue[d][0]))
                obstacles.sort(key=lambda o: o[1])
                for kind_o, pos in obstacles:
                    prof = _rest(sB - cur)
                    t_through = t + float(prof[pos - cur])
                    if kind_o == "signal":
                        if (t_through % SIGNAL_CYCLE) < SIGNAL_RED:
                            leg = _rest(pos - cur)
                            S_parts.append(np.arange(cur, pos + 1))
                            T_parts.append(t + leg)
                            t_arr = t + float(leg[-1])
                            green = math.floor(t_arr / SIGNAL_CYCLE) * SIGNAL_CYCLE + SIGNAL_RED
                            t = max(t_arr, green if (t_arr % SIGNAL_CYCLE) < SIGNAL_RED else t_arr)
                            S_parts.append(np.array([pos]))
                            T_parts.append(np.array([t]))
                            cur = pos
                    else:
                        h = int((t_through - day0) // 3600) + 4
                        if h in QUEUE_HOURS:
                            q0, q1 = queue[d]
                            leg = _rest(q0 - cur)
                            S_parts.append(np.arange(cur, q0 + 1))
                            T_parts.append(t + leg)
                            t = t + float(leg[-1])
                            crawl = QUEUE_LEN / VMAX + QUEUE_EXTRA
                            S_parts.append(np.arange(q0, q1 + 1))
                            T_parts.append(t + np.arange(QUEUE_LEN + 1) * (crawl / QUEUE_LEN))
                            t += crawl
                            cur = q1
                leg = _rest(sB - cur)
                S_parts.append(np.arange(cur, sB + 1))
                T_parts.append(t + leg)
                t = t + float(leg[-1])
                arr = t
                if i + 1 < b_i:
                    dwell = float(rng.gamma(shape_k, scale_k))
                    h = int((t - day0) // 3600) + 4
                    if d in hold and i + 1 == hold[d] and h in HOLD_HOURS:
                        dwell += HOLD_EXTRA
                    t += dwell
                    S_parts.append(np.array([sB]))
                    T_parts.append(np.array([t]))
                ev.append((sB, arr, t if i + 1 < b_i else None))
            S = np.concatenate(S_parts).astype(np.int64)
            T = np.concatenate(T_parts)
            dup = np.concatenate([[False], (np.diff(S) == 0) & (np.diff(T) == 0)])
            S, T = S[~dup], T[~dup]
            arr_end = float(T[-1])
            # --- truth
            ds, dtt = np.diff(S), np.diff(T)
            bins = np.where(ds > 0, S[:-1], S[:-1] - 1)
            mid = (T[:-1] + T[1:]) / 2
            hrs = np.clip(((mid - day0) // 3600).astype(int) + 4, 0, hours_n - 1)
            ok = bins >= 0
            np.add.at(truth[d, di], (hrs[ok], bins[ok]), dtt[ok])
            for q, i in enumerate(range(a_i, b_i)):
                td = ev[q][2]
                pas[d, di, min(int((td - day0) // 3600) + 4, hours_n - 1), i] += 1
            for q, (s_, arr, dpt) in enumerate(ev):
                ev_rows.append((day, tkey, stop_id[d][a_i + q], a_i + q + 1,
                                None if arr is None else int(round(arr * 1e6)),
                                None if dpt is None else int(round(dpt * 1e6))))
            # --- vehicle assignment and layover
            p_start, p_end = phys_of[d][a_i], phys_of[d][b_i]
            lay = float(rng.uniform(60, 240))
            pool = pools.setdefault(p_start, [])
            pool.sort()
            if pool and pool[0][0] <= dep - 30:
                av, vid = pool.pop(0)
                t_lay0 = max(av, dep - lay)
            else:
                veh_count += 1
                vid = f"V{veh_count:03d}"
                t_lay0 = dep - lay
            pools.setdefault(p_end, []).append([arr_end + 60.0, vid])
            trip_rows.append((day, tkey, trip_id, vid, d, f"d{d}_{pk}", dep, arr_end))
            # --- observation times
            if mode == "stib":
                lo, hi = np.searchsorted(live_t, [t_lay0, arr_end])
                ts = live_t[lo:hi]
            else:
                ilo, ihi = gtfsrt_interval
                n = int((arr_end - t_lay0) / ilo) + 2
                ts = t_lay0 + rng.uniform(0, ihi) + np.concatenate([[0], np.cumsum(rng.uniform(ilo, ihi, n))])
                ts = np.round(ts[ts < arr_end], 0)
                ts = ts[~_in_windows(ts, outage_w) & ~_in_windows(ts, frozen_w)]
            if ts.size == 0:
                continue
            pos = np.where(ts < dep, float(stop_s[d][a_i]), np.interp(ts, T, S.astype(float)))
            obs_parts.append({"ts": ts, "s": pos, "d": np.full(ts.size, d), "a": np.full(ts.size, a_i),
                              "b": np.full(ts.size, b_i), "trip": np.full(ts.size, tkey, dtype=object),
                              "trip_id": np.full(ts.size, trip_id, dtype=object),
                              "vid": np.full(ts.size, vid, dtype=object)})

    # ---------------- observations
    cat = {k: np.concatenate([p[k] for p in obs_parts]) for k in obs_parts[0]} if obs_parts else None
    raw = None
    if cat is None:
        obs = schema.empty(schema.OBSERVATION)
    elif mode == "stib":
        dd, s = cat["d"], cat["s"]
        idx = np.empty(s.size, dtype=np.int64)
        base = np.empty(s.size)
        term = np.empty(s.size, dtype=object)
        point = np.empty(s.size, dtype=object)
        for d in (0, 1):
            m = dd == d
            k = np.searchsorted(stop_s[d], s[m] + 1e-9, side="right") - 1
            k = np.minimum(np.maximum(k, cat["a"][m]), cat["b"][m])
            idx[m] = k
            base[m] = stop_s[d][k]
            ids = np.array(["0" + x for x in stop_id[d]], dtype=object)
            point[m] = ids[k]
            tb = ids[cat["b"][m]]
            if d in terminus_alias:
                tb = np.full(tb.size, str(terminus_alias[d]), dtype=object)
            term[m] = tb
        since = s - base
        moving = since > 1e-6
        dist = np.where(moving, np.maximum(since * feed_scale + rng.normal(0, noise_m, s.size), 0), 0.0)
        raw = pl.DataFrame({"ts": cat["ts"].astype(np.int64), "line": np.full(s.size, ROUTE, dtype=object),
                            "dir": term, "point": point, "dist": np.round(dist).astype(np.int64)},
                           schema={"ts": pl.Int64, "line": pl.Utf8, "dir": pl.Utf8, "point": pl.Utf8, "dist": pl.Int64})
        order = np.argsort(cat["ts"], kind="stable")
        raw = raw[order]
        from .io.stib import compact_to_observations
        obs = compact_to_observations(raw.lazy()).collect()
        cat = {k: v[order] for k, v in cat.items()}
    else:
        lo_, la_ = np.empty(cat["s"].size), np.empty(cat["s"].size)
        br = np.empty(cat["s"].size)
        for d in (0, 1):
            m = cat["d"] == d
            lo_[m], la_[m] = line.lonlat(d, cat["s"][m])
            br[m] = line.bearing(d, cat["s"][m])
        la_ = la_ + np.degrees(rng.normal(0, noise_m, la_.size) / R_EARTH)
        lo_ = lo_ + np.degrees(rng.normal(0, noise_m, lo_.size) / (R_EARTH * math.cos(math.radians(50.85))))
        order = np.lexsort((cat["vid"].astype(str), cat["ts"]))
        cat = {k: v[order] for k, v in cat.items()}
        obs = pl.DataFrame({
            "ts": (cat["ts"] * 1e6).astype(np.int64), "route_id": np.full(order.size, ROUTE, dtype=object),
            "vehicle_id": cat["vid"], "trip_id": cat["trip_id"], "direction_id": cat["d"].astype(np.int8),
            "lat": la_[order], "lon": lo_[order], "bearing": br[order].astype(np.float32),
        }).with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
        obs = schema.conform(obs, schema.OBSERVATION, schema.OBSERVATION_REQUIRED)
    if cat is not None:
        obs = obs.with_columns(
            pl.Series("true_direction_id", cat["d"].astype(np.int8)),
            pl.Series("true_s_m", cat["s"].astype(np.float32)),
            pl.Series("true_trip_key", cat["trip"].astype(str)),
            pl.Series("true_vehicle_id", cat["vid"].astype(str)),
        )

    # ---------------- tables
    n_rows = np.full(poll_t.size, OTHER_LINES_ROWS, dtype=np.int64)
    if mode == "stib" and raw is not None:
        u, c = np.unique(raw["ts"].to_numpy(), return_counts=True)
        pos_ = np.searchsorted(poll_t, u)
        n_rows[pos_] += c
    snaps = pl.DataFrame({"ts": (poll_t * 1e6).astype(np.int64), "n_rows": n_rows, "frozen": poll_frozen}).with_columns(
        pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")).cast(schema.SNAPSHOT)

    ev = pl.DataFrame(ev_rows, schema={"service_date": pl.Date, "trip_key": pl.Utf8, "stop_id": pl.Utf8,
                                       "stop_sequence": pl.Int32, "arrival": pl.Int64, "departure": pl.Int64},
                      orient="row").with_columns(
        *[pl.col(c).cast(pl.Datetime("us")).dt.replace_time_zone("UTC") for c in ("arrival", "departure")],
        pl.lit(ROUTE).alias("route_id"), pl.lit(True).alias("observed"))
    ev = ev.select([pl.col(k).cast(v) for k, v in schema.STOP_EVENT.items()])

    nz = np.nonzero(truth)
    truth_df = pl.DataFrame({
        "direction_id": nz[0].astype(np.int8), "s_bin": nz[3].astype(np.int32),
        "service_date": np.array(dates, dtype="datetime64[D]")[nz[1]],
        "hour": nz[2].astype(np.int8), "seconds": truth[nz].astype(np.float64)}
    ).with_columns(pl.col("service_date").cast(pl.Date)).sort("direction_id", "service_date", "hour", "s_bin")
    pz = np.nonzero(pas)
    pas_df = pl.DataFrame({"direction_id": pz[0].astype(np.int8), "link_idx": pz[3].astype(np.int16),
                           "service_date": np.array(dates, dtype="datetime64[D]")[pz[1]],
                           "hour": pz[2].astype(np.int8), "n": pas[pz].astype(np.int64)}
                          ).with_columns(pl.col("service_date").cast(pl.Date))
    trips_df = pl.DataFrame(trip_rows, orient="row", schema={
        "service_date": pl.Date, "trip_key": pl.Utf8, "trip_id": pl.Utf8, "vehicle_id": pl.Utf8,
        "direction_id": pl.Int8, "pattern": pl.Utf8, "dep": pl.Float64, "arr": pl.Float64})
    stops_df = pl.DataFrame([{"direction_id": d, "idx": i, "stop_id": stop_id[d][i], "stop_name": stop_name[d][i],
                              "s_m": int(stop_s[d][i])} for d in (0, 1) for i in range(n_stops)],
                            schema={"direction_id": pl.Int8, "idx": pl.Int16, "stop_id": pl.Utf8,
                                    "stop_name": pl.Utf8, "s_m": pl.Int32})
    pats = []
    for name, dset in sorted(used_pats.items()):
        d, pk = int(name[1]), name.split("_")[1]
        a, b = pat_defs[(d, pk)]
        pats.append({"name": name, "direction_id": d, "kind": "short" if pk == "short" else "main",
                     "stop_ids": stop_id[d][a:b + 1], "offset_m": float(stop_s[d][a]),
                     "length_m": float(stop_s[d][b] - stop_s[d][a]),
                     "first_date": dates[min(dset)], "last_date": dates[max(dset)]})
    return SimResult(
        mode=mode, gtfs_dir=out, dates=dates, tz=tz, observations=obs, snapshots=snaps, raw=raw,
        stop_events=ev, truth_seconds=truth_df, passages=pas_df, trips=trips_df, stops=stops_df,
        patterns=pl.DataFrame(pats), hotspots=hs, outages=outage_w, frozen=frozen_w, line_length_m=L,
        params={"seed": seed, "days": days, "mode": mode, "change_day": change_day, "poll_s": poll_s,
                "noise_m": noise_m, "feed_scale": feed_scale, "gtfsrt_interval": gtfsrt_interval,
                "signal": {"cycle_s": SIGNAL_CYCLE, "red_s": SIGNAL_RED}, "short_working": short_working},
    )


__all__ = ["simulate", "SimResult"]
