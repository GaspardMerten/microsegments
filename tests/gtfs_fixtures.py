"""Tiny synthetic GTFS feeds written as .txt directories (network / segments tests)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

LAT0, LON0 = 50.85, 4.35
M_LON = 1 / (111195.0 * np.cos(np.radians(LAT0)))   # degrees per metre east
M_LAT = 1 / 111195.0                                 # degrees per metre north


def ll(x_m: float, y_m: float = 0.0) -> tuple[float, float]:
    return LON0 + x_m * M_LON, LAT0 + y_m * M_LAT


def densify(pts_m, step: float = 7.0) -> list[tuple[float, float]]:
    """Polyline in metres -> lon/lat vertices every ~step metres."""
    out = []
    pts = np.asarray(pts_m, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        for t in np.arange(n) / n:
            out.append(ll(*(a + t * (b - a))))
    out.append(ll(*pts[-1]))
    return out


# Stops of the synthetic line, x in metres along an east-west street (y = 0).
STOPS = {"A": 0.0, "B": 310.0, "C": 650.0, "D": 900.0, "E": 1230.0, "F": 1500.0}


def stop_rows():
    return [(s, f"Stop {s}", *ll(x, 4.0)) for s, x in STOPS.items()]   # stops 4 m off the street


def street(stops: list[str], reroute_cd: bool = False) -> list[tuple[float, float]]:
    """Shape through the given stops; reroute_cd: C->D goes round a block 120 m north."""
    pts = []
    for a, b in zip(stops[:-1], stops[1:]):
        xa, xb = STOPS[a], STOPS[b]
        pts.append((xa, 0.0))
        if reroute_cd and {a, b} == {"C", "D"}:
            pts += [(xa, 120.0), (xb, 120.0)]
    pts.append((STOPS[stops[-1]], 0.0))
    return densify(pts)


def write_feed(path: Path, patterns: list[dict], *, start: dt.date, end: dt.date,
               calendar: bool = True, extra_dates: list[tuple[dt.date, int]] = (),
               shapes: bool = True, route_id: str = "R55", short_name: str = "55") -> Path:
    """patterns: dicts with stops (list of stop ids), direction, n_trips, shape (lon/lat list) and
    optional service ("WK" default). One service WK running every day start..end."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "agency.txt").write_text("agency_id,agency_name,agency_url,agency_timezone\nA,Ag,http://x,Europe/Brussels\n")
    (path / "routes.txt").write_text(f"route_id,route_short_name,route_long_name,route_type\n{route_id},{short_name},Line,0\n"
                                     "R99,99,Other,3\n")
    (path / "stops.txt").write_text("stop_id,stop_name,stop_lat,stop_lon\n" + "".join(
        f"{s},{n},{lat:.7f},{lon:.7f}\n" for s, n, lon, lat in stop_rows()))
    services = sorted({p.get("service", "WK") for p in patterns})
    if calendar:
        (path / "calendar.txt").write_text(
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n" + "".join(
                f"{s},1,1,1,1,1,1,1,{start:%Y%m%d},{end:%Y%m%d}\n" for s in services))
    cd = list(extra_dates)
    if not calendar:   # calendar_dates only: every day listed
        d = start
        while d <= end:
            cd.append((d, 1))
            d += dt.timedelta(days=1)
    if cd:
        (path / "calendar_dates.txt").write_text("service_id,date,exception_type\n" + "".join(
            f"{s},{d:%Y%m%d},{e}\n" for d, e in cd for s in services))
    trips, st, sh = [], [], []
    for pi, p in enumerate(patterns):
        sid = f"sh{pi}"
        if shapes and p.get("shape") is not None:
            sh += [f"{sid},{lat:.7f},{lon:.7f},{k}\n" for k, (lon, lat) in enumerate(p["shape"])]
        for t in range(p["n_trips"]):
            tid = f"t{pi}_{t}"
            shape_col = sid if shapes and p.get("shape") is not None else ""
            trips.append(f"{route_id},{p.get('service', 'WK')},{tid},{p['direction']},{shape_col}\n")
            for k, s in enumerate(p["stops"]):
                hh = 6 + t // 6
                st.append(f"{tid},{hh:02d}:{(t % 6) * 10 + k:02d}:00,{hh:02d}:{(t % 6) * 10 + k:02d}:00,{s},{k + 1}\n")
    # one trip of another route, ignored
    trips.append("R99,WK,x1,0,\n")
    st += ["x1,07:00:00,07:00:00,A,1\n", "x1,07:05:00,07:05:00,F,2\n"]
    (path / "trips.txt").write_text("route_id,service_id,trip_id,direction_id,shape_id\n" + "".join(trips))
    (path / "stop_times.txt").write_text("trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" + "".join(st))
    if sh:
        (path / "shapes.txt").write_text("shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n" + "".join(sh))
    return path


FULL = list("ABCDEF")
CUT = list("ABCDE")


def v1_patterns():
    return [
        {"stops": FULL, "direction": 0, "n_trips": 20, "shape": street(FULL)},
        {"stops": FULL[::-1], "direction": 1, "n_trips": 20, "shape": street(FULL[::-1])},
        {"stops": list("ABCD"), "direction": 0, "n_trips": 4, "shape": street(list("ABCD"))},   # short working
        {"stops": list("ACDEF"), "direction": 0, "n_trips": 2, "shape": street(list("ACDEF"))},  # skips B: detour
    ]


def v2_patterns():
    """From the second version: line cut after E, and C->D rerouted round a block."""
    return [
        {"stops": CUT, "direction": 0, "n_trips": 18, "shape": street(CUT, reroute_cd=True)},
        {"stops": CUT[::-1], "direction": 1, "n_trips": 18, "shape": street(CUT[::-1], reroute_cd=True)},
    ]


def two_versions(tmp: Path, d1=dt.date(2025, 1, 1), d2=dt.date(2025, 1, 11), d3=dt.date(2025, 1, 20)):
    """Feed v1 valid d1..d2-1, v2 valid d2..d3. Returns {start date: path}."""
    p1 = write_feed(tmp / "v1", v1_patterns(), start=d1, end=d2 - dt.timedelta(days=1))
    p2 = write_feed(tmp / "v2", v2_patterns(), start=d2, end=d3)
    return {d1: p1, d2: p2}
