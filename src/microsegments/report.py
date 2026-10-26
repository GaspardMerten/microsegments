"""Analysis -> the JSON contract of ``contract.py`` (used by ``html.export`` and the platform's
``GET /api/analysis``).

Additive keys beyond ``contract.py`` (all optional for consumers):

* ``bands``: the page's bands, in button order, ``{"am": [7, 10], "midday": [10, 16], "pm": [16, 19],
  "evening": [20, 24]}`` ([start, end) hours; ``day_hours`` keeps the internal 6-21 h day);
* ``coverage.inc``: per date, ``true`` if the day is included, else the exclusion reason;
* ``dirs[].alt``: one entry per non-display GTFS version of the direction (same keys as a ``dirs``
  item: ``pattern_uid, from, to, stops, links, seg, wk``), so the page can show its own segments;
* ``dirs[].versions``: the ``pattern_uid`` list of the direction, display first;
* ``hotspots[].dir``: alias of ``direction_id``;
* ``hotspots_status``: "computed" or "not_computed" (e.g. too few days);
* ``source``, ``title``: free text for the page footer / heading (``None`` when unknown);
* ``dirs[].wk.n``: [dow][h][link] included day-hours, so ``p / n`` = passages (vehicles) per hour;
* ``dirs[].ref``: both references per segment over all included days, obs per passage:
  ``{"evening": [...], "line": [...], "line_level_per_m": m}`` (line = m x segment length, m = median
  over running segments of the 6-23 h obs / passage / metre);
* ``reference_agreement``: per direction, ``metrics.reference_agreement`` (Spearman of the two day-band
  excess profiles, top-segment overlap, hotspots per criterion and share found by both);
* ``compare``: present on a two-period page (``compare.Comparison.to_contract``), see ``contract.py``.

The display version of a direction (``dirs[]``, ``versions[].display``) is the longest one (route length,
then segment count, then the most recent). Page options a host may add: ``host_controls``, ``dir_control``,
``method``, ``show_hotspots``, ``tile_url`` / ``tile_attr`` (its own basemap; default OpenStreetMap,
darkened). Exclusion reasons the page names: the metrics ones plus ``holiday``, ``reduced_service``,
``school_holiday``, ``user_period``. postMessage to the page: ``{ms: "dir", dir}``, ``{ms: "ver", uid}``
(a ``pattern_uid`` of ``dirs[]`` or ``dirs[].alt``; also ``?ver=<uid>`` on load).
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import polars as pl

from .contract import CONTRACT_VERSION
from .hotspots import hotspot_status
from .metrics import BANDS, DAY_HOURS, PAGE_BANDS, Analysis, reference_agreement, version_groups
from .schema import Flag

ROUTE_TYPES = {0: "tram", 1: "metro", 2: "rail", 3: "bus", 4: "ferry", 5: "cable_tram", 6: "aerial_lift",
               7: "funicular", 11: "trolleybus", 12: "monorail"}


def _round_ll(c, nd: int = 5) -> list[list[float]]:
    # segment geometry: 5 decimals ~ 1 m, well under a 10 m segment
    return [[round(float(x), nd), round(float(y), nd)] for x, y in (c or [])]


def _jsonable(v, nd: int = 4):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, nd)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, nd) for x in v]
    if isinstance(v, np.generic):
        return _jsonable(v.item(), nd)
    return v


# decimals of record columns (others: 4, enough for obs per passage): metres and seconds to 0.1, shares and
# correlations to 0.001, frequencies to 0.01 per hour
NDIGITS = {"x0_m": 1, "x1_m": 1, "delta_s": 1, "veh_time_s_per_day": 1, "veh_time_s_per_day_all": 1,
           "persistence": 3, "freq_change": 3, "spearman_day": 3, "spearman_day_all": 3, "top_overlap": 3,
           "hotspots_both_share": 3, "passages_per_h_a": 2, "passages_per_h_b": 2}


def frame_records(df: pl.DataFrame) -> list[dict]:
    return [{k: _jsonable(v, NDIGITS.get(k, 4)) for k, v in r.items()} for r in df.iter_rows(named=True)]


def _links_of(links: pl.DataFrame | None, segments: pl.DataFrame, uid: str) -> pl.DataFrame:
    if links is not None and "pattern_uid" in links.columns:
        lk = links.filter(pl.col("pattern_uid") == uid).sort("link_idx")
        if lk.height:
            return lk
    # fall back on the segments: link start / length, stop names from the link key ("A-B" or key)
    s = segments.filter(pl.col("pattern_uid") == uid).sort("seg_idx")
    g = (s.group_by("link_idx", maintain_order=True)
         .agg(pl.col("link_key").first(), pl.col("x0_m").min().alias("s0_m"), pl.col("len_m").sum(),
              pl.col("geometry").first()))
    names = [str(i) for i in range(g.height + 1)]
    return g.with_columns(pl.lit(uid).alias("pattern_uid"),
                          pl.Series("from_stop", names[:-1]), pl.Series("to_stop", names[1:]),
                          pl.Series("from_name", [f"#{i}" for i in range(g.height)]),
                          pl.Series("to_name", [f"#{i + 1}" for i in range(g.height)]))


def _stop_ll(stops: pl.DataFrame | None, links: pl.DataFrame) -> list[list[float] | None]:
    """lon/lat of every stop of the pattern (from the stops table, else the link geometries)."""
    ids = links["from_stop"].to_list() + links["to_stop"].to_list()[-1:]
    pos = {}
    if stops is not None and stops.height:
        pos = {r[0]: [r[1], r[2]] for r in stops.select("stop_id", "stop_lon", "stop_lat").iter_rows()}
    geo = links["geometry"].to_list() if "geometry" in links.columns else [None] * links.height
    out = []
    for i, sid in enumerate(ids):
        if sid in pos and pos[sid][0] is not None:
            out.append([round(pos[sid][0], 6), round(pos[sid][1], 6)])
            continue
        g = geo[i] if i < len(geo) else (geo[-1] if geo else None)
        if g:
            p = g[0] if i < len(geo) else g[-1]
            out.append([round(p[0], 6), round(p[1], 6)])
        else:
            out.append(None)
    return out


def _day_flags(n: int, D: int, min_days: int) -> int:
    """FEW_DAYS / PARTIAL_VERSIONS of a segment or link valid on n of the D included days."""
    return (Flag.FEW_DAYS if n < min_days else 0) | (Flag.PARTIAL_VERSIONS if n < D else 0)


def _dir_entry(an: Analysis, dirid: int, uid: str, links: pl.DataFrame | None, stops: pl.DataFrame | None,
               names: dict[str, str]) -> dict[str, Any]:
    dd = an.dirs[dirid]
    fix = lambda n: names.get(n, n) if n is not None else n  # noqa: E731
    lk = _links_of(links, an.segments, uid)
    wk = an.to_contract_wk(dirid, pattern_uid=uid)
    stops_e = _stops_entry(lk, stops, fix)
    lpos = {k: i for i, k in enumerate(dd.link_keys.tolist())}
    nd_link, D = dd.LV.sum(0), len(dd.days)
    return {
        "dir": int(dirid),
        "pattern_uid": uid,
        "from": stops_e[0]["name"] if stops_e else None,
        "to": stops_e[-1]["name"] if stops_e else None,
        "stops": stops_e,
        "links": [{"key": r["link_key"], "from": fix(r["from_name"]), "to": fix(r["to_name"]),
                   "len": round(float(r["len_m"]), 1),
                   "n_days": int(nd_link[lpos[r["link_key"]]]) if r["link_key"] in lpos else 0,
                   "flags": _day_flags(int(nd_link[lpos[r["link_key"]]]), D, an.quality.min_days) if r["link_key"] in lpos else 0}
                  for r in lk.iter_rows(named=True)],
        "seg": _seg_entry(an, dd, uid, wk["seg_key"], lk),
        "wk": {k: wk[k] for k in ("obs", "cov", "p", "days", "n")},
        "ref": _refs(an, dirid, wk["seg_key"]),
    }


def _stops_entry(lk: pl.DataFrame, stops: pl.DataFrame | None, fix) -> list[dict]:
    """The pattern's stops: id, shown name, lon/lat, position along the pattern (m)."""
    if not lk.height:
        return []
    s0 = lk["s0_m"].to_list()
    x_stops = s0 + [s0[-1] + lk["len_m"].to_list()[-1]]
    ids = lk["from_stop"].to_list() + lk["to_stop"].to_list()[-1:]
    nms = lk["from_name"].to_list() + lk["to_name"].to_list()[-1:]
    return [{"id": i, "name": fix(n), "ll": ll, "x": round(float(x), 1)}
            for i, n, ll, x in zip(ids, nms, _stop_ll(stops, lk), x_stops)]


def _seg_entry(an: Analysis, dd, uid: str, keys: list[str], lk: pl.DataFrame) -> dict[str, list]:
    """Per segment of the pattern, in the contract's seg_key order (= seg_idx order): link index, position,
    length, zone, geometry, flags."""
    seg = an.segments.filter(pl.col("pattern_uid") == uid).sort("seg_idx")
    col = lambda c: dict(zip(seg["seg_key"].to_list(), seg[c].to_list()))  # noqa: E731
    x0, ln, zone, slink, geo = col("x0_m"), col("len_m"), col("zone"), col("link_key"), col("geometry")
    link_order = {k: i for i, k in enumerate(lk["link_key"].to_list())}
    kpos = {k: i for i, k in enumerate(dd.seg_keys.tolist())}
    nd_seg, D = dd.V.sum(0), len(dd.days)
    return {
        "key": keys,
        "link": [link_order.get(slink[k], -1) for k in keys],
        "x0": [round(float(x0[k]), 1) for k in keys],
        "len": [round(float(ln[k]), 1) for k in keys],
        "zone": [zone[k] for k in keys],
        "c": [_round_ll(geo.get(k)) for k in keys],
        "flags": [_day_flags(int(nd_seg[kpos[k]]), D, an.quality.min_days) for k in keys],
    }


def _refs(an: Analysis, dirid: int, keys: list[str]) -> dict[str, Any]:
    """Both references per segment (all included days, obs per passage): evening and line level."""
    dd = an.dirs[dirid]
    est = an.estimate(dirid)
    kpos = {k: i for i, k in enumerate(dd.seg_keys.tolist())}
    idx = [kpos[k] for k in keys]
    r4 = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]  # noqa: E731
    return {"evening": r4(est["ref"][0][idx]), "line": r4(est["ref_line"][0][idx]),
            "line_level_per_m": _jsonable(float(est["line_level"][0]))}


NAME_PARTICLES = frozenset({"du", "de", "la", "des", "le", "van", "aux"})


def title_name(name: str | None) -> str | None:
    """Title-case an all-caps stop name, particles in lower case except first: "GARE DU NORD" ->
    "Gare du Nord", "SAINTE-MARIE" -> "Sainte-Marie", "LEOPOLD III" -> "Leopold III". Names that
    already mix cases are returned unchanged."""
    if not name or name != name.upper():
        return name
    words = name.split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i and lw in NAME_PARTICLES:
            out.append(lw)
        elif w and set(w) <= set("IVX") and len(w) <= 4 and i:
            out.append(w)
        else:
            out.append("-".join("'".join(p[:1].upper() + p[1:] for p in part.split("'")) for part in lw.split("-")))
    return " ".join(out)


def name_map(names, net=None) -> dict[str, str]:
    """{gtfs name: shown name} from ``names``: a dict (as is), "title" (``title_name`` on every stop
    name of ``net``), or None."""
    if not names:
        return {}
    if isinstance(names, dict):
        return dict(names)
    if names == "title":
        pool: set[str] = set()
        if net is not None:
            pool |= {n for n in net.stops["stop_name"].to_list() if n}
            pool |= {n for c in ("from_name", "to_name") for n in net.links[c].to_list() if n}
        return {n: title_name(n) for n in pool if title_name(n) != n}
    raise ValueError(f"unknown names option {names!r} (a dict or 'title')")


def to_contract(analysis: Analysis, net=None, segments: pl.DataFrame | None = None,
                hotspots: pl.DataFrame | None = None, *, line: str | None = None, route_id: str | None = None,
                mode: str | None = None, names: dict[str, str] | str | None = None, ctx: list | None = None,
                coverage: pl.DataFrame | None = None, title: str | None = None,
                source: str | None = None, route_color: str | None = None,
                route_text_color: str | None = None) -> dict[str, Any]:
    """Build the JSON contract (``contract.py``) of an analysis.

    ``net``: the :class:`~microsegments.network.Network` (stop names and positions; without it stops are
    numbered). ``segments``: defaults to ``analysis.segments`` (must carry the geometry for the map).
    ``hotspots``: ``hotspots.hotspots(analysis)`` output. ``names``: stop name fixes {gtfs name: shown name},
    or "title" to title-case all-caps names (:func:`title_name`).
    ``coverage``: COVERAGE table for the coverage strip (else rebuilt from ``analysis.day_hours``).
    ``route_color`` / ``route_text_color``: GTFS colours of the line ("RRGGBB", "#" optional) for the page's
    line badge; invalid values are dropped (:func:`route_colors` reads them from a feed)."""
    an = analysis
    if segments is not None:
        an = _with_segments(an, segments)
    names = name_map(names, net)
    links = net.links if net is not None else None
    stops = net.stops if net is not None else None
    if net is not None and line is None:
        line = str(net.route)
    if net is not None and route_id is None and net.patterns.height:
        route_id = net.patterns["route_id"][0]

    dates, c, inc, exc = _coverage(an, coverage)
    p = an.params
    period = {
        "first": (min(an.days) if an.days else (dates[0] if dates else None)),
        "last": (max(an.days) if an.days else (dates[-1] if dates else None)),
        "days": len(an.days),
        "per_dow": list(an.per_dow),
        "excluded": [{"date": d.isoformat(), "reason": r} for d, r in sorted(exc.items())],
    }
    hs_ok = hotspots is not None and hotspots.height
    agree = frame_records(reference_agreement(an, hotspots if hs_ok else None))
    return {
        "v": CONTRACT_VERSION,
        "line": line,
        "route_id": route_id,
        "mode": mode,
        "route_color": _hex(route_color),
        "route_text_color": _hex(route_text_color),
        "title": title,
        "source": source,
        "params": {"segment_m": p.segment_m, "phase_m": p.phase_m, "grid": p.grid, "tick_s": p.tick_s,
                   "gap_cap_s": p.gap_cap_s if math.isfinite(p.gap_cap_s) else None,
                   "stop_zone": list(p.stop_zone), "reference_hours": list(p.reference_hours),
                   "min_passages_per_day": an.quality.min_passages_per_day},
        "period": {k: _jsonable(v) for k, v in period.items()},
        "hours": list(an.hours),
        "bands": {k: list(BANDS[k]) for k in PAGE_BANDS},
        "day_hours": list(DAY_HOURS),
        "coverage": {"dates": [d.isoformat() for d in dates], "c": c, "inc": inc},
        "versions": [{"dir": int(r["direction_id"]), "pattern_uid": r["pattern_uid"], "first": _jsonable(r["first"]),
                      "last": _jsonable(r["last"]), "n_days": int(r["n_days"]), "links_added": list(r["links_added"] or []),
                      "links_removed": list(r["links_removed"] or []), "display": bool(r["display"])}
                     for r in an.versions.iter_rows(named=True)],
        "dirs": _dirs(an, links, stops, names),
        "hotspots": _hotspot_rows(hotspots, names) if hs_ok else [],
        "hotspots_status": "computed" if hotspots is not None and hotspot_status(an)[0] else "not_computed",
        "reference_agreement": agree,
        "ctx": ctx or [],
    }


def _dirs(an: Analysis, links, stops, names) -> list[dict]:
    """One entry per direction with included days (its display version), the other versions under ``alt``."""
    dirs = []
    for dirid in sorted(an.dirs):
        dd = an.dirs[dirid]
        if len(dd.days) == 0:
            continue
        e = _dir_entry(an, dirid, dd.display_uid, links, stops, names)
        others = [u for u in version_groups(dd) if u != dd.display_uid]   # one per distinct link sequence
        e["versions"] = [dd.display_uid] + others
        e["alt"] = [{k: v for k, v in _dir_entry(an, dirid, u, links, stops, names).items() if k != "dir"}
                    for u in others]
        dirs.append(e)
    return dirs


def _coverage(an: Analysis, coverage: pl.DataFrame | None):
    """Coverage per date x output hour over every selected date (included or not): dates, c, inc (True or the
    exclusion reason), and {date: reason} of the excluded days."""
    exc = {r[0]: r[1] for r in an.excluded_days.iter_rows()}
    dh = an.day_hours if coverage is None else coverage.select("service_date", "hour", "coverage")
    dates = sorted(set(dh["service_date"].to_list()) | set(exc) | set(an.days))
    rng = an.select.date_range()
    if rng is not None:
        dates = [d for d in dates if rng[0] <= d <= rng[1]]
    cmap = {(r[0], int(r[1])): r[2] for r in dh.select("service_date", "hour", "coverage").iter_rows()}
    c = [[None if cmap.get((d, h)) is None else round(float(cmap[(d, h)]), 3) for h in an.hours] for d in dates]
    inc_set = set(an.days)
    return dates, c, [True if d in inc_set else exc.get(d, "not_selected") for d in dates], exc


def _hotspot_rows(hotspots: pl.DataFrame, names: dict[str, str]) -> list[dict]:
    """Hotspot records for the page (no seg_keys; ``dir`` alias; stop names fixed)."""
    out = []
    for r in frame_records(hotspots.select([c for c in hotspots.columns if c != "seg_keys"])):
        r["dir"] = r.get("direction_id")
        for k in ("from_stop_name", "to_stop_name"):
            if r.get(k) in names:
                r[k] = names[r[k]]
        out.append(r)
    return out


def _with_segments(an: Analysis, segments: pl.DataFrame) -> Analysis:
    from dataclasses import replace
    return replace(an, segments=segments)


def route_mode(source, route: str, date) -> str | None:
    """'tram' / 'bus' / ... from the GTFS route_type of ``route`` on ``date`` (None if unknown)."""
    try:
        feed = source.feed_for(date, route)
        rt = feed.routes["route_type"].drop_nulls()
        return ROUTE_TYPES.get(int(rt[0])) if len(rt) else None
    except Exception:  # noqa: BLE001 - optional nicety
        return None


def _hex(c) -> str | None:
    """A GTFS colour as "RRGGBB" (upper case), None when missing or malformed."""
    c = str(c or "").strip().lstrip("#")
    return c.upper() if len(c) == 6 and all(x in "0123456789abcdefABCDEF" for x in c) else None


def route_colors(source, route: str, date) -> tuple[str | None, str | None]:
    """GTFS ``route_color`` and ``route_text_color`` of ``route`` on ``date`` ("RRGGBB"; None if unknown)."""
    try:
        r = source.feed_for(date, route).routes
        return tuple(_hex(r[c].drop_nulls()[0]) if c in r.columns and r[c].drop_nulls().len() else None
                     for c in ("route_color", "route_text_color"))
    except Exception:  # noqa: BLE001 - optional nicety
        return None, None


__all__ = ["ROUTE_TYPES", "frame_records", "route_colors", "route_mode", "to_contract"]
