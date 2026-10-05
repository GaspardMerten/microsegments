"""Analysis -> the JSON contract of ``contract.py`` (used by ``html.export`` and the platform's
``GET /api/analysis``).

Additive keys beyond ``contract.py`` (all optional for consumers):

* ``bands``: ``{"am": [7, 10], "pm": [16, 19], "day": [6, 21], "evening": [20, 23]}`` ([start, end) hours);
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
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import polars as pl

from .contract import CONTRACT_VERSION
from .metrics import BANDS, DAY_HOURS, Analysis
from .schema import Flag

ROUTE_TYPES = {0: "tram", 1: "metro", 2: "rail", 3: "bus", 4: "ferry", 5: "cable_tram", 6: "aerial_lift",
               7: "funicular", 11: "trolleybus", 12: "monorail"}


def _round_ll(c, nd: int = 6) -> list[list[float]]:
    return [[round(float(x), nd), round(float(y), nd)] for x, y in (c or [])]


def _jsonable(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 4)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return _jsonable(v.item())
    return v


def frame_records(df: pl.DataFrame) -> list[dict]:
    return [{k: _jsonable(v) for k, v in r.items()} for r in df.iter_rows(named=True)]


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


def _dir_entry(an: Analysis, dirid: int, uid: str, links: pl.DataFrame | None, stops: pl.DataFrame | None,
               names: dict[str, str]) -> dict[str, Any]:
    dd = an.dirs[dirid]
    fix = lambda n: names.get(n, n) if n is not None else n
    lk = _links_of(links, an.segments, uid)
    seg = an.segments.filter(pl.col("pattern_uid") == uid).sort("seg_idx")
    wk = an.to_contract_wk(dirid, pattern_uid=uid)
    # wk is in the analysis seg_key order of the pattern (= seg_idx order); align the geometry on it
    geo = dict(zip(seg["seg_key"].to_list(), seg["geometry"].to_list()))
    kpos = {k: i for i, k in enumerate(dd.seg_keys.tolist())}
    lpos = {k: i for i, k in enumerate(dd.link_keys.tolist())}
    D = len(dd.days)
    nd_seg = dd.V.sum(0)
    nd_link = dd.LV.sum(0)
    link_key_order = {k: i for i, k in enumerate(lk["link_key"].to_list())}
    s0 = lk["s0_m"].to_list()
    x_stops = s0 + [s0[-1] + lk["len_m"].to_list()[-1]] if lk.height else []
    stop_ids = lk["from_stop"].to_list() + lk["to_stop"].to_list()[-1:] if lk.height else []
    stop_names = lk["from_name"].to_list() + lk["to_name"].to_list()[-1:] if lk.height else []
    lls = _stop_ll(stops, lk) if lk.height else []

    def seg_flags(k):
        n = int(nd_seg[kpos[k]])
        return (Flag.FEW_DAYS if n < an.quality.min_days else 0) | (Flag.PARTIAL_VERSIONS if n < D else 0)

    def link_flags(k):
        n = int(nd_link[lpos[k]])
        return (Flag.FEW_DAYS if n < an.quality.min_days else 0) | (Flag.PARTIAL_VERSIONS if n < D else 0)

    keys = wk["seg_key"]
    x0 = dict(zip(seg["seg_key"].to_list(), seg["x0_m"].to_list()))
    ln = dict(zip(seg["seg_key"].to_list(), seg["len_m"].to_list()))
    zone = dict(zip(seg["seg_key"].to_list(), seg["zone"].to_list()))
    slink = dict(zip(seg["seg_key"].to_list(), seg["link_key"].to_list()))
    return {
        "dir": int(dirid),
        "pattern_uid": uid,
        "from": fix(stop_names[0]) if stop_names else None,
        "to": fix(stop_names[-1]) if stop_names else None,
        "stops": [{"id": i, "name": fix(n), "ll": ll, "x": round(float(x), 1)}
                  for i, n, ll, x in zip(stop_ids, stop_names, lls, x_stops)],
        "links": [{"key": r["link_key"], "from": fix(r["from_name"]), "to": fix(r["to_name"]),
                   "len": round(float(r["len_m"]), 1), "n_days": int(nd_link[lpos[r["link_key"]]]) if r["link_key"] in lpos else 0,
                   "flags": link_flags(r["link_key"]) if r["link_key"] in lpos else 0}
                  for r in lk.iter_rows(named=True)],
        "seg": {
            "key": keys,
            "link": [link_key_order.get(slink[k], -1) for k in keys],
            "x0": [round(float(x0[k]), 1) for k in keys],
            "len": [round(float(ln[k]), 1) for k in keys],
            "zone": [zone[k] for k in keys],
            "c": [_round_ll(geo.get(k)) for k in keys],
            "flags": [seg_flags(k) for k in keys],
        },
        "wk": {k: wk[k] for k in ("obs", "cov", "p", "days", "n")},
        "ref": _refs(an, dirid, keys),
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
                source: str | None = None) -> dict[str, Any]:
    """Build the JSON contract (``contract.py``) of an analysis.

    ``net``: the :class:`~microsegments.network.Network` (stop names and positions; without it stops are
    numbered). ``segments``: defaults to ``analysis.segments`` (must carry the geometry for the map).
    ``hotspots``: ``hotspots.hotspots(analysis)`` output. ``names``: stop name fixes {gtfs name: shown name},
    or "title" to title-case all-caps names (:func:`title_name`).
    ``coverage``: COVERAGE table for the coverage strip (else rebuilt from ``analysis.day_hours``)."""
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

    dirs = []
    for dirid in sorted(an.dirs):
        dd = an.dirs[dirid]
        if len(dd.days) == 0:
            continue
        e = _dir_entry(an, dirid, dd.display_uid, links, stops, names)
        from .metrics import version_groups
        others = [u for u in version_groups(dd) if u != dd.display_uid]   # one per distinct link sequence
        e["versions"] = [dd.display_uid] + others
        e["alt"] = [{k: v for k, v in _dir_entry(an, dirid, u, links, stops, names).items() if k != "dir"}
                    for u in others]
        dirs.append(e)

    # coverage per date x output hour, over every selected date (included or not)
    exc = {r[0]: r[1] for r in an.excluded_days.iter_rows()}
    dh = an.day_hours
    if coverage is not None:
        dh = coverage.select("service_date", "hour", "coverage")
    dates = sorted(set(dh["service_date"].to_list()) | set(exc) | set(an.days))
    rng = an.select.date_range()
    if rng is not None:
        dates = [d for d in dates if rng[0] <= d <= rng[1]]
    cmap = {(r[0], int(r[1])): r[2] for r in dh.select("service_date", "hour", "coverage").iter_rows()}
    c = [[None if cmap.get((d, h)) is None else round(float(cmap[(d, h)]), 3) for h in an.hours] for d in dates]
    inc_set = set(an.days)
    inc = [True if d in inc_set else exc.get(d, "not_selected") for d in dates]

    p = an.params
    period = {
        "first": (min(an.days) if an.days else (dates[0] if dates else None)),
        "last": (max(an.days) if an.days else (dates[-1] if dates else None)),
        "days": len(an.days),
        "per_dow": list(an.per_dow),
        "excluded": [{"date": d.isoformat(), "reason": r} for d, r in sorted(exc.items())],
    }
    period = {k: _jsonable(v) for k, v in period.items()}
    versions = []
    for r in an.versions.iter_rows(named=True):
        versions.append({"dir": int(r["direction_id"]), "pattern_uid": r["pattern_uid"],
                         "first": _jsonable(r["first"]), "last": _jsonable(r["last"]), "n_days": int(r["n_days"]),
                         "links_added": list(r["links_added"] or []), "links_removed": list(r["links_removed"] or []),
                         "display": bool(r["display"])})
    from .metrics import reference_agreement
    agree = frame_records(reference_agreement(an, hotspots if hotspots is not None and hotspots.height else None))
    from .hotspots import hotspot_status
    hs_status = "computed" if hotspots is not None and hotspot_status(an)[0] else "not_computed"
    hs = []
    if hotspots is not None and hotspots.height:
        keep = [c for c in hotspots.columns if c != "seg_keys"]
        for r in frame_records(hotspots.select(keep)):
            r["dir"] = r.get("direction_id")
            for k in ("from_stop_name", "to_stop_name"):
                if r.get(k) in names:
                    r[k] = names[r[k]]
            hs.append(r)
    return {
        "v": CONTRACT_VERSION,
        "line": line,
        "route_id": route_id,
        "mode": mode,
        "title": title,
        "source": source,
        "params": {"segment_m": p.segment_m, "phase_m": p.phase_m, "grid": p.grid, "tick_s": p.tick_s,
                   "gap_cap_s": p.gap_cap_s if math.isfinite(p.gap_cap_s) else None,
                   "stop_zone": list(p.stop_zone), "reference_hours": list(p.reference_hours)},
        "period": period,
        "hours": list(an.hours),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "day_hours": list(DAY_HOURS),
        "coverage": {"dates": [d.isoformat() for d in dates], "c": c, "inc": inc},
        "versions": versions,
        "dirs": dirs,
        "hotspots": hs,
        "hotspots_status": hs_status,
        "reference_agreement": agree,
        "ctx": ctx or [],
    }


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


__all__ = ["ROUTE_TYPES", "frame_records", "route_mode", "to_contract"]
