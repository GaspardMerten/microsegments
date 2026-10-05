"""Linear-referenced feeds ("last stop passed + metres since", STIB vehicle-distance) -> Placed.

Port of ``place()`` of the tram 55 prototype (BrusselsTransitStuck/time30m_2025.py), per service
date and GTFS version:

1. **Pattern.** A row (stop ``p``, terminus code ``t``) goes to the day's main pattern that holds
   ``p`` and whose remaining stops contain the terminus (same id, or same stop *name*: STIB codes a
   terminus by one platform, the pattern may stop at the other one). Exactly one main must match,
   else ``ambiguous``; no main: the day's detour / other patterns are tried (rows keep that
   pattern; several fit: the one whose last stop is the terminus, then the busiest that day), then
   ``off_pattern`` (this also drops rows at or past the vehicle's own terminus). Unless ``compat``,
   a row whose terminus code fits no pattern at its stop is placed by the stop alone if a single
   main serves it and the code is that main's first stop (some feeds code both directions with
   one terminus, e.g. STIB bus 58).
   Short workings ride their parent main pattern (flag SHORT_WORKING when the terminus is not the
   main's last stop; extra column ``term_s_m``: where their terminus is, on the link leading to
   it). Without a terminus code, ``direction_id`` (if any) restricts the mains.
   Terminus codes that are no stop of the network (STIB 9943 for Da Vinci) are learned by majority
   vote: the last stop of the main pattern that holds most of the code's rows' stops.
2. **Terminus pseudo rows** (``count`` False, flag LAYOVER): a row whose stop has the name of a
   main's first (last) stop and is within ``layover_m`` of it, or heads to it, is also put at
   ``s = 0`` (``s = length``) of that main, so the tracker sees a vehicle reach / leave a terminus
   between two polls.
3. **Layover.** Rows on link 0 under ``layover_m`` from the first stop are dropped (the pseudo row
   stands for them).
4. **Feed length.** Feed metres are rescaled per (pattern, link) to GTFS metres:
   ``pos = min(dist / flen, 1) * len``, ``flen`` = p99.5 of the link's feed distances over the call's
   rows (after the layover drop, before the short-working drop). ``feed_length="gtfs"``: no rescale;
   a DataFrame (pattern_uid, link_idx, flen) reuses lengths learned on another period.
5. **At stop.** ``dist < at_stop_m`` is the vehicle standing at the stop: it is placed at the end of
   the link arriving there (``pos = len`` of link i-1, flag AT_STOP), as in the half-open ]A-B].
6. **Short working end.** A short working in the last quarter (``frac >= 0.75``) of the link that
   leads to its own terminus stands at its terminus platform: dropped.

The "short working start" rule (a track starting with >= 100 s standing at an intermediate stop) needs
tracks: see :func:`microsegments.tracks.track_anonymous`, applied by :func:`microsegments.locate.place`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl

from .. import schema
from ..config import Params
from ..io.ids import normalise_py
from ..schema import Flag
from .common import PatInfo, Placed, Version, date_versions, ensure_local_time, finish, pattern_infos

AT_STOP_M = 10.0
SW_END_FRAC = 0.75
FLEN_Q = 0.995
_NULL = "\x00"


def _names_map(net, normalise) -> dict[str, str]:
    out: dict[str, str] = {}
    for sid, name in net.stops.select("stop_id", "stop_name").iter_rows():
        k = normalise(sid)
        if k is not None and k not in out:
            out[k] = name
    return out


def learn_aliases(df: pl.DataFrame, versions: list[Version], known: set[str],
                  given: dict[str, str]) -> dict[str, str]:
    """Terminus codes unknown to the network -> last stop of the pattern (mains first, then the
    day's other patterns) holding most of their rows' stops."""
    codes = (df.filter(pl.col("terminus_id").is_not_null() & ~pl.col("terminus_id").is_in(list(known | set(given))))
             .group_by("terminus_id", "stop_id").agg(n=pl.len()))
    out = dict(given)
    if not len(codes):
        return out
    # main patterns first (ties go to them), then the day's other patterns
    pats: dict[str, PatInfo] = {}
    for v in versions:
        for p in v.mains:
            pats.setdefault(p.uid, p)
    for v in versions:
        for p in v.others:
            pats.setdefault(p.uid, p)
    for (code,), g in codes.group_by("terminus_id"):
        cnt = dict(zip(g["stop_id"].to_list(), g["n"].to_list()))
        best, score = None, 0
        for p in pats.values():
            sc = sum(cnt.get(s, 0) for s in set(p.norm[:-1]))
            if sc > score:
                best, score = p, sc
        if best is not None:
            out[code] = best.norm[-1]
    return out


def _match(pats: list[PatInfo], p: str, tid: str | None, tname: str | None, direction: int | None):
    hits = []
    for P in pats:
        if p not in P.norm:
            continue
        i = P.norm.index(p)
        last = len(P.norm) - 1
        if i == last:
            continue
        if tid is None and tname is None:
            if direction is None or P.direction == direction:
                hits.append((P, i, last))
            continue
        j = next((k for k in range(i + 1, last + 1)
                  if P.norm[k] == tid or (tname is not None and P.names[k] == tname)), None)
        if j is not None:
            hits.append((P, i, j))
    return hits


def _resolve(combos: pl.DataFrame, versions: list[Version], aliases: dict[str, str], names: dict[str, str],
             compat: bool = False):
    rows = []
    for ver, p, t, d in combos.iter_rows():
        tid = None if t is None else aliases.get(t, t)
        tname = None if tid is None else names.get(tid)
        V = versions[ver]
        status, hit = "off_pattern", None
        if p is None:
            status = "no_position"
        else:
            hits = _match(V.mains, p, tid, tname, d)
            if len(hits) == 1:
                status, hit = "ok", hits[0]
            elif len(hits) > 1:
                status = "ambiguous"
            else:
                hits = _match(V.others, p, tid, tname, d)
                if len(hits) > 1 and not compat:
                    # several detours / other patterns fit: the one run to its end (not a part of
                    # it), then the busiest that day (others are sorted by trips), same direction only
                    full = [h for h in hits if h[2] == len(h[0].norm) - 1] or hits
                    if len({h[0].direction for h in full}) == 1:
                        hits = full[:1]
                if len(hits) == 1:
                    status, hit = "ok_other", hits[0]
                elif len(hits) > 1:
                    status = "ambiguous"
                elif t is not None and tid not in names:
                    status = "unknown_terminus"
                if hit is None and not compat and status != "ambiguous":
                    # the terminus code fits no pattern here (some feeds code both directions with one
                    # terminus): the stop alone, if exactly one main pattern serves it (not as its end)
                    # and the code is that main's first stop (a vehicle past an intermediate own
                    # terminus, e.g. a short working running on to the depot, stays dropped)
                    hits = _match(V.mains, p, None, None, d)
                    if len(hits) == 1 and tid is not None and (
                            hits[0][0].norm[0] == tid or (tname is not None and hits[0][0].names[0] == tname)):
                        status, hit = "ok_stop_only", hits[0]
        if hit is None:
            rows.append((ver, p, t, d, status, None, None, None, None, None, None))
        else:
            P, i, j = hit
            last = len(P.norm) - 1
            sw_next = j == i + 1 and j != last and P.names[j] != P.names[last]
            rows.append((ver, p, t, d, status, P.uid, i, j, j != last, sw_next, P.direction))
    return pl.DataFrame(rows, orient="row", schema={
        "_ver": pl.Int32, "_p": pl.Utf8, "_t": pl.Utf8, "_d": pl.Int8, "_status": pl.Utf8,
        "pattern_uid": pl.Utf8, "_li": pl.Int16, "_term": pl.Int16, "_short": pl.Boolean, "_sw_next": pl.Boolean,
        "_pdir": pl.Int8})


def _link_table(infos: dict[str, PatInfo], uids) -> pl.DataFrame:
    rows = []
    for u in uids:
        P = infos[u]
        for i in range(len(P.lens)):
            rows.append((u, i, float(P.lens[i]), float(P.s[i]), P.keys[i]))
    return pl.DataFrame(rows, orient="row", schema={"pattern_uid": pl.Utf8, "_li": pl.Int16, "_glen": pl.Float64,
                                                    "_s0": pl.Float64, "_key": pl.Utf8})


def feed_lengths(rows: pl.DataFrame, links: pl.DataFrame, q: float = FLEN_Q, min_rows: int = 20) -> pl.DataFrame:
    """p``q`` of feed metres per (pattern_uid, link_idx); GTFS length when fewer than ``min_rows``."""
    f = (rows.group_by("pattern_uid", "_li")
         .agg(pl.col("dist_m").cast(pl.Float64).quantile(q, interpolation="linear").alias("_flen"), pl.len().alias("n")))
    out = links.join(f, on=["pattern_uid", "_li"], how="left").with_columns(
        pl.col("n").fill_null(0).cast(pl.UInt32))
    out = out.with_columns(pl.when((pl.col("n") >= min_rows) & (pl.col("_flen") > 0)).then(pl.col("_flen"))
                           .otherwise(pl.col("_glen")).alias("_flen"))
    return out.select(pl.col("pattern_uid"), pl.col("_li").alias("link_idx"), pl.col("_key").alias("link_key"),
                      pl.col("_glen").alias("glen"), pl.col("_flen").alias("flen"), "n")


def place_linear(obs, net, params: Params | None = None, *, feed_length="auto",
                 terminus_aliases: dict[str, str] | None = None, tz: str = "Europe/Brussels",
                 service_day_start: str = "04:00", id_normaliser: str = "leading_digits",
                 at_stop_m: float = AT_STOP_M, flen_min_rows: int = 20, compat: bool = False) -> Placed:
    """Place linear observations (``stop_id`` + ``dist_m`` [+ ``terminus_id`` / ``direction_id``]).

    See the module docstring. ``obs``: OBSERVATION frame (lazy or not; extra columns are carried
    along); stop and terminus ids must already be normalised like ``id_normaliser`` does (io does
    it), the network's GTFS ids are normalised here. ``terminus_aliases``: {code: stop_id} known
    in advance (others are learned). ``compat``: the prototype's rules exactly (several other patterns
    fitting = ambiguous; no stop-only placement)."""
    params = params or Params()
    if isinstance(obs, pl.LazyFrame):
        obs = obs.collect()
    df = schema.conform(obs, schema.OBSERVATION, schema.OBSERVATION_REQUIRED)
    extras = [c for c in df.columns if c not in schema.OBSERVATION and c not in ("service_date", "hour", "dow")]
    df = ensure_local_time(df, tz, service_day_start).with_row_index("obs_row")
    n0 = df.height

    def norm(x):
        return normalise_py(x, id_normaliser)

    infos = pattern_infos(net, norm)
    names = _names_map(net, norm)
    dv, versions, n_fb = date_versions(net, df["service_date"].unique().to_list(), infos)
    df = df.join(dv.select("service_date", "_ver"), on="service_date", how="left")
    given = {str(k): norm(v) or str(v) for k, v in (terminus_aliases or {}).items()}
    aliases = learn_aliases(df, versions, set(names), given)

    # ---- pattern of each (version, stop, terminus, direction)
    key = [pl.col("_ver"), pl.col("stop_id").alias("_p"), pl.col("terminus_id").alias("_t"),
           pl.col("direction_id").alias("_d")]
    combos = df.select(key).unique()
    res = _resolve(combos, versions, aliases, names, compat)
    fill = [pl.col("_p").fill_null(_NULL), pl.col("_t").fill_null(_NULL), pl.col("_d").fill_null(-1)]
    df = (df.with_columns(key[1:]).with_columns(fill)
          .join(res.with_columns(fill), on=["_ver", "_p", "_t", "_d"], how="left", maintain_order="left"))
    df = df.with_columns(pl.when(pl.col("dist_m").is_null()).then(pl.lit("no_position")).otherwise(pl.col("_status"))
                         .alias("_status"))
    report = Counter(df["_status"].to_list())

    # ---- terminus pseudo rows (feed the tracker, never counted)
    pname = pl.col("stop_id").replace_strict(names, default=None, return_dtype=pl.Utf8)
    tname = (pl.col("terminus_id").replace_strict(aliases, default=pl.col("terminus_id"), return_dtype=pl.Utf8)
             .replace_strict(names, default=None, return_dtype=pl.Utf8))
    base = df.with_columns(pname.alias("_pname"), tname.alias("_tname"))
    pseudo = []
    for vi, V in enumerate(versions):
        bv = base.filter(pl.col("_ver") == vi)
        if not bv.height:
            continue
        for P in V.mains:
            for end in (0, -1):
                N = P.names[end]
                if N is None:
                    continue
                m = bv.filter((pl.col("_pname") == N) & ((pl.col("dist_m") < params.layover_m) | (pl.col("_tname") == N)))
                if not m.height:
                    continue
                li = 0 if end == 0 else len(P.lens) - 1
                pos = 0.0 if end == 0 else float(P.lens[-1])
                pseudo.append(m.with_columns(
                    pl.lit(P.uid).alias("pattern_uid"), pl.lit(li, pl.Int16).alias("link_idx"),
                    pl.lit(P.keys[li]).alias("link_key"), pl.lit(pos).alias("pos_m"),
                    pl.lit(0.0 if end == 0 else P.length).alias("s_m"), pl.lit(P.direction, pl.Int8).alias("_pdir"),
                    pl.lit(False).alias("count"), pl.lit(Flag.LAYOVER, pl.UInt16).alias("flags"),
                    pl.lit(None, pl.Float64).alias("term_s_m")))

    # ---- placed rows
    x = df.filter(pl.col("_status").is_in(["ok", "ok_other", "ok_stop_only"]))
    lay = (pl.col("_li") == 0) & (pl.col("dist_m") < params.layover_m)
    report["layover_first_stop"] = int(x.select(lay.sum()).item())
    x = x.filter(~lay)
    links = _link_table(infos, sorted({u for V in versions for u in [p.uid for p in V.mains + V.others]}))
    if isinstance(feed_length, pl.DataFrame):
        fl = feed_length.select("pattern_uid", pl.col("link_idx").cast(pl.Int16).alias("_li"),
                                pl.col("flen").cast(pl.Float64).alias("_flen"))
        flen = (links.join(fl, on=["pattern_uid", "_li"], how="left")
                .with_columns(pl.col("_flen").fill_null(pl.col("_glen")))
                .select("pattern_uid", pl.col("_li").alias("link_idx"), pl.col("_key").alias("link_key"),
                        pl.col("_glen").alias("glen"), pl.col("_flen").alias("flen"), pl.lit(None, pl.UInt32).alias("n")))
    elif feed_length == "gtfs":
        flen = links.select("pattern_uid", pl.col("_li").alias("link_idx"), pl.col("_key").alias("link_key"),
                            pl.col("_glen").alias("glen"), pl.col("_glen").alias("flen"), pl.lit(None, pl.UInt32).alias("n"))
    elif feed_length == "auto":
        flen = feed_lengths(x, links, min_rows=flen_min_rows)
    else:
        raise ValueError(f"feed_length must be 'auto', 'gtfs' or a DataFrame, not {feed_length!r}")
    x = x.join(flen.select("pattern_uid", pl.col("link_idx").alias("_li"), pl.col("flen").alias("_flen")),
               on=["pattern_uid", "_li"], how="left").join(links, on=["pattern_uid", "_li"], how="left")
    frac = pl.min_horizontal(pl.col("dist_m").cast(pl.Float64) / pl.col("_flen"), pl.lit(1.0))
    x = x.with_columns(frac.alias("_frac"))
    sw_end = pl.col("_short") & pl.col("_sw_next") & (pl.col("_frac") >= SW_END_FRAC)
    report["layover_short_working_end"] = int(x.select(sw_end.sum()).item())
    x = x.filter(~sw_end)
    at = pl.col("dist_m") < at_stop_m
    # standing at stop i (> 0): end of link i - 1
    back = at & (pl.col("_li") > 0)
    x = x.with_columns(pl.when(at).then(0.0).otherwise(pl.col("_frac") * pl.col("_glen")).alias("pos_m"),
                       at.alias("_at"))
    x = x.with_columns(pl.when(back).then(pl.col("_li") - 1).otherwise(pl.col("_li")).cast(pl.Int16).alias("link_idx"))
    x = x.join(links.select("pattern_uid", pl.col("_li").alias("link_idx"), pl.col("_glen").alias("_glen2"),
                            pl.col("_s0").alias("_s02"), pl.col("_key").alias("link_key")),
               on=["pattern_uid", "link_idx"], how="left")
    x = x.with_columns(pl.when(back).then(pl.col("_glen2")).otherwise(pl.col("pos_m")).alias("pos_m"))
    x = x.with_columns((pl.col("_s02") + pl.col("pos_m")).alias("s_m"),
                       # short working on the link leading to its own terminus: where that terminus is
                       pl.when(pl.col("_short") & (pl.col("link_idx") + 1 == pl.col("_term")))
                       .then(pl.col("_s02") + pl.col("_glen2")).otherwise(None).alias("term_s_m"))
    flags = (pl.when(pl.col("_short")).then(Flag.SHORT_WORKING).otherwise(0)
             + pl.when(pl.col("_at")).then(Flag.AT_STOP).otherwise(0))
    x = x.with_columns(flags.cast(pl.UInt16).alias("flags"), pl.lit(True).alias("count"))

    cols = ["ts", "service_date", "hour", "dow", "route_id", "pattern_uid", "link_idx", "link_key", "pos_m", "s_m",
            "dist_m", "vehicle_id", "count", "flags", "_pdir", "obs_row", "term_s_m"] + extras
    parts = [x.select(cols)] + [p.select(cols) for p in pseudo]
    out = pl.concat(parts, how="vertical_relaxed") if parts else x.select(cols)
    out = out.with_columns(pl.col("dist_m").alias("raw_dist_m"), pl.col("vehicle_id").alias("track_id"),
                           pl.col("_pdir").alias("direction_id"))
    out = finish(out, ["direction_id", "obs_row", "term_s_m"] + extras)

    report = {k: int(v) for k, v in report.items()}
    report["total"] = n0
    report["terminus_pseudo"] = sum(p.height for p in pseudo)
    report["counted"] = x.height
    report["gtfs_date_fallback_days"] = n_fb
    learned = {k: v for k, v in aliases.items() if k not in given}
    report["aliases_learned"] = len(learned)
    return Placed(frame=out, report=report, aliases=aliases, flen=flen)
