"""Passages (vehicles per link and hour), from position tracks and / or stop events.

Feed passages (port of ``track()`` of BrusselsTransitStuck/time30m_2025.py): along each pattern,
counting lines every 10 m (at 5, 15, 25 ... m); a track crossing a line in hour h adds one. Flow is
conserved inside a link, so every line of a link should see the same count; tracks only get lost
(where the feed loses vehicles: tunnels, termini where an arriving vehicle swaps places with a
departing one within one poll). Each link keeps the median over its lines more than 10 m from a
stop, over its second half only on the pattern's first link (away from the layover), over its first
half on the last link (away from the swap).

Tracks: ``track_id`` (a vehicle id, split into monotone runs: a new run when the vehicle goes back
more than 100 m or vanishes for more than 300 s), or, for anonymous feeds,
:func:`track_anonymous`: vehicles of one pattern do not overtake each other, so per service date and
pattern, the vehicles of each poll, sorted by position, are matched front to back to the open tracks.
A track matches a detection if it is at most 60 m ahead of it (feed noise) and not further behind
than 25 m/s x elapsed + 60 m; an unmatched track survives 300 s (vehicle missing from a few polls);
an unmatched detection opens a new track. A track that starts standing >= 100 s at an intermediate
stop is a short working waiting to start: those rows are layover (``count`` False, flag LAYOVER).

Event passages (port of ``passages()``): a trip passes link A->B (consecutive stop_sequence) if its
timed calls (observed or inferred) reach A or B; hour = time at A, interpolated between the trip's
timed calls, else time at B.

Both sources only ever miss vehicles, so ``n = max(n_feed, n_events)``. Rows are per
(service_date, hour, link_key): patterns sharing a link (branches, detours, short workings) add up,
``pattern_uid`` is the day's main pattern holding the link (else the busiest one).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import schema
from .io.ids import normalise_expr, normalise_py
from .schema import Flag

GHOST_S = 300.0
AHEAD_M = 60.0
VMAX_MS = 25.0
STANDING_S = 100.0
LINE_STEP_M = 10.0
LINE_OFFSET_M = 5.0
RESET_M = 100.0
STALE_S = 45.0


# ---------------------------------------------------------------- anonymous tracker
def _track_group(t: list, s: list, minpos: list, real: list, end_s: float | None, ghost: float, ahead: float,
                 vmax: float, stale: float | None = None):
    """Rows sorted by (t, s desc, row desc). ``minpos``: a track must be at least there to take the row
    (terminus pseudo rows: on the last link). ``end_s``: pattern end; a real (not pseudo) detection
    there prefers a reachable track still running over one already parked at the end (None: prototype
    rule). ``stale``: see the refinement in :func:`track_anonymous` (None: prototype rule).
    Returns (track index per row, tracks (row lists))."""
    n = len(t)
    tid = [0] * n
    tracks: list[list[int]] = []
    open_: list[list] = []          # [last_t, last_pos, track index]
    end = float("inf") if end_s is None else end_s - 0.01
    i = 0
    while i < n:
        tt = t[i]
        j = i
        while j < n and t[j] == tt:
            j += 1
        if open_:
            open_ = [tr for tr in open_ if tt - tr[0] <= ghost]
            open_.sort(key=lambda tr: -tr[1])
        k = 0
        for r in range(i, j):
            p = s[r]
            m = None
            while k < len(open_):
                tr = open_[k]
                if tr[1] > p + ahead:            # track ahead of this vehicle: not seen now
                    k += 1
                    continue
                if (stale is not None and tr[1] > p and tt - tr[0] > stale and k + 1 < len(open_)):
                    # a stale ghost just ahead vs a fresh track behind: vehicles do not reverse, the
                    # detection is the follower (the ghost's vehicle vanished, e.g. at a terminus)
                    nx = open_[k + 1]
                    if (nx[1] <= p and tt - nx[0] <= stale and p - nx[1] <= vmax * (tt - nx[0]) + ahead
                            and nx[1] >= minpos[r]):
                        k += 1
                        continue
                if p - tr[1] <= vmax * (tt - tr[0]) + ahead and tr[1] >= minpos[r]:
                    if p >= end and real[r] and tr[1] >= end and k + 1 < len(open_):
                        nx = open_[k + 1]
                        if p - nx[1] <= vmax * (tt - nx[0]) + ahead and nx[1] >= minpos[r]:
                            open_[k], open_[k + 1] = nx, tr
                            tr = nx
                    m = tr
                    k += 1
                break
            if m is None:
                m = [tt, p, len(tracks)]
                open_.append(m)
                tracks.append([])
            m[0] = tt
            if p > m[1]:
                m[1] = p
            tracks[m[2]].append(r)
            tid[r] = m[2]
        i = j
    return tid, tracks


def track_anonymous(placed: pl.DataFrame, *, ghost_s: float = GHOST_S, ahead_m: float = AHEAD_M,
                    vmax_ms: float = VMAX_MS, standing_s: float | None = STANDING_S,
                    compat: bool = False) -> pl.DataFrame:
    """Rebuild vehicle tracks of an anonymous feed: fills ``track_id`` (``date/pattern/n``) of every
    row with a position, and turns the standing start of short workings into layover (see module
    docstring; ``standing_s`` None disables it). Row order is kept.

    Three refinements over the prototype (``compat=True`` turns them off, for exact prototype
    tracks): a terminus pseudo row continues a track only if that track was on the last link (a
    ghost, e.g. a short working that ended earlier, must not jump to the terminus and count every
    line on the way); a detection at the pattern end prefers a reachable running track over one
    already parked there (else the arriving vehicle's track is orphaned and the next vehicle,
    picking it up, loses its crossings of the last link); a stale track (unseen for more than 45 s)
    just ahead of a detection does not take it when a fresh track behind can (vehicles do not reverse:
    the ghost's vehicle vanished and the detection is its follower)."""
    df = placed.with_row_index("_ix")
    ok = df.filter(pl.col("pattern_uid").is_not_null() & pl.col("s_m").is_not_null())
    ok = ok.sort(["service_date", "pattern_uid", "ts", "s_m", "_ix"], descending=[False, False, False, True, True])
    t_all = (ok["ts"].dt.epoch("us").to_numpy() / 1e6)
    s_all = ok["s_m"].cast(pl.Float64).to_numpy()
    fl = ok["flags"].fill_null(0).to_numpy().astype(np.int64)
    standing_all = ((fl & Flag.AT_STOP) != 0) & ((fl & Flag.LAYOVER) == 0)
    ix_all = ok["_ix"].to_numpy()
    pos_all = ok["pos_m"].cast(pl.Float64).fill_null(0.0).to_numpy()
    end_pseudo = ((fl & Flag.LAYOVER) != 0) & ~ok["count"].fill_null(True).to_numpy() & (pos_all > 0)
    real_all = ok["count"].fill_null(True).to_numpy() | ((fl & Flag.LAYOVER) == 0)
    minpos_all = np.where(end_pseudo & (not compat), s_all - pos_all - ahead_m, -np.inf)
    grp = ok.select(((pl.col("service_date") != pl.col("service_date").shift(1))
                     | (pl.col("pattern_uid") != pl.col("pattern_uid").shift(1))).fill_null(True).cum_sum()
                    .alias("g"))["g"].to_numpy()
    bounds = np.flatnonzero(np.diff(grp)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(grp)]])
    labels = np.empty(len(grp), dtype=object)
    drop = []
    dates = ok["service_date"].to_list()
    uids = ok["pattern_uid"].to_list()
    for a, b in zip(starts, ends):
        if b <= a:
            continue
        t = t_all[a:b].tolist()
        s = s_all[a:b].tolist()
        end_s = None if compat else float(max(s))
        tid, tracks = _track_group(t, s, minpos_all[a:b].tolist(), real_all[a:b].tolist(), end_s, ghost_s,
                                   ahead_m, vmax_ms, None if compat else STALE_S)
        prefix = f"{dates[a]}/{uids[a]}/"
        names = [prefix + str(k) for k in range(len(tracks))]
        labels[a:b] = [names[k] for k in tid]
        if standing_s is None:
            continue
        for rows in tracks:
            r0 = rows[0]
            if not standing_all[a + r0]:
                continue
            q = 0
            while q < len(rows) and abs(s[rows[q]] - s[r0]) < 1e-3:
                q += 1
            if t[rows[q - 1]] - t[r0] >= standing_s:
                drop += [a + r for r in rows[:q]]
    lab = pl.DataFrame({"_ix": ix_all, "_tid": pl.Series(labels.tolist(), dtype=pl.Utf8),
                        "_drop": np.isin(np.arange(len(grp)), drop)})
    out = df.join(lab, on="_ix", how="left", maintain_order="left")
    out = out.with_columns(
        pl.coalesce("_tid", "track_id").alias("track_id"),
        pl.when(pl.col("_drop").fill_null(False)).then(False).otherwise(pl.col("count")).alias("count"),
        pl.when(pl.col("_drop").fill_null(False)).then(pl.col("flags") | Flag.LAYOVER).otherwise(pl.col("flags"))
        .cast(pl.UInt16).alias("flags"),
    )
    return out.drop("_ix", "_tid", "_drop")


# ---------------------------------------------------------------- crossings of counting lines
def _service_hour(df: pl.DataFrame, t_col: str, tz: str) -> pl.Expr:
    local = pl.from_epoch((pl.col(t_col) * 1e6).cast(pl.Int64), time_unit="us").dt.replace_time_zone("UTC") \
        .dt.convert_time_zone(tz).dt.replace_time_zone(None)
    return ((local - pl.col("service_date").cast(pl.Datetime("us"))).dt.total_seconds() // 3600).cast(pl.Int8)


def crossings(placed: pl.DataFrame, *, ghost_s: float = GHOST_S, reset_m: float = RESET_M,
              step_m: float = LINE_STEP_M, offset_m: float = LINE_OFFSET_M, extend_short: bool = True) -> pl.DataFrame:
    """One row per counting-line crossing: service_date, pattern_uid, line (index k: line at
    offset + k*step metres), t (epoch s, interpolated along the track's running max position).

    ``extend_short``: a run whose last row is a short working on the link to its own terminus
    (``term_s_m`` set by the linear locator) reached that terminus: its feed rows there are layover
    and dropped, so the lines up to it are counted (at the time of the last row)."""
    d = (placed.filter(pl.col("track_id").is_not_null() & pl.col("s_m").is_not_null() & pl.col("pattern_uid").is_not_null())
         .select("service_date", "pattern_uid", "track_id", (pl.col("ts").dt.epoch("us") / 1e6).alias("t"),
                 pl.col("s_m").cast(pl.Float64).alias("s"),
                 (pl.col("term_s_m").cast(pl.Float64) if "term_s_m" in placed.columns else pl.lit(None, pl.Float64))
                 .alias("term"))
         .sort("pattern_uid", "service_date", "track_id", "t", maintain_order=True))
    empty = pl.DataFrame(schema={"service_date": pl.Date, "pattern_uid": pl.Utf8, "line": pl.Int64, "t": pl.Float64})
    if not d.height:
        return empty
    key = ["pattern_uid", "service_date", "track_id"]
    newkey = pl.any_horizontal([pl.col(c) != pl.col(c).shift(1) for c in key]).fill_null(True)
    brk = newkey | ((pl.col("t") - pl.col("t").shift(1)) > ghost_s) | ((pl.col("s") - pl.col("s").shift(1)) < -reset_m)
    d = d.with_columns(brk.fill_null(True).cum_sum().alias("run"))
    d = d.with_columns(pl.col("s").cum_max().over("run").alias("cm"))
    runs = (d.group_by("run", maintain_order=True)
            .agg(pl.col("service_date").first(), pl.col("pattern_uid").first(), pl.len().alias("n"),
                 pl.col("s").first().alias("s0"), pl.col("cm").last().alias("s1"),
                 pl.col("term").last().alias("term"), pl.col("t").last().alias("tl"))
            .filter(pl.col("n") >= 2))
    if not runs.height:
        return empty
    s1 = runs["s1"].to_numpy()
    s1x = np.fmax(s1, runs["term"].to_numpy()) if extend_short else s1
    kmin = np.floor((runs["s0"].to_numpy() - offset_m) / step_m).astype(np.int64) + 1
    kmax = np.floor((s1x - offset_m) / step_m).astype(np.int64)
    # lines exactly at s0 are not crossed (> s0), lines at s1 are (<= s1)
    nl = np.maximum(kmax - np.maximum(kmin, 0) + 1, 0)
    kmin = np.maximum(kmin, 0)
    rid = np.repeat(runs["run"].to_numpy(), nl)
    first = np.repeat(np.cumsum(nl) - nl, nl)
    k = np.repeat(kmin, nl) + (np.arange(nl.sum()) - first)
    line_s = offset_m + k * step_m
    big = 1e7
    xp = d["run"].to_numpy().astype(np.float64) * big + d["cm"].to_numpy()
    fp = d["t"].to_numpy()
    tc = np.interp(rid.astype(np.float64) * big + line_s, xp, fp)
    sel = np.repeat(np.arange(runs.height), nl)
    beyond = line_s > s1[sel]
    tc[beyond] = runs["tl"].to_numpy()[sel][beyond]
    return pl.DataFrame({"service_date": runs["service_date"].gather(sel), "pattern_uid": runs["pattern_uid"].gather(sel),
                         "line": k, "t": tc})


def _line_selection(stops_s: np.ndarray, lines: np.ndarray) -> list[np.ndarray]:
    """Counting lines used for each link (prototype rules)."""
    out = []
    n = len(stops_s) - 1
    for i in range(n):
        lo, hi = stops_s[i], stops_s[i + 1]
        mid = (lo + hi) / 2
        if i == 0 and n > 1:
            sel = (lines > mid) & (lines < hi - 10)
        elif i == n - 1 and n > 1:
            sel = (lines > lo + 10) & (lines <= mid)
        else:
            sel = (lines > lo + 10) & (lines < hi - 10)
        if not sel.any():
            sel = (lines > lo) & (lines < hi)
        out.append(np.flatnonzero(sel))
    return out


def feed_passages(placed: pl.DataFrame, net, *, tz: str = "Europe/Brussels", ghost_s: float = GHOST_S,
                  reset_m: float = RESET_M, extend_short: bool = True) -> pl.DataFrame:
    """Median line crossings per (service_date, hour, pattern_uid, link_key) -> column n_feed."""
    from .locate.common import pattern_infos
    cr = crossings(placed, ghost_s=ghost_s, reset_m=reset_m, extend_short=extend_short)
    out_schema = {"service_date": pl.Date, "hour": pl.Int8, "pattern_uid": pl.Utf8, "link_key": pl.Utf8,
                  "n_feed": pl.Float64}
    if not cr.height:
        return pl.DataFrame(schema=out_schema)
    cr = cr.with_columns(_service_hour(cr, "t", tz).alias("hour"))
    cnt = cr.group_by("pattern_uid", "service_date", "hour", "line").agg(pl.len().alias("n"))
    infos = pattern_infos(net)
    parts = []
    for (uid,), g in cnt.group_by("pattern_uid"):
        P = infos.get(uid)
        if P is None:
            continue
        lines = np.arange(LINE_OFFSET_M, P.length, LINE_STEP_M)
        g = g.filter(pl.col("line") < len(lines))
        dh = g.select("service_date", "hour").unique().sort("service_date", "hour").with_row_index("r")
        g = g.join(dh, on=["service_date", "hour"])
        C = np.zeros((dh.height, len(lines)))
        C[g["r"].to_numpy(), g["line"].to_numpy()] = g["n"].to_numpy()
        sels = _line_selection(P.s, lines)
        med = np.column_stack([np.median(C[:, sel], axis=1) if len(sel) else np.zeros(dh.height) for sel in sels])
        rr, ll = np.nonzero(med > 0)
        parts.append(pl.DataFrame({
            "service_date": dh["service_date"].gather(rr), "hour": dh["hour"].gather(rr),
            "pattern_uid": [uid] * len(rr), "link_key": [P.keys[i] for i in ll], "n_feed": med[rr, ll]},
            schema=out_schema))
    return pl.concat(parts) if parts else pl.DataFrame(schema=out_schema)


# ---------------------------------------------------------------- stop events
def _link_keys_by_date(net, dates, id_normaliser: str):
    """service_date -> table (a, b) normalised stop pair -> link_key, pattern_uid, is_main."""
    from .locate.common import date_versions, pattern_infos

    def norm(x):
        return normalise_py(x, id_normaliser)
    infos = pattern_infos(net, norm)
    dv, versions, _ = date_versions(net, dates, infos)
    rows = []
    for vi, V in enumerate(versions):
        for rank, P in enumerate(V.mains + V.shorts + V.others):
            is_main = rank < len(V.mains)
            for i in range(len(P.lens)):
                rows.append((vi, P.norm[i], P.norm[i + 1], P.keys[i], P.uid, is_main, P.direction))
    tab = pl.DataFrame(rows, orient="row", schema={"_ver": pl.Int32, "a": pl.Utf8, "b": pl.Utf8, "link_key": pl.Utf8,
                                                   "pattern_uid": pl.Utf8, "is_main": pl.Boolean, "direction_id": pl.Int8})
    tab = tab.sort("_ver", "a", "b", "is_main", descending=[False, False, False, True]).unique(["_ver", "a", "b"], keep="first")
    return dv.select("service_date", "_ver"), tab


def event_passages(events, net, *, tz: str = "Europe/Brussels", id_normaliser: str = "leading_digits") -> pl.DataFrame:
    """Trips per (service_date, hour, link_key) from STOP_EVENT rows -> column n_events."""
    ev = events.collect() if isinstance(events, pl.LazyFrame) else events
    out_schema = {"service_date": pl.Date, "hour": pl.Int8, "pattern_uid": pl.Utf8, "link_key": pl.Utf8,
                  "n_events": pl.Float64}
    if not ev.height:
        return pl.DataFrame(schema=out_schema)
    trip = ["service_date", "trip_key"]
    ev = (ev.sort("service_date", "trip_key", "stop_sequence")
          .with_columns(normalise_expr(pl.col("stop_id"), id_normaliser).alias("a"),
                        (pl.coalesce("departure", "arrival").dt.epoch("us") / 1e6).alias("t"))
          .with_columns(pl.int_range(pl.len()).over(trip).alias("k")))
    ev = ev.with_columns(
        pl.col("t").interpolate().over(trip).alias("e"),
        pl.when(pl.col("t").is_not_null()).then(pl.col("k")).min().over(trip).alias("f"),
        pl.when(pl.col("t").is_not_null()).then(pl.col("k")).max().over(trip).alias("l"),
    ).with_columns(
        pl.col("a").shift(-1).over(trip).alias("b"),
        pl.col("stop_sequence").shift(-1).over(trip).alias("next_seq"),
        pl.col("e").shift(-1).over(trip).alias("eb"),
    )
    lk = ev.filter((pl.col("next_seq") == pl.col("stop_sequence") + 1) & (pl.col("f") <= pl.col("k") + 1)
                   & (pl.col("l") >= pl.col("k")))
    lk = lk.with_columns(pl.coalesce("e", "eb").alias("tt")).filter(pl.col("tt").is_not_null())
    lk = lk.with_columns(_service_hour(lk, "tt", tz).alias("hour"))
    dv, tab = _link_keys_by_date(net, lk["service_date"].unique().to_list(), id_normaliser)
    lk = lk.join(dv, on="service_date", how="left").join(tab, on=["_ver", "a", "b"], how="inner")
    return (lk.group_by("service_date", "hour", "link_key")
            .agg(pl.col("pattern_uid").first(), pl.len().cast(pl.Float64).alias("n_events"))
            .select(list(out_schema)))


# ---------------------------------------------------------------- combined
def passages(placed, net, events=None, mode: str = "auto", *, tz: str = "Europe/Brussels",
             id_normaliser: str = "leading_digits", compat: bool = False) -> pl.DataFrame:
    """PASSAGES per (service_date, hour, link_key).

    ``placed``: Placed or PLACED frame. ``mode``: "auto" (feed tracks + events when given, n = max),
    "feed", "events". Rows without any ``track_id`` are first tracked with :func:`track_anonymous`.
    ``compat``: the prototype's exact tracker and no short-working extension (regression tests)."""
    frame = getattr(placed, "frame", placed)
    use_feed = mode in ("auto", "feed")
    use_ev = mode in ("auto", "events") and events is not None
    if mode not in ("auto", "feed", "events"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "events" and events is None:
        raise ValueError("mode='events' needs events")
    key = ["service_date", "hour", "link_key"]
    parts = []
    if use_feed:
        if frame.height and frame["track_id"].null_count() == frame.height:
            frame = track_anonymous(frame, compat=compat)
        f = feed_passages(frame, net, tz=tz, extend_short=not compat)
        parts.append(f.group_by(key).agg(pl.col("n_feed").sum(),
                                         pl.col("pattern_uid").sort_by("n_feed", descending=True).first().alias("_pf")))
    if use_ev:
        e = event_passages(events, net, tz=tz, id_normaliser=id_normaliser)
        parts.append(e.rename({"pattern_uid": "_pe"}))
    if not parts:
        return schema.empty(schema.PASSAGES)
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on=key, how="full", coalesce=True)
    if "n_feed" not in out.columns:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("n_feed"), pl.lit(None, pl.Utf8).alias("_pf"))
    elif use_feed:
        out = out.with_columns(pl.col("n_feed").fill_null(0.0))
    if "n_events" not in out.columns:
        out = out.with_columns(pl.lit(None, pl.Float64).alias("n_events"), pl.lit(None, pl.Utf8).alias("_pe"))
    elif use_ev:
        out = out.with_columns(pl.col("n_events").fill_null(0.0))
    # pattern_uid: the day's main pattern holding the link, else the busiest source pattern
    mains = net.mains().join(net.links.select("pattern_uid", "link_key"), on="pattern_uid") \
        .select("service_date", "link_key", pl.col("pattern_uid").alias("_pm")).unique(["service_date", "link_key"])
    out = out.join(mains, on=["service_date", "link_key"], how="left")
    out = out.with_columns(pl.coalesce("_pm", "_pf", "_pe").alias("pattern_uid"),
                           pl.max_horizontal("n_feed", "n_events").alias("n"))
    out = out.select([pl.col(c).cast(t) for c, t in schema.PASSAGES.items()])
    return out.sort("service_date", "pattern_uid", "link_key", "hour")


__all__ = ["track_anonymous", "crossings", "feed_passages", "event_passages", "passages"]
