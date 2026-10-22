"""Lat/lon fixes (GTFS-RT VehiclePositions, AVL exports) -> Placed.

Each fix is projected (local metric frame, shapely STRtree over the shape segments) onto the
patterns running that service date: the main pattern of each direction and the day's detours /
other patterns (short workings ride their parent main). Candidates farther than
``params.max_lateral_m`` are rejected; a fix with no candidate is ``off_pattern`` (dropped,
reported). On a self-overlapping pattern (loop, out-and-back) a fix has several candidate passes;
the vehicle's previous position picks the next pass along (monotone ``s``).

Pattern choice, first that applies:

1. ``trip_id`` found among the GTFS trips of the network (pattern of the trip; short working ->
   its parent main, flag SHORT_WORKING);
2. ``direction_id``: candidates of that direction only (then as below among them);
3. per-vehicle continuity (``vehicle_id``): over a sliding window of the vehicle's fixes, the
   candidate along which the vehicle progresses (``s`` increasing) wins; several progressing
   candidates (a detour sharing the trunk): lowest lateral error, main preferred;
4. ``bearing`` vs the shape tangent (within 45 degrees);
5. a single candidate within reach; else ``ambiguous`` (dropped, reported).

Positions: ``s_m`` along the pattern (a fix less than ``at_stop_m`` past a stop is standing at it:
snapped to the stop, flag AT_STOP), ``link_idx`` / ``pos_m`` with the same half-open ``(lo, hi]``
links as the linear locator (a vehicle at stop B is at the end of link A->B). A fix within
``params.layover_m`` of the pattern's (or the short working's) first stop, or at / past its last
stop (within ``layover_m`` of it when the direction was inferred), is terminus layover: kept for
tracking, flag LAYOVER, ``count`` False. ``track_id`` =
``vehicle_id`` (else ``trip_id``).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl
import shapely

from ..config import Params
from ..network.geometry import LocalFrame, cumlen
from ..schema import Flag
from .common import (
    PatInfo,
    Placed,
    date_versions,
    finish,
    link_of_s,
    pattern_infos,
    prepare_obs,
)

WINDOW = 3                 # fixes on each side for the continuity score
MIN_PROGRESS_M = 15.0      # net progress over the window to call a direction
MARGIN_M = 15.0
MAX_STEP_M = 300.0
BEARING_TOL = 45.0
MAIN_PREF_M = 2.0
CHUNK = 200_000
AT_STOP_M = 10.0


class _Shape:
    """A pattern's polyline in the local frame with along-pattern metres at each vertex."""

    def __init__(self, P: PatInfo, coords_ll: np.ndarray, frame: LocalFrame):
        ll = np.asarray(coords_ll, dtype=np.float64).reshape(-1, 2)
        keep = np.ones(len(ll), bool)
        keep[1:] = np.any(np.diff(ll, axis=0) != 0, axis=1)
        ll = ll[keep]
        self.xy = frame.xy(ll)
        cum = cumlen(ll)
        self.cum = cum * (P.length / cum[-1]) if cum[-1] > 0 else cum
        self.tree = shapely.STRtree(shapely.linestrings(np.stack([self.xy[:-1], self.xy[1:]], axis=1)))
        v = np.diff(self.xy, axis=0)
        self.bearing = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360

    def project(self, pts: np.ndarray, max_d: float):
        """Candidate passes within max_d: arrays (point idx, s, lateral d, bearing) one per pass."""
        outs = []
        for c0 in range(0, len(pts), CHUNK):
            p = pts[c0:c0 + CHUNK]
            q = self.tree.query(shapely.points(p), predicate="dwithin", distance=max_d)
            if not q.shape[1]:
                continue
            i, k = q[0], q[1]
            a = self.xy[k]
            v = self.xy[k + 1] - a
            vv = (v * v).sum(axis=1)
            rel = p[i] - a
            t = np.clip(np.where(vv > 0, (rel * v).sum(axis=1) / np.where(vv > 0, vv, 1), 0), 0, 1)
            d = np.hypot(*(rel - t[:, None] * v).T)
            s = self.cum[k] + t * (self.cum[k + 1] - self.cum[k])
            o = np.lexsort((s, i))
            i, s, d, k = i[o], s[o], d[o], k[o]
            # passes: contiguous runs of s per point (a gap of more than 2 x max_d = another pass)
            new = np.ones(len(i), bool)
            new[1:] = (i[1:] != i[:-1]) | (s[1:] - s[:-1] > 2 * max_d)
            cid = np.cumsum(new) - 1
            best = np.full(cid[-1] + 1, np.inf)
            np.minimum.at(best, cid, d)
            pick = d <= best[cid]
            first = np.zeros(len(i), bool)
            first[np.flatnonzero(pick)[np.unique(cid[pick], return_index=True)[1]]] = True
            outs.append((i[first] + c0, s[first], d[first], self.bearing[k[first]]))
        if not outs:
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
        return tuple(np.concatenate(x) for x in zip(*outs))


def _angle(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def place_latlon(obs, net, params: Params | None = None, *, tz: str = "Europe/Brussels",
                 service_day_start: str = "04:00", window: int = WINDOW, at_stop_m: float = AT_STOP_M) -> Placed:
    """Map-match lat/lon observations onto the network's patterns. See the module docstring."""
    params = params or Params()
    df, extras = prepare_obs(obs, tz, service_day_start)
    report: Counter = Counter()
    report["total"] = df.height
    bad = df["lat"].is_null() | df["lon"].is_null()
    report["no_position"] = int(bad.sum())
    df = df.filter(~bad)

    infos = pattern_infos(net)
    dv, versions, n_fb = date_versions(net, df["service_date"].unique().to_list(), infos)
    df = df.join(dv.select("service_date", "_ver"), on="service_date", how="left")
    geo = dict(zip(net.patterns["pattern_uid"].to_list(), net.patterns["geometry"].to_list()))
    allpts = [np.asarray(g, dtype=np.float64).reshape(-1, 2) for g in geo.values() if g]
    frame = LocalFrame.around(*allpts) if allpts else LocalFrame(4.35, 50.85)
    shapes: dict[str, _Shape] = {}
    # trip_id -> pattern (restricted to the patterns of the row's version)
    trips = net.trips if getattr(net, "trips", None) is not None else None
    parts = []
    for vi, V in enumerate(versions):
        x = df.filter(pl.col("_ver") == vi)
        if not x.height:
            continue
        cands = V.mains + V.others
        for P in cands:
            if P.uid not in shapes:
                shapes[P.uid] = _Shape(P, geo[P.uid], frame)
        parts.append(_place_version(x, V, cands, shapes, frame, trips, infos, params, window, at_stop_m, report))
    if parts:
        out = _start_layover(pl.concat(parts, how="vertical_relaxed"), params.layover_m).sort("obs_row")
        out = out.drop("direction_id").rename({"direction_id_": "direction_id"})
    else:
        out = df.head(0)
    out = finish(out, ["direction_id", "obs_row", "term_s_m"] + extras)
    report["counted"] = int(out["count"].sum()) if out.height else 0
    report["layover"] = int((~out["count"]).sum()) if out.height else 0
    report["gtfs_date_fallback_days"] = n_fb
    return Placed(frame=out, report={k: int(v) for k, v in report.items()})


def _place_version(x: pl.DataFrame, V, cands: list, shapes: dict, frame, trips, infos, params: Params, window: int,
                   at_stop_m: float, report: Counter) -> pl.DataFrame:
    """The fixes ``x`` of one network version, matched onto its patterns ``cands`` (mains first)."""
    S, Dd, B, multi = _project_all(cands, shapes, frame.xy(x.select("lon", "lat").to_numpy()), float(params.max_lateral_m))
    has = np.isfinite(Dd)
    choice, how, short_of = _by_trip(x, trips, V, cands)                       # 1. trip_id
    dirs = np.array([P.direction for P in cands])
    xd = x["direction_id"].to_numpy()
    xd = np.where(x["direction_id"].is_null().to_numpy(), -1, xd).astype(np.int64)
    allowed = has & ((xd[None, :] < 0) | (dirs[:, None] == xd[None, :]))     # 2. direction_id restricts
    t = x["ts"].dt.epoch("us").to_numpy() / 1e6
    score = _continuity(S, x, t, window)                                      # 3. continuity per vehicle
    _choose(choice, how, allowed, has, score, B, x["bearing"].cast(pl.Float64).to_numpy(), Dd, dirs, len(V.mains))
    report["off_pattern"] += int(np.isin(how, ["off", "off_direction"]).sum())
    report["ambiguous"] += int((how == "ambiguous").sum())
    for k in ("trip", "single", "continuity", "bearing"):
        report[f"by_{k}"] += int((how == k).sum())
    rows = np.flatnonzero(choice >= 0)
    ch = choice[rows]
    s = S[ch, rows]
    err = Dd[ch, rows]
    # loops: several passes -> the one following the vehicle's previous position
    if multi:
        s = _resolve_passes(rows, ch, s, err, multi, x, t)
    # end zone: with a known trip / direction a fix is on its trip until the last stop (as the linear feed,
    # which reports the stop once reached); an inferred direction is not trusted within layover_m of the end
    # (a vehicle standing there fits both directions)
    known = np.isin(how[rows], ["trip", "single"]) & (xd[rows] >= 0) | (how[rows] == "trip")
    cols = _positions(cands, V, infos, ch, s, short_of[rows], known, params.layover_m, at_stop_m)
    return x[rows].with_columns(
        pl.Series("pattern_uid", cols["uid"].tolist(), dtype=pl.Utf8),
        pl.Series("link_idx", cols["li"], dtype=pl.Int16),
        pl.Series("link_key", cols["key"].tolist(), dtype=pl.Utf8),
        pl.Series("pos_m", cols["pos"], dtype=pl.Float32),
        pl.Series("s_m", s, dtype=pl.Float32),
        pl.Series("match_err_m", err, dtype=pl.Float32),
        pl.Series("count", cols["cnt"]),
        pl.Series("_srel", cols["srel"]),
        pl.Series("flags", cols["flags"], dtype=pl.UInt16),
        pl.Series("direction_id_", cols["pdir"], dtype=pl.Int8),
        pl.Series("term_s_m", cols["term"], dtype=pl.Float64, nan_to_null=True),
        pl.coalesce("vehicle_id", "trip_id").alias("track_id"),
    )


def _project_all(cands: list, shapes: dict, pts: np.ndarray, max_d: float):
    """(C, n) nearest pass of each fix on each candidate: s, lateral distance (inf: none), shape bearing; and
    per candidate with several passes for some fix, those fixes' passes (point idx, s, d)."""
    C, n = len(cands), len(pts)
    S, Dd, B = np.full((C, n), np.nan), np.full((C, n), np.inf), np.full((C, n), np.nan)
    multi: dict[int, tuple] = {}
    for c, P in enumerate(cands):
        i, s, d, b = shapes[P.uid].project(pts, max_d)
        # nearest pass first; several passes are resolved by continuity later
        o = np.lexsort((d, i))
        i, s, d, b = i[o], s[o], d[o], b[o]
        f = np.ones(len(i), bool)
        f[1:] = i[1:] != i[:-1]
        S[c, i[f]], Dd[c, i[f]], B[c, i[f]] = s[f], d[f], b[f]
        if (~f).any():
            m = np.isin(i, np.unique(i[~f]))
            multi[c] = (i[m], s[m], d[m])
    return S, Dd, B, multi


def _by_trip(x: pl.DataFrame, trips, V, cands: list):
    """Rule 1: fixes whose trip_id is a trip of the version: choice (candidate index, -1 if none), how ("trip"),
    and the short working ridden on its parent main (short_of, its uid)."""
    n = x.height
    choice = np.full(n, -1)
    how = np.full(n, "", dtype=object)
    short_of = np.full(n, None, dtype=object)
    uid_idx = {P.uid: c for c, P in enumerate(cands)}
    if trips is None or x["trip_id"].null_count() >= n:
        return choice, how, short_of
    tp = trips.select("trip_id", "pattern_uid").filter(pl.col("pattern_uid").is_in(list(V.kinds)))
    tj = x.select("trip_id").with_row_index("_i").join(tp, on="trip_id", how="inner").unique("_i", keep="first")
    for r, uid in zip(tj["_i"].to_list(), tj["pattern_uid"].to_list()):
        kind, parent, _ = V.kinds.get(uid, (None, None, None))
        tgt = parent if kind == "short" and parent in uid_idx else uid
        if tgt in uid_idx:
            choice[r] = uid_idx[tgt]
            how[r] = "trip"
            if tgt != uid:
                short_of[r] = uid
    return choice, how, short_of


def _continuity(S: np.ndarray, x: pl.DataFrame, t: np.ndarray, window: int) -> np.ndarray:
    """(C, n) net progress (m) along each candidate over +-window fixes of the same vehicle run (NaN: no
    vehicle ids); a run ends at another vehicle or a gap over 300 s."""
    C, n = S.shape
    score = np.full((C, n), np.nan)
    if not any(v is not None for v in x["vehicle_id"].to_list()):
        return score
    vid = x.select(pl.col("vehicle_id").rank("dense")).to_series().fill_null(0).to_numpy()
    order = np.lexsort((t, vid))
    vs, ts_ = vid[order], t[order]
    same = np.concatenate([[False], (vs[1:] == vs[:-1]) & (vs[1:] > 0) & (np.diff(ts_) < 300)])
    # the window, clamped inside the run
    seg = np.cumsum(~same)
    idx = np.arange(n)
    starts = np.flatnonzero(~same)
    start_of = np.zeros(n, np.int64)
    start_of[starts] = starts
    start_of = np.maximum.accumulate(start_of)
    end_of = np.r_[starts[1:] - 1, n - 1][seg - 1]
    lo = np.maximum(np.maximum(idx - window, 0), start_of)
    hi = np.minimum(np.minimum(idx + window, n - 1), end_of)
    for c in range(C):
        so = S[c, order]
        step = np.where(same, np.clip(np.nan_to_num(np.diff(so, prepend=np.nan), nan=0.0), -MAX_STEP_M, MAX_STEP_M), 0.0)
        cs = np.concatenate([[0.0], np.cumsum(step)])
        score[c, order] = cs[hi + 1] - cs[lo + 1]
    return score


def _choose(choice, how, allowed, has, score, B, brg, Dd, dirs, n_mains: int) -> None:
    """Rules 3-5 for the fixes not placed by trip_id (in place): continuity, bearing, single candidate; several
    winners of one direction: lowest lateral error, a main within MAIN_PREF_M preferred."""
    for r in range(len(choice)):
        if choice[r] >= 0:
            continue
        ok = np.flatnonzero(allowed[:, r])
        if not len(ok):
            how[r] = "off" if not has[:, r].any() else "off_direction"
            continue
        if len(ok) == 1:
            choice[r], how[r] = ok[0], "single"
            continue
        sc = score[ok, r]
        pick = None
        if np.isfinite(sc).any():
            best = np.nanmax(sc)
            good = ok[(sc >= best - MARGIN_M) & (sc >= MIN_PROGRESS_M)]
            if len(good):
                pick, how_r = good, "continuity"
        if pick is None and np.isfinite(brg[r]):
            good = ok[_angle(B[ok, r], brg[r]) < BEARING_TOL]
            if len(good):
                pick, how_r = good, "bearing"
        if pick is None:
            how[r] = "ambiguous"
            continue
        if len(pick) > 1:
            # both directions look right (standing, noise): cannot tell
            if len({int(dirs[c]) for c in pick}) > 1:
                how[r] = "ambiguous"
                continue
            dmin = Dd[pick, r].min()
            mains = [c for c in pick if c < n_mains and Dd[c, r] <= dmin + MAIN_PREF_M]
            pick = [mains[0]] if mains else [pick[np.argmin(Dd[pick, r])]]
        choice[r], how[r] = pick[0], how_r


def _positions(cands: list, V, infos, ch, s, sw_all, known, layover_m: float, at_stop_m: float) -> dict:
    """Per placed fix (candidate ch, along-pattern s, updated in place when snapped to a stop): pattern, link,
    position, flags, count, s from the trip's start (srel) and the terminus reached (term)."""
    k = len(ch)
    out = {"flags": np.zeros(k, np.int64), "srel": np.zeros(k), "cnt": np.ones(k, bool), "uid": np.empty(k, dtype=object),
           "li": np.zeros(k, np.int16), "pos": np.zeros(k), "key": np.empty(k, dtype=object), "pdir": np.zeros(k, np.int8),
           "term": np.full(k, np.nan)}
    for c, P in enumerate(cands):
        m = ch == c
        if not m.any():
            continue
        out["uid"][m] = P.uid
        out["pdir"][m] = P.direction
        # within at_stop_m after a stop: standing at it (GPS noise), end of the link arriving there, as the
        # linear rule (feed distance < 10 m after the stop = at the stop)
        k_ = np.searchsorted(P.s, s[m], side="right") - 1
        at = P.s[np.clip(k_, 0, len(P.s) - 1)]
        snap = (k_ >= 1) & (k_ < len(P.s) - 1) & (s[m] - at < at_stop_m)
        s[m] = np.where(snap, at, s[m])
        a, b = link_of_s(s[m], P.s)
        out["li"][m], out["pos"][m] = a, b
        out["key"][m] = np.array(P.keys, dtype=object)[a]
        sw = sw_all[m]
        st, en = np.full(m.sum(), 0.0), np.full(m.sum(), P.length)
        for j, su in enumerate(sw):
            if su is not None:
                off = V.kinds[su][2]
                st[j] = P.s[off]
                en[j] = P.s[min(off + len(infos[su].s) - 1, len(P.s) - 1)]
        is_sw = np.array([v is not None for v in sw])
        end_m = np.where(known[m], 0.5, layover_m)
        lay = (s[m] - st < layover_m) | (s[m] > en - end_m)
        out["flags"][m] = np.where(lay, Flag.LAYOVER, 0) | np.where(is_sw, Flag.SHORT_WORKING, 0) | np.where(snap, Flag.AT_STOP, 0)
        out["cnt"][m] = ~lay
        out["srel"][m] = s[m] - st
        # a known trip on the link leading to its terminus: a per-trip feed stops once the trip has arrived, so
        # the track is taken to reach that terminus (term_s_m, as the linear locator)
        ei = np.searchsorted(P.s, en, side="left")
        last_link = s[m] > P.s[np.clip(ei - 1, 0, len(P.s) - 1)]
        out["term"][m] = np.where(known[m] & last_link, en, np.nan)
    return out


def _start_layover(df: pl.DataFrame, layover_m: float) -> pl.DataFrame:
    """A vehicle waiting at its first stop jitters around it (GPS noise): every fix of a run before it
    clearly leaves (3 x layover_m) that is not past its last fix under layover_m is layover too."""
    tr = pl.coalesce("vehicle_id", "trip_id")
    d = df.with_columns(tr.alias("_tr")).sort("_tr", "ts", maintain_order=True)
    run = ((pl.col("_tr") != pl.col("_tr").shift(1)) | (pl.col("pattern_uid") != pl.col("pattern_uid").shift(1))
           | ((pl.col("ts") - pl.col("ts").shift(1)).dt.total_seconds() > 600)).fill_null(True).cum_sum()
    d = d.with_columns(run.alias("_run"))
    dep = pl.when(pl.col("_srel") >= 3 * layover_m).then(pl.col("ts")).min().over("_run")
    d = d.with_columns(dep.alias("_dep"))
    last = (pl.when((pl.col("_srel") < layover_m) & ((pl.col("ts") < pl.col("_dep")) | pl.col("_dep").is_null()))
            .then(pl.col("ts")).max().over("_run"))
    lay = (pl.col("_tr").is_not_null() & (pl.col("_srel") < 3 * layover_m) & (pl.col("ts") <= last)
           & ((pl.col("ts") < pl.col("_dep")) | pl.col("_dep").is_null()))
    d = d.with_columns(
        pl.when(lay).then(False).otherwise(pl.col("count")).alias("count"),
        pl.when(lay).then(pl.col("flags") | Flag.LAYOVER).otherwise(pl.col("flags")).cast(pl.UInt16).alias("flags"))
    return d.drop("_tr", "_run", "_dep", "_srel")


def _resolve_passes(rows, ch, s, err, multi, x, t):
    """Fixes with several passes on their pattern: per track, the first pass at or after the
    previous position (minus 50 m of noise); none -> the earliest pass (new run)."""
    s = s.copy()
    cand: dict[tuple[int, int], list] = {}
    for c, (i, ss, dd) in multi.items():
        for a, b, d in zip(i.tolist(), ss.tolist(), dd.tolist()):
            cand.setdefault((c, a), []).append((b, d))
    if not cand:
        return s
    track = x["vehicle_id"].fill_null(x["trip_id"]).to_list()
    pos_in = {r: k for k, r in enumerate(rows.tolist())}
    by_track: dict = {}
    for k, r in enumerate(rows.tolist()):
        by_track.setdefault((track[r], int(ch[k])), []).append(r)
    for (tr, c), rs in by_track.items():
        if tr is None or not any((c, r) in cand for r in rs):
            continue
        rs.sort(key=lambda r: t[r])
        prev, prev_t = None, None
        for r in rs:
            k = pos_in[r]
            opts = sorted(cand.get((c, r), [(s[k], err[k])]))
            if prev is None or t[r] - prev_t > 600:
                pick = opts[0][0]
            else:
                ahead = [o for o in opts if o[0] >= prev - 50]
                pick = ahead[0][0] if ahead else opts[0][0]
            s[k] = pick
            prev, prev_t = pick, t[r]
    return s
