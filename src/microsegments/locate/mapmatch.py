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

from .. import schema
from ..config import Params
from ..network.geometry import LocalFrame, cumlen
from ..schema import Flag
from .common import PatInfo, Placed, date_versions, ensure_local_time, finish, link_of_s, pattern_infos

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
    x = np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)
    return x


def place_latlon(obs, net, params: Params | None = None, *, tz: str = "Europe/Brussels",
                 service_day_start: str = "04:00", window: int = WINDOW, at_stop_m: float = AT_STOP_M) -> Placed:
    """Map-match lat/lon observations onto the network's patterns. See the module docstring."""
    params = params or Params()
    if isinstance(obs, pl.LazyFrame):
        obs = obs.collect()
    df = schema.conform(obs, schema.OBSERVATION, schema.OBSERVATION_REQUIRED)
    extras = [c for c in df.columns if c not in schema.OBSERVATION and c not in ("service_date", "hour", "dow")]
    df = ensure_local_time(df, tz, service_day_start).with_row_index("obs_row")
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

    # trip_id -> pattern (restricted to the patterns of the row's version below)
    trips = net.trips if getattr(net, "trips", None) is not None else None
    max_d = float(params.max_lateral_m)
    parts = []
    for vi, V in enumerate(versions):
        x = df.filter(pl.col("_ver") == vi)
        if not x.height:
            continue
        cands = V.mains + V.others
        for P in cands:
            if P.uid not in shapes:
                shapes[P.uid] = _Shape(P, geo[P.uid], frame)
        n, C = x.height, len(cands)
        pts = frame.xy(x.select("lon", "lat").to_numpy())
        S = np.full((C, n), np.nan)
        Dd = np.full((C, n), np.inf)
        B = np.full((C, n), np.nan)
        multi: dict[int, tuple] = {}
        for c, P in enumerate(cands):
            i, s, d, b = shapes[P.uid].project(pts, max_d)
            # nearest pass first; several passes are resolved by continuity below
            o = np.lexsort((d, i))
            i, s, d, b = i[o], s[o], d[o], b[o]
            f = np.ones(len(i), bool)
            f[1:] = i[1:] != i[:-1]
            S[c, i[f]], Dd[c, i[f]], B[c, i[f]] = s[f], d[f], b[f]
            if (~f).any():
                rep = np.unique(i[~f])
                m = np.isin(i, rep)
                multi[c] = (i[m], s[m], d[m])
        has = np.isfinite(Dd)

        # ---- 1. trip_id
        choice = np.full(n, -1)
        how = np.full(n, "", dtype=object)
        uid_idx = {P.uid: c for c, P in enumerate(cands)}
        short_of = np.full(n, None, dtype=object)
        if trips is not None and x["trip_id"].null_count() < n:
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
        # ---- 2. direction_id restricts the candidates
        dirs = np.array([P.direction for P in cands])
        xd = x["direction_id"].to_numpy()
        xd = np.where(x["direction_id"].is_null().to_numpy(), -1, xd).astype(np.int64)
        allowed = has & ((xd[None, :] < 0) | (dirs[:, None] == xd[None, :]))
        # ---- 3. continuity per vehicle
        veh = x["vehicle_id"].to_list()
        t = x["ts"].dt.epoch("us").to_numpy() / 1e6
        score = np.full((C, n), np.nan)
        if any(v is not None for v in veh):
            vid = x.select(pl.col("vehicle_id").rank("dense")).to_series().fill_null(0).to_numpy()
            order = np.lexsort((t, vid))
            vs, ts_ = vid[order], t[order]
            same = np.concatenate([[False], (vs[1:] == vs[:-1]) & (vs[1:] > 0) & (np.diff(ts_) < 300)])
            for c in range(C):
                so = S[c, order]
                step = np.where(same, np.clip(np.nan_to_num(np.diff(so, prepend=np.nan), nan=0.0),
                                              -MAX_STEP_M, MAX_STEP_M), 0.0)
                step[~same] = 0.0
                cs = np.concatenate([[0.0], np.cumsum(step)])
                # segment id so the window never crosses another vehicle / a long gap
                seg = np.cumsum(~same)
                idx = np.arange(n)
                lo = np.maximum(idx - window, 0)
                hi = np.minimum(idx + window, n - 1)
                # clamp the window inside the run of the same segment
                start_of = np.zeros(n, np.int64)
                starts = np.flatnonzero(~same)
                start_of[starts] = starts
                start_of = np.maximum.accumulate(start_of)
                ends = np.r_[starts[1:] - 1, n - 1]
                end_of = ends[seg - 1]
                lo = np.maximum(lo, start_of)
                hi = np.minimum(hi, end_of)
                sc = cs[hi + 1] - cs[lo + 1]
                score[c, order] = sc
        brg = x["bearing"].cast(pl.Float64).to_numpy()
        for r in range(n):
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
                ang = _angle(B[ok, r], brg[r])
                good = ok[ang < BEARING_TOL]
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
                mains = [c for c in pick if c < len(V.mains) and Dd[c, r] <= dmin + MAIN_PREF_M]
                pick = [mains[0]] if mains else [pick[np.argmin(Dd[pick, r])]]
            choice[r], how[r] = pick[0], how_r

        placed = choice >= 0
        report["off_pattern"] += int(np.isin(how, ["off", "off_direction"]).sum())
        report["ambiguous"] += int((how == "ambiguous").sum())
        for k in ("trip", "single", "continuity", "bearing"):
            report[f"by_{k}"] += int((how == k).sum())
        rows = np.flatnonzero(placed)
        ch = choice[rows]
        s = S[ch, rows]
        err = Dd[ch, rows]
        # ---- loops: several passes -> the one following the vehicle's previous position
        if multi:
            s = _resolve_passes(rows, ch, s, err, multi, x, t)
        flags = np.zeros(len(rows), np.int64)
        srel = np.zeros(len(rows))
        cnt = np.ones(len(rows), bool)
        uid = np.empty(len(rows), dtype=object)
        li = np.zeros(len(rows), np.int16)
        pos = np.zeros(len(rows))
        key = np.empty(len(rows), dtype=object)
        pdir = np.zeros(len(rows), np.int8)
        term = np.full(len(rows), np.nan)
        for c, P in enumerate(cands):
            m = ch == c
            if not m.any():
                continue
            uid[m] = P.uid
            pdir[m] = P.direction
            # within at_stop_m after a stop: standing at it (GPS noise), end of the link arriving there,
            # as the linear rule (feed distance < 10 m after the stop = at the stop)
            k_ = np.searchsorted(P.s, s[m], side="right") - 1
            snap = (k_ >= 1) & (k_ < len(P.s) - 1) & (s[m] - P.s[np.clip(k_, 0, len(P.s) - 1)] < at_stop_m)
            sm = np.where(snap, P.s[np.clip(k_, 0, len(P.s) - 1)], s[m])
            s[m] = sm
            a, b = link_of_s(sm, P.s)
            li[m], pos[m] = a, b
            key[m] = np.array(P.keys, dtype=object)[a]
            start, end = 0.0, P.length
            st = np.full(m.sum(), start)
            en = np.full(m.sum(), end)
            sw = short_of[rows[m]]
            for k_, su in enumerate(sw):
                if su is not None:
                    _, parent, off = V.kinds[su]
                    Ps = infos[su]
                    st[k_] = P.s[off]
                    en[k_] = P.s[min(off + len(Ps.s) - 1, len(P.s) - 1)]
            is_sw = np.array([v is not None for v in sw])
            # end zone: with a known trip / direction a fix is on its trip until the last stop (as the
            # linear feed, which reports the stop once reached); an inferred direction is not trusted
            # within layover_m of the end (a vehicle standing there fits both directions)
            known = np.isin(how[rows[m]], ["trip", "single"]) & (xd[rows[m]] >= 0) | (how[rows[m]] == "trip")
            end_m = np.where(known, 0.5, params.layover_m)
            lay = (s[m] - st < params.layover_m) | (s[m] > en - end_m)
            f = np.where(lay, Flag.LAYOVER, 0) | np.where(is_sw, Flag.SHORT_WORKING, 0)
            f |= np.where(snap, Flag.AT_STOP, 0)
            flags[m] = f
            cnt[m] = ~lay
            srel[m] = s[m] - st
            # a known trip on the link leading to its terminus: a per-trip feed stops once the trip has
            # arrived, so the track is taken to reach that terminus (term_s_m, as the linear locator)
            ei = np.searchsorted(P.s, en, side="left")
            last_link = s[m] > P.s[np.clip(ei - 1, 0, len(P.s) - 1)]
            term[m] = np.where(known & last_link, en, np.nan)
        xr = x[rows]
        part = xr.with_columns(
            pl.Series("pattern_uid", uid.tolist(), dtype=pl.Utf8),
            pl.Series("link_idx", li, dtype=pl.Int16),
            pl.Series("link_key", key.tolist(), dtype=pl.Utf8),
            pl.Series("pos_m", pos, dtype=pl.Float32),
            pl.Series("s_m", s, dtype=pl.Float32),
            pl.Series("match_err_m", err, dtype=pl.Float32),
            pl.Series("count", cnt),
            pl.Series("_srel", srel),
            pl.Series("flags", flags, dtype=pl.UInt16),
            pl.Series("direction_id_", pdir, dtype=pl.Int8),
            pl.Series("term_s_m", term, dtype=pl.Float64, nan_to_null=True),
            pl.coalesce("vehicle_id", "trip_id").alias("track_id"),
        )
        parts.append(part)
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
