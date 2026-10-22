"""Dense day x hour x segment arrays of one direction (``DirData``), built from the cube and the
passages, and its GTFS versions (main patterns grouped by link sequence; the display version)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

# same key order in both (band columns follow it); new bands go at the end so the codes stay stable
BANDS: dict[str, tuple[int, int]] = {"am": (7, 10), "pm": (16, 19), "day": (6, 21), "evening": (20, 24), "midday": (10, 16)}
BAND_HOUR: dict[str, int] = {"am": -1, "pm": -2, "day": -3, "evening": -4, "midday": -5}
DISPLAY_MIN_DAYS, DISPLAY_MIN_SHARE = 5, 0.2   # a version shown by default runs on >= max(5 days, 20 % of the days)


@dataclass
class DirData:
    """Dense day x hour x segment arrays of one direction (internal, used by hotspots / tune)."""
    direction_id: int
    display_uid: str
    days: np.ndarray                 # datetime64[D] (D,)
    dow: np.ndarray                  # (D,) 0 = Monday
    main_uid: np.ndarray             # (D,) main pattern of each day
    hours: np.ndarray                # (H,) internal hours
    cols: np.ndarray                 # (Hx,) column labels: hours, then band codes -1..-5
    seg_keys: np.ndarray             # (K,)
    seg_link: np.ndarray             # (K,) index into link_keys
    seg_len: np.ndarray              # (K,)
    seg_zone: np.ndarray             # (K,)
    link_keys: np.ndarray            # (Lk,)
    pattern_segs: dict[str, np.ndarray]    # uid -> K indices in seg_idx order
    pattern_seg_idx: dict[str, np.ndarray]
    pattern_x0: dict[str, np.ndarray]
    pattern_links: dict[str, np.ndarray]   # uid -> link indices in link order
    V: np.ndarray                    # (D,K) bool
    LV: np.ndarray                   # (D,Lk) bool
    E: np.ndarray                    # (D,H) bool, day-hour included
    C: np.ndarray                    # (D,H) coverage, 0 where excluded
    N: np.ndarray                    # (D,H,K)
    P: np.ndarray                    # (D,H,Lk)
    _x: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def col_index(self, labels) -> np.ndarray:
        pos = {int(c): i for i, c in enumerate(self.cols)}
        return np.array([pos[int(c)] for c in labels if int(c) in pos], dtype=np.int64)

    @property
    def hour_cols(self) -> np.ndarray:
        return np.flatnonzero(self.cols >= 0)

    def stacked(self) -> dict[str, np.ndarray]:
        """(D, Hx, K) float32 arrays NM, PM, CM, M: masked numerator, passages, coverage, cell count;
        band columns are sums over their hours (M: 1 if any hour of the band is included)."""
        if self._x:
            return self._x
        M = self.E[:, :, None] & self.V[:, None, :]
        NM = np.where(M, self.N, 0.0)
        PM = np.where(M, self.P[:, :, self.seg_link], 0.0)
        CM = np.where(M, self.C[:, :, None], 0.0)
        bands = []
        for b, (a, z) in BANDS.items():
            idx = np.flatnonzero((self.hours >= a) & (self.hours < z))
            bands.append(idx)
        def ext(A, red):
            return np.concatenate([A] + [red(A[:, i, :])[:, None, :] for i in bands], axis=1).astype(np.float32)
        self._x = {
            "NM": ext(NM, lambda a: a.sum(1)),
            "PM": ext(PM, lambda a: a.sum(1)),
            "CM": ext(CM, lambda a: a.sum(1)),
            "M": ext(M.astype(np.float32), lambda a: a.max(1) if a.shape[1] else a.sum(1)),
        }
        return self._x

    def take(self, idx: np.ndarray) -> "DirData":
        idx = np.asarray(idx)
        return replace(self, days=self.days[idx], dow=self.dow[idx], main_uid=self.main_uid[idx],
                       V=self.V[idx], LV=self.LV[idx], E=self.E[idx], C=self.C[idx], N=self.N[idx],
                       P=self.P[idx], _x={})


def _build_dir(dirid, md, seg, cube, passages, passage_col, covA, ex, day_pos, hours, cols, wA=None) -> DirData | None:
    """The DirData of direction ``dirid`` from its main pattern per day ``md`` (service_date, pattern_uid);
    ``covA``, ``ex``, ``wA``: (all days, 28 h) coverage, exclusion and report weight, rows ``day_pos``."""
    days_l = md["service_date"].to_list()
    main_uid = np.array(md["pattern_uid"].to_list(), dtype=object)
    if not days_l:
        return None
    uids = _display_order(seg, days_l, main_uid)
    s = seg.filter(pl.col("pattern_uid").is_in(uids.tolist()))
    if s.height == 0:
        return None
    uk, seg_keys, link_keys, pat, inc, linc = _universe(s, uids)
    upos = {u: i for i, u in enumerate(uids)}
    mi = np.array([upos[u] for u in main_uid])
    gi = np.array([day_pos[d] for d in days_l])
    E = ~ex[gi][:, hours]
    C = np.where(E, np.nan_to_num(covA[gi][:, hours]), 0.0)
    N, P = _dense(cube, passages, passage_col, days_l, hours, seg_keys, link_keys)
    if wA is not None:
        N *= wA[gi][:, hours][:, :, None]
    lpos = {k: i for i, k in enumerate(link_keys)}
    days = np.array(days_l, dtype="datetime64[D]")
    dow = ((days.astype("int64") + 3) % 7).astype(np.int64)   # 1970-01-01 was a Thursday
    return DirData(direction_id=dirid, display_uid=str(uids[0]), days=days, dow=dow, main_uid=main_uid, hours=hours,
                   cols=cols, seg_keys=seg_keys, seg_link=np.array([lpos[k] for k in uk["link_key"].to_list()], dtype=np.int64),
                   seg_len=uk["len_m"].to_numpy().astype(float), seg_zone=np.array(uk["zone"].to_list(), dtype=object),
                   link_keys=link_keys, **pat, V=inc[mi], LV=linc[mi], E=E, C=C, N=N, P=P)


def _display_order(seg: pl.DataFrame, days_l: list, main_uid: np.ndarray) -> np.ndarray:
    """The main patterns, display (reference version) first: among the versions (link_key sequences; patterns
    differing by shape edits only are pooled) running on >= max(5 days, 20 % of the days), the longest, then the
    one with most segments, then the most recent; within it the pattern with the most days. None that frequent:
    the most frequent one."""
    uids, counts = np.unique(main_uid, return_counts=True)
    sg = seg.filter(pl.col("pattern_uid").is_in(uids.tolist()))
    seq = {u: tuple(g.sort("link_idx")["link_key"].unique(maintain_order=True).to_list()) for (u,), g in sg.group_by(["pattern_uid"])}
    size = {u: (float(g["len_m"].sum()), g.height) for (u,), g in sg.group_by(["pattern_uid"])}
    last_of = {u: max(d for d, x in zip(days_l, main_uid) if x == u) for u in uids}
    key = lambda u: seq.get(u, (u,))  # noqa: E731
    last: dict[tuple, dt.date] = {}
    gdays: dict[tuple, int] = {}
    for u, c in zip(uids, counts):
        last[key(u)] = max(last.get(key(u), last_of[u]), last_of[u])
        gdays[key(u)] = gdays.get(key(u), 0) + int(c)
    need = max(DISPLAY_MIN_DAYS, DISPLAY_MIN_SHARE * len(days_l))
    ln = lambda i: -round(size.get(uids[i], (0.0, 0))[0], 1)  # noqa: E731
    recent = lambda i: -last[key(uids[i])].toordinal()  # noqa: E731
    if any(gdays[key(u)] >= need for u in uids):
        order = sorted(range(len(uids)), key=lambda i: (gdays[key(uids[i])] < need, ln(i), -size.get(uids[i], (0.0, 0))[1],
                                                        recent(i), -counts[i]))
    else:
        order = sorted(range(len(uids)), key=lambda i: (-gdays[key(uids[i])], ln(i), recent(i), -counts[i]))
    return uids[order]


def _universe(s: pl.DataFrame, uids: np.ndarray):
    """Segment and link keys of the patterns ``uids`` (display first; first occurrence wins), the unique-key rows
    ``uk``, each pattern's segments / links (DirData.pattern_* fields) and (pattern, key) membership."""
    rank = {u: i for i, u in enumerate(uids)}
    s = s.with_columns(pl.col("pattern_uid").replace_strict(rank, return_dtype=pl.Int32).alias("_r")).sort("_r", "seg_idx")
    uk = s.unique("seg_key", keep="first", maintain_order=True)
    seg_keys = np.array(uk["seg_key"].to_list(), dtype=object)
    kpos = {k: i for i, k in enumerate(seg_keys)}
    link_keys = np.array(s.unique("link_key", keep="first", maintain_order=True)["link_key"].to_list(), dtype=object)
    lpos = {k: i for i, k in enumerate(link_keys)}
    pat = {"pattern_segs": {}, "pattern_seg_idx": {}, "pattern_x0": {}, "pattern_links": {}}
    inc = np.zeros((len(uids), len(seg_keys)), bool)
    linc = np.zeros((len(uids), len(link_keys)), bool)
    for i, u in enumerate(uids):
        su = s.filter(pl.col("pattern_uid") == u).sort("seg_idx")
        ks = np.array([kpos[k] for k in su["seg_key"].to_list()], dtype=np.int64)
        lk = su.sort("link_idx").unique("link_idx", keep="first", maintain_order=True)["link_key"].to_list()
        pat["pattern_segs"][u], pat["pattern_seg_idx"][u], pat["pattern_x0"][u] = ks, su["seg_idx"].to_numpy(), su["x0_m"].to_numpy()
        pat["pattern_links"][u] = np.array([lpos[k] for k in lk], dtype=np.int64)
        inc[i, ks] = True
        linc[i, pat["pattern_links"][u]] = True
    return uk, seg_keys, link_keys, pat, inc, linc


def _dense(cube, passages, passage_col, days_l, hours, seg_keys, link_keys) -> tuple[np.ndarray, np.ndarray]:
    """N (D, H, K) observations of the cube and P (D, H, Lk) passages, on the given days, hours and keys."""
    D, H, Lk = len(days_l), len(hours), len(link_keys)
    days_df = pl.DataFrame({"service_date": days_l, "_d": np.arange(D)}, schema={"service_date": pl.Date, "_d": pl.Int64})
    hours_df = pl.DataFrame({"hour": hours, "_h": np.arange(H)}, schema={"hour": pl.Int8, "_h": pl.Int64})

    def counts(frame: pl.LazyFrame, key: str, keys, val: pl.Expr) -> np.ndarray:
        kdf = pl.DataFrame({key: keys.tolist(), "_k": np.arange(len(keys))}, schema={key: pl.Utf8, "_k": pl.Int64})
        c = (frame.select("service_date", pl.col("hour").cast(pl.Int8), key, val.alias("_v"))
             .join(days_df.lazy(), on="service_date").join(hours_df.lazy(), on="hour").join(kdf.lazy(), on=key)
             .select("_d", "_h", "_k", "_v").collect())
        flat = (c["_d"].to_numpy() * H + c["_h"].to_numpy()) * len(keys) + c["_k"].to_numpy()
        return np.bincount(flat, weights=c["_v"].to_numpy().astype(np.float64), minlength=D * H * len(keys)).reshape(D, H, len(keys))

    N = counts(cube.lazy(), "seg_key", seg_keys, pl.col("obs"))
    P = np.zeros((D, H, Lk))
    if passages is not None and passages.height:
        col = passage_col if passage_col in passages.columns else "n"
        P = counts(passages.lazy(), "link_key", link_keys, pl.col(col).fill_null(0))
    return N, P


def version_groups(dd: DirData) -> dict[str, list[str]]:
    """Main patterns of a direction grouped by link_key sequence: {representative uid: uids}, the
    display pattern first. Patterns that differ only by small shape edits (same link keys, hence the
    same segments) are one version; the representative is the display pattern or the one with most days."""
    seqs: dict[tuple, list[str]] = {}
    for u in dd.pattern_segs:
        if (dd.main_uid == u).any():
            seqs.setdefault(tuple(dd.link_keys[dd.pattern_links[u]].tolist()), []).append(u)
    out: dict[str, list[str]] = {}
    groups = sorted(seqs.values(), key=lambda us: (dd.display_uid not in us, -sum(int((dd.main_uid == u).sum()) for u in us)))
    for us in groups:
        rep = dd.display_uid if dd.display_uid in us else max(us, key=lambda u: int((dd.main_uid == u).sum()))
        out[rep] = us
    return out


def _versions(dirs: dict[int, DirData]) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, val = [], []
    for k, dd in dirs.items():
        disp = set(dd.link_keys[dd.pattern_links[dd.display_uid]].tolist())
        for u, us in version_groups(dd).items():
            m = np.isin(dd.main_uid, np.array(us, dtype=object))
            links = set(dd.link_keys[dd.pattern_links[u]].tolist())
            rows.append({"direction_id": k, "pattern_uid": u, "first": dd.days[m].min().item(), "last": dd.days[m].max().item(),
                         "n_days": int(m.sum()), "display": u == dd.display_uid,
                         "links_added": sorted(links - disp), "links_removed": sorted(disp - links)})
        val.extend((k, d.item(), u) for d, u in zip(dd.days, dd.main_uid))
    vs = {"direction_id": pl.Int8, "pattern_uid": pl.Utf8, "first": pl.Date, "last": pl.Date, "n_days": pl.UInt16,
          "display": pl.Boolean, "links_added": pl.List(pl.Utf8), "links_removed": pl.List(pl.Utf8)}
    versions = pl.DataFrame(rows, schema=vs) if rows else pl.DataFrame(schema=vs)
    validity = pl.DataFrame(val, schema={"direction_id": pl.Int8, "service_date": pl.Date, "pattern_uid": pl.Utf8}, orient="row")
    return versions, validity
