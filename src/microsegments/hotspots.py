"""Hotspots: stretches of segments where vehicles linger more than at the reference hours.

Per direction, on the display pattern, for every bin (segment x hour, plus the am / pm bands):

1. Day bootstrap (B draws, Rao-Wu rescaled, weekday-stratified, whole days resampled): numerator, passages and
   reference are recomputed from the same resampled days (``Analysis.estimate`` with day weights).
2. Candidate bin when
   * the lower bound of the percentile CI (``level``) of ``excess_per_passage * tick_s`` > ``theta_s``;
   * daily excess (day's obs/passages minus the full-sample reference) is positive on
     ≥ ``min_persistence`` of the days with passages;
   * it passes Benjamini-Hochberg at ``q`` over all tested bins. p-values are one-sided, from the
     bootstrap standard error (z = (x - θ) / se): percentile p-values cannot go below 1/(B+1), which
     BH over thousands of bins never accepts.
3. Temporal persistence: an hourly candidate counts only within a run of ≥ 2 adjacent candidate hours;
   a band candidate (am / pm) counts for all its hours.
4. Stretches: candidate segments merge along the pattern with a ``gap``-bin tolerance, and never
   across a stop / running zone border.
5. Classes (``fixed_score`` F and ``peak_ratio`` are returned as extra columns):
   * F = q20 over day hours 6-20 of the stretch occupancy per metre (obs / passage / m), divided by the
     median of the same quantity over non-candidate segments of the same zone within ±500 m (the
     "free-flow" level of the day; high when the delay never goes away during the day);
   * peak_ratio = mean summed excess over peak hours (am / pm bands) / mean over the other day hours;
   * infrastructure: F ≥ ``f_min`` and peak_ratio < ``peak_ratio``; congestion: peak_ratio ≥
     ``peak_ratio`` and F < ``f_min``; mixed otherwise.
   Note: a delay that is also present at the reference hours cancels in the excess and is not detected.

Stretches are ranked by excess observations per day (Σ over stretch hours and segments of
obs - ref * passages, divided by the number of days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .metrics import BANDS, BAND_HOUR, Analysis, DirData
from .schema import HOTSPOT, conform

PEAK_BANDS = ("am", "pm")
ALLDAY_HOURS = (6, 23)      # [start, end): window of the persistent criterion, evening included
MIN_HOTSPOT_DAYS = 5        # floor under quality.min_days: below this a day bootstrap is meaningless


@dataclass
class BinStats:
    """Per-bin statistics of one direction, on the display pattern (columns = output hours + am + pm)."""
    dd: DirData
    ks: np.ndarray               # (Kd,) K indices, display order
    seg_idx: np.ndarray          # (Kd,)
    x0: np.ndarray
    length: np.ndarray
    zone: np.ndarray
    cols: np.ndarray             # (Hc,) labels: hours, then -1 (am), -2 (pm)
    x: np.ndarray                # (Hc,Kd) excess per passage (obs / vehicle)
    est: dict                    # full-sample estimate restricted to (Hc|all hours, Kd), see _cut
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    se: np.ndarray
    p: np.ndarray
    reject: np.ndarray
    persistence: np.ndarray
    candidate: np.ndarray        # per-bin rule (CI, persistence, BH)
    draws: np.ndarray            # (B,Hc,Kd) bootstrap excess per passage
    daily: np.ndarray            # (D,Hc,Kd) daily excess (nan where no passage)
    theta: float                 # threshold in obs per passage
    # persistent lingering (criterion b), over the all-day hours ALLDAY_HOURS; None when not computed
    p_hours: np.ndarray | None = None        # (Hp,) hours of the all-day window
    p_level: float = float("nan")            # typical running level, obs / passage / m (median of running segs)
    p_x: np.ndarray | None = None            # (Kd,) all-day excess over the typical level, obs / passage
    p_xh: np.ndarray | None = None           # (Hp,Kd) same per hour
    p_draws: np.ndarray | None = None        # (B,Kd) bootstrap p_x
    p_draws_h: np.ndarray | None = None      # (B,Hp,Kd) bootstrap p_xh
    p_daily: np.ndarray | None = None        # (D,Kd) daily all-day excess (nan where no passage)
    p_ci_lo: np.ndarray | None = None
    p_ci_hi: np.ndarray | None = None
    p_reject: np.ndarray | None = None
    p_persistence: np.ndarray | None = None  # share of days with positive all-day excess
    p_hour_share: np.ndarray | None = None   # share of all-day hours with excess > p_theta
    p_candidate: np.ndarray | None = None
    p_theta: float = float("nan")
    p_obs: np.ndarray | None = None          # (Hp,Kd) full-sample obs / passages of the all-day hours
    p_pas: np.ndarray | None = None


def bootstrap_weights(dow: np.ndarray, B: int, rng: np.random.Generator, stratified: bool = True) -> np.ndarray:
    """(B, D) day weights of a Rao-Wu rescaled day bootstrap (within each weekday when ``stratified``):
    n_h - 1 days drawn with replacement in a stratum of n_h days, multiplicities scaled by
    n_h / (n_h - 1). This removes the (n_h - 1) / n_h variance shrinkage of the naive bootstrap,
    which matters with few days per weekday. A stratum with one day keeps weight 1."""
    D = len(dow)
    W = np.zeros((B, D), np.float32)
    groups = [np.flatnonzero(dow == w) for w in np.unique(dow)] if stratified else [np.arange(D)]
    rows = np.arange(B)[:, None]
    for g in groups:
        n = len(g)
        if n == 0:
            continue
        if n == 1:
            W[:, g] = 1.0
            continue
        pick = g[rng.integers(0, n, (B, n - 1))]
        np.add.at(W, (np.broadcast_to(rows, pick.shape), pick), n / (n - 1))
    return W


def _bh(p: np.ndarray, q: float) -> np.ndarray:
    flat = p.reshape(-1)
    ok = np.isfinite(flat)
    out = np.zeros(flat.shape, bool)
    pv = flat[ok]
    m = len(pv)
    if m == 0:
        return out.reshape(p.shape)
    order = np.argsort(pv)
    passed = pv[order] <= q * np.arange(1, m + 1) / m
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        sel = np.zeros(m, bool)
        sel[order[: kmax + 1]] = True
        out[np.flatnonzero(ok)[sel]] = True
    return out.reshape(p.shape)


_erfc = np.frompyfunc(math.erfc, 1, 1)


def bin_stats(an: Analysis, direction_id: int, *, B: int | None = None, theta_s: float = 2.0, q: float = 0.1,
              level: float = 0.95, min_persistence: float = 0.6, seed: int = 0, stratified: bool = True,
              chunk: int = 50, theta_persistent_s: float | None = None, min_hour_share: float = 0.75,
              persistent: bool = True) -> BinStats:
    dd = an.dirs[direction_id]
    B = an.params.bootstrap if B is None else B
    tick = an.params.tick_s
    theta = theta_s / tick
    ks = dd.pattern_segs[dd.display_uid]
    out_hours = set(an.hours)
    ci = np.array([i for i, c in enumerate(dd.cols) if (c >= 0 and int(c) in out_hours)]
                  + [int(np.flatnonzero(dd.cols == BAND_HOUR[b])[0]) for b in PEAK_BANDS])
    cols = dd.cols[ci]

    est = an.estimate(direction_id)
    x = est["excess_per_passage"][0][ci][:, ks]

    rng = np.random.default_rng(seed)
    W = bootstrap_weights(dd.dow, B, rng, stratified)
    draws = np.empty((B, len(ci), len(ks)), np.float32)
    hrs_all = dd.cols[dd.hour_cols]
    pcols = dd.hour_cols[(hrs_all >= ALLDAY_HOURS[0]) & (hrs_all < ALLDAY_HOURS[1])]
    length = dd.seg_len[ks]
    running = dd.seg_zone[ks] == "running"
    do_p = persistent and len(pcols) > 0 and running.any()
    if do_p:
        p_draws = np.empty((B, len(ks)), np.float32)
        p_draws_h = np.empty((B, len(pcols), len(ks)), np.float32)
    for a in range(0, B, chunk):
        e = an.estimate(direction_id, W[a:a + chunk])
        draws[a:a + chunk] = e["excess_per_passage"][:, ci][:, :, ks]
        if do_p:
            _, xa, xh = _persistent_level(e["obs"][:, pcols][:, :, ks], e["passages"][:, pcols][:, :, ks], length, running)
            p_draws[a:a + chunk] = xa
            p_draws_h[a:a + chunk] = xh
    alpha = 1 - level
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
            se = np.nanstd(draws, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - theta) / se
        z = np.where(se > 0, z, np.where(x > theta, np.inf, -np.inf))
        p = 0.5 * _erfc(z / math.sqrt(2)).astype(float)
    p[~np.isfinite(x)] = np.nan
    reject = _bh(p, q)

    X = dd.stacked()
    NM, PM = X["NM"][:, ci][:, :, ks], X["PM"][:, ci][:, :, ks]
    ref = est["ref"][0][ks]
    with np.errstate(divide="ignore", invalid="ignore"):
        daily = np.where(PM > 0, NM / PM - ref[None, None, :], np.nan)
        npos = (daily > 0).sum(0)
        nval = np.isfinite(daily).sum(0)
        pers = np.where(nval > 0, npos / nval, np.nan)
    cand = (lo > theta) & (pers >= min_persistence) & reject
    cut = {k: v[0][ci][:, ks] for k, v in est.items() if k != "ref"}
    cut["ref"] = ref
    cut["all"] = {k: v[0][:, ks] for k, v in est.items() if k != "ref"}
    u = dd.display_uid
    st = BinStats(dd=dd, ks=ks, seg_idx=dd.pattern_seg_idx[u], x0=dd.pattern_x0[u], length=dd.seg_len[ks],
                  zone=dd.seg_zone[ks], cols=cols, x=x, est=cut, ci_lo=lo, ci_hi=hi, se=se, p=p, reject=reject,
                  persistence=pers, candidate=cand, draws=draws, daily=daily, theta=theta)
    if do_p:
        _persistent_stats(st, an, est, X, pcols, p_draws, p_draws_h, running,
                          (theta_s if theta_persistent_s is None else theta_persistent_s) / tick,
                          q, level, min_persistence, min_hour_share)
    return st


def _persistent_level(obs: np.ndarray, pas: np.ndarray, length: np.ndarray, running: np.ndarray):
    """obs, pas: (B, Hp, Kd) sums over the all-day hours. Returns the typical running level m (B,)
    in obs / passage / m (median over running segments of all-day obs / passages / length), the all-day
    excess over it (B, Kd) and the hourly excess (B, Hp, Kd), in obs / passage."""
    import warnings
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        O, P = obs.sum(1), pas.sum(1)
        occ = np.where(P > 0, O / np.where(P > 0, P, 1), np.nan)
        m = np.nanmedian(np.where(running[None, :], occ / length[None, :], np.nan), axis=1)
        typ = m[:, None] * length[None, :]
        xa = occ - typ
        xh = np.where(pas > 0, obs / np.where(pas > 0, pas, 1), np.nan) - typ[:, None, :]
    return m, xa, xh


def _persistent_stats(st: BinStats, an, est, X, pcols, p_draws, p_draws_h, running, theta, q, level,
                      min_persistence, min_hour_share) -> None:
    """Criterion (b), persistent lingering: a running-zone segment whose all-day (ALLDAY_HOURS)
    observations per passage exceed the typical running level of the direction (median over running
    segments of obs / passage / m, times its length) by more than ``theta``: bootstrap CI lower bound
    above theta, positive on >= min_persistence of the days, BH at q over the running segments, and
    above theta in >= min_hour_share of the all-day hours and of the reference (evening) hours."""
    dd, ks = st.dd, st.ks
    import warnings
    obs = est["obs"][0][pcols][:, ks]
    pas = est["passages"][0][pcols][:, ks]
    m, xa, xh = _persistent_level(obs[None], pas[None], st.length, running)
    m, xa, xh = float(m[0]), xa[0], xh[0]
    alpha = 1 - level
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(p_draws, [alpha / 2, 1 - alpha / 2], axis=0)
        se = np.nanstd(p_draws, axis=0, ddof=1)
        z = (xa - theta) / se
        z = np.where(se > 0, z, np.where(xa > theta, np.inf, -np.inf))
        p = 0.5 * _erfc(z / math.sqrt(2)).astype(float)
        p[~(np.isfinite(xa) & running)] = np.nan
        NM, PM = X["NM"][:, pcols][:, :, ks].sum(1), X["PM"][:, pcols][:, :, ks].sum(1)
        daily = np.where(PM > 0, NM / PM - m * st.length[None, :], np.nan)
        npos, nval = (daily > 0).sum(0), np.isfinite(daily).sum(0)
        pers = np.where(nval > 0, npos / nval, np.nan)
        ok_h = np.isfinite(xh)
        above = (xh > theta) & ok_h
        share = np.where(ok_h.sum(0) > 0, above.sum(0) / ok_h.sum(0), np.nan)
        hrs = dd.cols[pcols]
        ev = (hrs >= an.params.reference_hours[0]) & (hrs < an.params.reference_hours[1])
        ev_share = np.where(ok_h[ev].sum(0) > 0, above[ev].sum(0) / np.maximum(ok_h[ev].sum(0), 1), np.nan) \
            if ev.any() else np.ones(len(ks))
    reject = _bh(p, q)
    cand = running & (lo > theta) & (pers >= min_persistence) & reject & (share >= min_hour_share) & (ev_share >= 0.5)
    st.p_hours, st.p_level, st.p_x, st.p_xh = hrs, m, xa, xh
    st.p_draws, st.p_draws_h, st.p_daily = p_draws, p_draws_h, daily
    st.p_ci_lo, st.p_ci_hi, st.p_reject, st.p_persistence = lo, hi, reject, pers
    st.p_hour_share, st.p_candidate, st.p_theta = share, cand, theta
    st.p_obs, st.p_pas = obs, pas


def bins_frame(st: BinStats) -> pl.DataFrame:
    Hc, Kd = st.x.shape
    return pl.DataFrame({
        "pattern_uid": [st.dd.display_uid] * (Hc * Kd),
        "direction_id": np.full(Hc * Kd, st.dd.direction_id, np.int8),
        "seg_idx": np.tile(st.seg_idx, Hc),
        "hour": np.repeat(st.cols, Kd).astype(np.int8),
        "excess_per_passage": st.x.reshape(-1),
        "ci_lo": st.ci_lo.reshape(-1), "ci_hi": st.ci_hi.reshape(-1), "se": st.se.reshape(-1),
        "p": st.p.reshape(-1), "bh": st.reject.reshape(-1), "persistence": st.persistence.reshape(-1),
        "candidate": st.candidate.reshape(-1),
    }, nan_to_null=True)


def seg_hours(st: BinStats) -> list[set[int]]:
    """Hours per segment that pass the temporal rule (≥ 2 adjacent candidate hours, or a whole band)."""
    hcols = np.flatnonzero(st.cols >= 0)
    hrs = st.cols[hcols]
    out = []
    for j in range(st.x.shape[1]):
        c = st.candidate[hcols, j]
        keep: set[int] = set()
        for i in range(len(hrs)):
            if not c[i]:
                continue
            prev = i > 0 and c[i - 1] and hrs[i - 1] == hrs[i] - 1
            nxt = i + 1 < len(hrs) and c[i + 1] and hrs[i + 1] == hrs[i] + 1
            if prev or nxt:
                keep.add(int(hrs[i]))
        for b in PEAK_BANDS:
            bi = np.flatnonzero(st.cols == BAND_HOUR[b])
            if len(bi) and st.candidate[bi[0], j]:
                a, z = BANDS[b]
                keep |= {h for h in range(a, z) if h in set(hrs.tolist())}
        out.append(keep)
    return out


def stretches(st: BinStats, gap: int = 1) -> list[tuple[int, int, set[int]]]:
    """[(j0, j1, hours)] in display order (inclusive), merged with a ``gap``-bin tolerance, split at zones."""
    hs = seg_hours(st)
    out: list[list] = []
    for j, h in enumerate(hs):
        if not h:
            continue
        if out:
            j0, j1, hh = out[-1]
            if j - j1 <= gap + 1 and all(st.zone[i] == st.zone[j] for i in range(j1, j + 1)):
                out[-1] = [j0, j, hh | h]
                continue
        out.append([j, j, set(h)])
    return [tuple(s) for s in out]


def persistent_stretches(st: BinStats, gap: int = 1) -> list[tuple[int, int, set[int]]]:
    """Criterion (b) stretches [(j0, j1, hours)]: persistent candidate segments merged like
    :func:`stretches`; hours = output hours whose summed hourly excess over the typical level > theta."""
    if st.p_candidate is None:
        return []
    out: list[list] = []
    for j in np.flatnonzero(st.p_candidate):
        if out and j - out[-1][1] <= gap + 1 and all(st.zone[i] == st.zone[j] for i in range(out[-1][1], j + 1)):
            out[-1][1] = j
        else:
            out.append([j, j])
    hrs = st.p_hours
    keep = np.isin(hrs, st.cols[st.cols >= 0])
    res = []
    for j0, j1 in out:
        sx = np.nansum(st.p_xh[:, j0:j1 + 1], axis=1)
        res.append((int(j0), int(j1), {int(h) for h, v, k in zip(hrs, sx, keep) if k and v > st.p_theta}))
    return res


def _groups(st: BinStats, a: list, b: list, gap: int) -> list[tuple[int, int, set[int], set[int]]]:
    """Merge peak (a) and persistent (b) stretches that overlap or touch (same zone in between) into
    [(j0, j1, hours_a, hours_b)]."""
    items = sorted([(j0, j1, set(h), set()) for j0, j1, h in a] + [(j0, j1, set(), set(h)) for j0, j1, h in b])
    out: list[list] = []
    for j0, j1, ha, hb in items:
        if out and j0 - out[-1][1] <= gap + 1 and all(st.zone[i] == st.zone[j0] for i in range(min(out[-1][1], j0), j0 + 1)):
            g = out[-1]
            g[1] = max(g[1], j1)
            g[2] |= ha
            g[3] |= hb
            g[4] |= bool(ha)
            g[5] |= bool(hb)
        else:
            out.append([j0, j1, ha, hb, bool(ha) or not hb, bool(hb)])
    # flags: g[4] peak stretch inside, g[5] persistent stretch inside
    return [(g[0], g[1], g[2], g[3], g[4], g[5]) for g in out]


def hotspot_status(an: Analysis) -> tuple[bool, str]:
    """(ok, status) of the day rule: hotspots need at least max(quality.min_days, MIN_HOTSPOT_DAYS)
    included days (the day bootstrap and the day persistence are meaningless below)."""
    need = max(int(an.quality.min_days), MIN_HOTSPOT_DAYS)
    n = len(an.days)
    if n < need:
        return False, f"too_few_days: {n} included days < {need}"
    return True, "computed"


HOTSPOT_EXTRA = {"excess_obs_per_day": pl.Float32, "fixed_score": pl.Float32, "peak_ratio": pl.Float32,
                 "seg_keys": pl.List(pl.Utf8), "criterion": pl.Utf8, "excess_s_per_passage": pl.Float32,
                 "peak_excess_per_passage": pl.Float32, "allday_excess_per_passage": pl.Float32,
                 "peak_hours": pl.List(pl.Int8)}


def hotspots(an: Analysis, *, B: int | None = None, theta_s: float = 2.0, q: float = 0.1, level: float = 0.95,
             min_persistence: float = 0.6, gap: int = 1, f_min: float = 1.5, peak_ratio: float = 2.0,
             seed: int = 0, stratified: bool = True, links: pl.DataFrame | None = None,
             stats: dict[int, BinStats] | None = None, theta_persistent_s: float | None = None,
             min_hour_share: float = 0.75, persistent: bool = True, return_status: bool = False):
    """Hotspot stretches for every direction of ``an`` (schema.HOTSPOT + HOTSPOT_EXTRA columns).
    ``links`` (schema.LINK) names the stops.

    Two criteria, one list: ``criterion`` = "peak" (excess over the evening reference, module doc
    steps 1-4), "persistent" (running-zone lingering above the direction's typical running level over
    all day hours, evening included: catches fixed delays such as a signal, which cancel in the
    excess over the evening) or "both" (overlapping stretches merged). ``excess_per_passage`` is the
    headline in observations per vehicle summed over the stretch (``excess_s_per_passage`` ≈ seconds):
    peak -> at the peak hour vs the evening; persistent -> all-day mean vs the typical level; both ->
    at the worst hour vs the typical level. Ranked per direction by ``excess_obs_per_day``.

    Fewer included days than max(quality.min_days, MIN_HOTSPOT_DAYS): an empty frame (status
    "too_few_days: ..."). ``return_status``: return (frame, status) instead of the frame."""
    schema = {**HOTSPOT, **HOTSPOT_EXTRA}
    ok, status = hotspot_status(an)
    rows = []
    if ok:
        names = {}
        if links is not None:
            for r in links.select("link_key", "from_name", "to_name").iter_rows():
                names.setdefault(r[0], (r[1], r[2]))
        for dirid in an.dirs:
            if len(an.dirs[dirid].days) == 0:
                continue
            st = stats[dirid] if stats is not None and dirid in stats else bin_stats(
                an, dirid, B=B, theta_s=theta_s, q=q, level=level, min_persistence=min_persistence, seed=seed,
                stratified=stratified, theta_persistent_s=theta_persistent_s, min_hour_share=min_hour_share,
                persistent=persistent)
            pstr = persistent_stretches(st, gap) if persistent else []
            rows += _describe(an, st, _groups(st, stretches(st, gap), pstr, gap), names, level, f_min, peak_ratio)
    if not rows:
        df = pl.DataFrame(schema=schema)
    else:
        df = pl.DataFrame(rows, schema=schema, orient="row")
        df = df.sort(["direction_id", "excess_obs_per_day"], descending=[False, True])
        df = df.with_columns(pl.int_range(1, pl.len() + 1).over("direction_id").cast(pl.Int16).alias("rank"))
        df = conform(df, schema)
    return (df, status) if return_status else df


def _q(a, alpha):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan"), float("nan")
    lo, hi = np.quantile(a, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _describe(an, st: BinStats, groups, names, level, f_min, peak_ratio_min) -> list[tuple]:
    dd = st.dd
    hcols = np.flatnonzero(st.cols >= 0)
    hrs = st.cols[hcols]
    alpha = 1 - level
    allc = st.est["all"]
    hour_cols_all = dd.hour_cols
    hours_all = dd.cols[hour_cols_all]
    day_sel = hour_cols_all[(hours_all >= 6) & (hours_all <= 20)]
    opp_all = allc["obs_per_passage"]                       # (Hx, Kd)
    cand = np.zeros(len(st.ks), bool)
    for g in groups:
        cand[g[0]:g[1] + 1] = True
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seg_ff = np.nanquantile(opp_all[day_sel] / st.length[None, :], 0.2, axis=0)
    mid = st.x0 + st.length / 2
    D = max(len(dd.days), 1)
    peak_hours = {h for b in PEAK_BANDS for h in range(*BANDS[b])}
    obs_h, pas_h, ref = st.est["obs"], st.est["passages"], st.est["ref"]
    tick = an.params.tick_s
    has_p = st.p_x is not None
    out = []
    for j0, j1, ha, hb, fa, fb in groups:
        sl = slice(j0, j1 + 1)
        crit = "both" if (fa and fb) else ("persistent" if fb else "peak")
        # (a) peak excess over the evening, at the best hour of the peak hours
        pk_val = pk_lo = pk_hi = pk_pers = float("nan")
        pk_hour, exc_a = None, float("nan")
        if ha:
            hidx = np.array([hcols[np.flatnonzero(hrs == h)[0]] for h in sorted(ha)])
            sx = np.nansum(st.x[hidx][:, sl], axis=1)
            pk = int(np.nanargmax(sx))
            pcol = hidx[pk]
            pk_hour = int(st.cols[pcol])
            pk_val = float(sx[pk])
            pk_lo, pk_hi = _q(np.nansum(st.draws[:, pcol, sl], axis=1), alpha)
            dly = st.daily[:, pcol, sl]
            okd = np.isfinite(dly).all(1)
            pk_pers = float((dly[okd].sum(1) > 0).mean()) if okd.any() else float("nan")
            exc_a = float(np.nansum(obs_h[hidx][:, sl] - ref[None, sl] * pas_h[hidx][:, sl]) / D)
        # (b) all-day excess over the typical running level
        al_val = al_lo = al_hi = al_pers = exc_b = float("nan")
        if has_p and st.zone[j0] == "running":     # the typical level is a running-zone level
            al_val = float(np.nansum(st.p_x[sl]))
            al_lo, al_hi = _q(np.nansum(st.p_draws[:, sl], axis=1), alpha)
            dl = st.p_daily[:, sl]
            okd = np.isfinite(dl).all(1)
            al_pers = float((dl[okd].sum(1) > 0).mean()) if okd.any() else float("nan")
            typ = st.p_level * st.length[sl]
            exc_b = float(np.nansum(st.p_obs[:, sl] - typ[None, :] * st.p_pas[:, sl]) / D)
        if crit == "peak":
            val, lo, hi, pers, peak_hour, exc_day = pk_val, pk_lo, pk_hi, pk_pers, pk_hour, exc_a
            hours = ha
        else:
            hp = st.p_hours
            inout = np.isin(hp, hrs)
            sxh = np.where(inout, np.nansum(st.p_xh[:, sl], axis=1), -np.inf)
            ih = int(np.argmax(sxh))
            if crit == "persistent":
                val, (lo, hi), pers, exc_day = al_val, (al_lo, al_hi), al_pers, exc_b
            else:
                val = float(sxh[ih])
                lo, hi = _q(np.nansum(st.p_draws_h[:, ih, sl], axis=1), alpha)
                pers, exc_day = al_pers, float(np.nanmax([exc_a, exc_b]))
            peak_hour = int(hp[ih]) if np.isfinite(sxh[ih]) else pk_hour
            hours = ha | hb
        # fixed score
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            occ = np.nansum(opp_all[day_sel][:, sl], axis=1) / st.length[sl].sum()
            f_num = float(np.nanquantile(occ, 0.2)) if np.isfinite(occ).any() else float("nan")
        zone = st.zone[j0]
        c = 0.5 * (mid[j0] + mid[j1])
        base_m = (~cand) & (st.zone == zone) & np.isfinite(seg_ff)
        near = base_m & (np.abs(mid - c) <= 500)
        pool = seg_ff[near] if near.sum() >= 5 else seg_ff[base_m]
        base = float(np.median(pool)) if len(pool) else float("nan")
        F = f_num / base if base and base > 0 else float("nan")
        # peak ratio over day hours, from the full-sample per-hour summed excess over the evening
        dh = [h for h in range(6, 20) if h in set(hrs.tolist())]
        summed = {h: float(np.nansum(st.x[hcols[np.flatnonzero(hrs == h)[0]], sl])) for h in dh}
        pv = [v for h, v in summed.items() if h in peak_hours]
        ov = [v for h, v in summed.items() if h not in peak_hours]
        pmean = np.mean(pv) if pv else float("nan")
        omean = np.mean(ov) if ov else float("nan")
        ratio = float(pmean / max(omean, st.theta / 2)) if np.isfinite(pmean) else float("nan")
        if crit == "persistent":
            kind = "infrastructure"          # present all day, evening included
        elif crit == "both":
            kind = "mixed"                   # a fixed all-day delay plus a peak surplus
        elif np.isfinite(F) and F >= f_min and not ratio >= peak_ratio_min:
            kind = "infrastructure"
        elif ratio >= peak_ratio_min and not (np.isfinite(F) and F >= f_min):
            kind = "congestion"
        else:
            kind = "mixed"
        lk0 = dd.link_keys[dd.seg_link[st.ks[j0]]]
        lk1 = dd.link_keys[dd.seg_link[st.ks[j1]]]
        n0 = names.get(lk0, (lk0, None))[0]
        n1 = names.get(lk1, (None, lk1))[1]
        out.append((dd.display_uid, dd.direction_id, 0, int(st.seg_idx[j0]), int(st.seg_idx[j1]), float(st.x0[j0]),
                    float(st.x0[j1] + st.length[j1]), n0, n1, zone, kind, sorted(int(h) for h in hours), peak_hour,
                    val, lo, hi, pers, int(dd.V[:, st.ks[sl]].sum(0).min()), exc_day, F, ratio,
                    dd.seg_keys[st.ks[sl]].tolist(), crit, val * tick, pk_val, al_val, sorted(int(h) for h in ha)))
    return out



def to_geojson(hotspots: pl.DataFrame, segments: pl.DataFrame) -> dict:
    """GeoJSON FeatureCollection, one LineString per stretch (concatenated segment geometries)."""
    geo = {}
    for (u,), g in segments.select("pattern_uid", "seg_idx", "geometry").group_by(["pattern_uid"]):
        g = g.sort("seg_idx")
        geo[u] = (g["seg_idx"].to_numpy(), g["geometry"].to_list())
    feats = []
    for r in hotspots.iter_rows(named=True):
        coords: list = []
        if r["pattern_uid"] in geo:
            idx, gs = geo[r["pattern_uid"]]
            for i in np.flatnonzero((idx >= r["from_seg"]) & (idx <= r["to_seg"])):
                for pt in gs[i] or []:
                    if not coords or coords[-1] != list(pt):
                        coords.append(list(pt))
        props = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "LineString", "coordinates": coords} if len(coords) >= 2 else None})
    return {"type": "FeatureCollection", "features": feats}
