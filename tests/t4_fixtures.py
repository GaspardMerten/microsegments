"""Synthetic Placed / Segment / Coverage / Passages / PatternDay for the T4 tests (aggregate, metrics,
hotspots, tune). Self-contained: does not use simulate.py / segments.py.

Truth: an occupancy density per metre per passage, dens(x, h) [obs / m / vehicle], made of
* free running at ``speed`` m/s: 1 / (speed * tick) per metre,
* a dwell bump of ``stop_dwell_s`` before every stop (Gaussian, sd 5 m, centred 6 m before the stop),
* injected hotspots: ``extra_s`` seconds spread uniformly over [x0, x1] during ``hours``, times a
  per-day Gamma(shape) multiplier (mean 1).
Counts per day x hour x 1 m cell are Poisson(passages * coverage * dens * day factor), which is the
same as Poisson counts per vehicle ~ dwell / tick summed over vehicles.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from microsegments.schema import COVERAGE, PASSAGES, PATTERN_DAY, PLACED, SEGMENT, conform

TICK = 20.0
PEAK = (7, 8, 9, 16, 17, 18)
INFRA_HOURS = tuple(range(6, 20))


@dataclass
class Hot:
    link: int
    lo: float            # link-relative
    hi: float
    extra_s: float
    hours: tuple[int, ...]
    kind: str            # "infrastructure" | "congestion"
    shape: float = 8.0   # day-to-day gamma shape


@dataclass
class Sim:
    placed: pl.DataFrame
    coverage: pl.DataFrame
    passages: pl.DataFrame
    pattern_days: pl.DataFrame
    link_len: np.ndarray
    link_keys: list[str]
    hots: list[Hot]
    days: list[dt.date]
    hours: list[int]
    speed: float
    stop_dwell_s: float
    cut_days: set = field(default_factory=set)
    stop_zone: tuple[float, float] = (30.0, 60.0)

    @property
    def link_x0(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.link_len)[:-1]])

    def segment_fn(self, length: float, phase: float = 0.0, stop_zone=None) -> pl.DataFrame:
        segs = make_segments(self.link_len, self.link_keys, "P0", length, phase, stop_zone or self.stop_zone)
        if self.cut_days:
            n = len(self.link_len) - 1
            segs = pl.concat([segs, segs.filter(pl.col("link_idx") < n).with_columns(pl.lit("P1").alias("pattern_uid"))])
        return segs

    def dens(self, x: np.ndarray, hour: int, factor: float | np.ndarray = 1.0) -> np.ndarray:
        """True density obs / m / passage at pattern position x (m)."""
        return density(x, hour, self.link_len, self.hots, self.speed, self.stop_dwell_s, factor)

    def truth_seg(self, segs: pl.DataFrame, hour: int, ref_hours=(20, 23)) -> dict[str, np.ndarray]:
        """Expected obs per passage per segment of P0 at ``hour`` and its evening reference."""
        s = segs.filter(pl.col("pattern_uid") == "P0").sort("seg_idx")
        x0, ln = s["x0_m"].to_numpy(), s["len_m"].to_numpy()
        tot = self.link_len.sum()
        grid = np.arange(0.05, tot, 0.1)
        cell = np.searchsorted(np.concatenate([x0, [tot]]), grid, side="right") - 1
        cell = np.clip(cell, 0, len(x0) - 1)
        def occ(h):
            # the generator draws counts per 1 m cell with the density at the cell midpoint
            return np.bincount(cell, weights=self.dens(np.floor(grid) + 0.5, h) * 0.1, minlength=len(x0))
        ref = np.mean([occ(h) for h in range(*ref_hours)], axis=0)
        o = occ(hour)
        return {"opp": o, "ref": ref, "excess": o - ref}


def density(x, hour, link_len, hots, speed, stop_dwell_s, factor=1.0):
    x = np.asarray(x, float)
    d = np.full(x.shape, 1.0 / (speed * TICK))
    ends = np.cumsum(link_len)
    sd = 5.0
    for e in ends:
        c = e - 6.0
        d += (stop_dwell_s / TICK) * np.exp(-0.5 * ((x - c) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    x0 = np.concatenate([[0.0], ends[:-1]])
    for i, hs in enumerate(hots):
        if hour in hs.hours:
            a, b = x0[hs.link] + hs.lo, x0[hs.link] + hs.hi
            f = factor[i] if np.ndim(factor) else factor
            d += np.where((x > a) & (x <= b), hs.extra_s / TICK / (b - a), 0.0) * f
    return d


def make_segments(link_len, link_keys, pattern_uid, length, phase=0.0, stop_zone=(30.0, 60.0)) -> pl.DataFrame:
    """Equal split per link, L' = len / round(len / L); a phase shifts the interior boundaries."""
    rows = []
    up, down = stop_zone
    seg_idx = 0
    x0 = 0.0
    for li, (ln, lk) in enumerate(zip(link_len, link_keys)):
        n = max(1, int(round(ln / length)))
        lp = ln / n
        ph = phase % lp
        inner = [ph + j * lp for j in range(n + 1)]
        bnd = [0.0] + [b for b in inner if 1.0 <= b <= ln - 1.0] + [ln]
        bnd = sorted(set(round(b, 6) for b in bnd))
        for k, (a, b) in enumerate(zip(bnd[:-1], bnd[1:])):
            mid = 0.5 * (a + b)
            zone = "stop" if (mid >= ln - up or mid <= down) else "running"
            rows.append((pattern_uid, seg_idx, li, lk, f"{lk}/{k}@{length:g}:{phase:g}", k, a, x0 + a, b - a, zone,
                         [[0.001 * (x0 + a), 50.0], [0.001 * (x0 + b), 50.0]]))
            seg_idx += 1
        x0 += ln
    df = pl.DataFrame(rows, schema=list(SEGMENT), orient="row")
    return conform(df, SEGMENT)


def simulate(*, link_len=(450, 380, 520, 410, 480), n_days=40, seed=0, hots=(), speed=7.0, stop_dwell_s=20.0,
             veh_per_h=None, outages=None, cut_days=(), start=dt.date(2025, 3, 3), weekdays=(0, 1, 2, 3, 4),
             hours=tuple(range(5, 24)), extra_false_rows=0) -> Sim:
    """outages: {(day_index, hour): coverage}; cut_days: day indices where the last link is not run
    (pattern P1 is main)."""
    rng = np.random.default_rng(seed)
    link_len = np.asarray(link_len, float)
    link_keys = [f"S{i}-S{i + 1}" for i in range(len(link_len))]
    days = []
    d = start
    while len(days) < n_days:
        if d.weekday() in weekdays:
            days.append(d)
        d += dt.timedelta(days=1)
    veh_per_h = veh_per_h or {h: (10 if h in PEAK else 4 if h < 6 else 6 if h >= 20 else 8) for h in hours}
    outages = outages or {}
    cut = {days[i] for i in cut_days}
    tot = link_len.sum()
    ends = np.cumsum(link_len)
    x0s = np.concatenate([[0.0], ends[:-1]])
    cells = np.arange(int(np.floor(tot)))            # 1 m cells (j, j+1]
    cmid = cells + 0.5
    cell_link = np.searchsorted(ends, cmid, side="right")
    cell_link = np.minimum(cell_link, len(link_len) - 1)

    rows_d, rows_h, rows_x, rows_v = [], [], [], []
    pas_rows, cov_rows = [], []
    for di, day in enumerate(days):
        fac = np.array([rng.gamma(h.shape, 1.0 / h.shape) for h in hots]) if hots else 1.0
        for h in hours:
            c = outages.get((di, h), 1.0)
            cov_rows.append((day, h, int(c * 180), c * 3600.0, c, 0.0, 0))
            P = max(1, rng.poisson(veh_per_h[h]))
            dens = density(cmid, h, link_len, hots, speed, stop_dwell_s, fac)
            if day in cut:
                dens = np.where(cell_link == len(link_len) - 1, 0.0, dens)
            n = rng.poisson(P * c * dens)
            idx = np.repeat(cells, n)
            rows_d.append(np.full(len(idx), di))
            rows_h.append(np.full(len(idx), h))
            rows_x.append(idx + 1.0 - rng.random(len(idx)))
            rows_v.append(rng.integers(0, P, len(idx)))
            for li, lk in enumerate(link_keys):
                if day in cut and li == len(link_len) - 1:
                    continue
                pas_rows.append((day, h, "P1" if day in cut else "P0", lk, float(P), None, float(P)))
    di = np.concatenate(rows_d)
    hh = np.concatenate(rows_h)
    x = np.concatenate(rows_x)
    veh = np.concatenate(rows_v)
    li = np.minimum(np.searchsorted(ends, x, side="left"), len(link_len) - 1)
    pos = x - x0s[li]
    day_arr = np.array(days, dtype="datetime64[D]")[di]
    ts = (day_arr.astype("datetime64[s]") + (hh * 3600 + rng.integers(0, 3600, len(x))).astype("timedelta64[s]"))
    pat = np.where(np.isin(day_arr, np.array(sorted(cut), dtype="datetime64[D]")), "P1", "P0")
    placed = pl.DataFrame({
        "ts": ts.astype("datetime64[us]"),
        "service_date": day_arr,
        "hour": hh,
        "dow": ((day_arr.astype("int64") + 3) % 7),
        "route_id": np.full(len(x), "R"),
        "pattern_uid": pat,
        "link_idx": li,
        "link_key": np.array(link_keys, dtype=object)[li],
        "pos_m": pos,
        "s_m": x,
        "track_id": np.char.add(np.char.add(di.astype(str), "_"), np.char.add(hh.astype(str), np.char.add("_", veh.astype(str)))),
        "count": np.ones(len(x), bool),
        "flags": np.zeros(len(x), np.uint16),
    }).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    if extra_false_rows:
        extra = placed.sample(extra_false_rows, seed=seed).with_columns(pl.lit(False).alias("count"))
        placed = pl.concat([placed, extra])
    placed = conform(placed, PLACED)
    coverage = conform(pl.DataFrame(cov_rows, schema=["service_date", "hour", "n_snapshots", "covered_s", "coverage",
                                                      "frozen_s", "flags"], orient="row"), COVERAGE)
    passages = conform(pl.DataFrame(pas_rows, schema=["service_date", "hour", "pattern_uid", "link_key", "n_feed",
                                                      "n_events", "n"], orient="row"), PASSAGES)
    pd_rows = []
    for day in days:
        if day in cut:
            pd_rows += [(day, "R", 0, "P1", 100, True), (day, "R", 0, "P0", 0, False)]
        else:
            pd_rows.append((day, "R", 0, "P0", 100, True))
    pattern_days = conform(pl.DataFrame(pd_rows, schema=list(PATTERN_DAY), orient="row"), PATTERN_DAY)
    return Sim(placed=placed, coverage=coverage, passages=passages, pattern_days=pattern_days, link_len=link_len,
               link_keys=link_keys, hots=list(hots), days=days, hours=list(hours), speed=speed,
               stop_dwell_s=stop_dwell_s, cut_days=cut)


def hotspot_line(n_links=10, seed=0):
    """A longer line with 4 infrastructure-like and 4 congestion-like hotspots, mid-link (running zone)."""
    link_len = [520, 480, 560, 500, 540, 470, 510, 530, 490, 550][:n_links]
    hots = []
    for i in range(min(n_links, 8)):
        lo = 180.0 + 20 * (i % 3)
        if i % 2 == 0:
            hots.append(Hot(i, lo, lo + 60, 16.0, INFRA_HOURS, "infrastructure"))
        else:
            hots.append(Hot(i, lo, lo + 60, 24.0, PEAK, "congestion"))
    return link_len, hots
