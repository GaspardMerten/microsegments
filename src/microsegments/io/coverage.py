"""Feed health per local service date x hour (``schema.COVERAGE``).

Each non-frozen poll at t_i covers [t_i, t_i + min(t_next - t_i, gap_cap_s)), t_next being the next
non-frozen poll (the last poll covers min(tick_s, gap_cap_s)). Each frozen snapshot covers
[t_i, t_i + min(t_next_any - t_i, gap_cap_s)) of ``frozen_s`` (t_next_any: next snapshot of any kind).
Intervals are split exactly at local hour boundaries and credited to the hour they fall in.
``coverage = covered_s / 3600`` clipped to 1. Frozen snapshots never count as polls.

FROZEN flag: an hour is flagged when it overlaps a *stall*, a run of consecutive frozen snapshots
lasting more than ``max_frozen_s`` (each covering min(next snapshot - t, gap_cap_s)). Isolated
repeats are normal for feeds that refresh more slowly than they are polled (STIB since 2026: about
one poll in five repeats the previous one, 300-700 s per hour in total) and do not flag the hour;
``frozen_s`` still reports their total.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from ..schema import COVERAGE, SNAPSHOT, UTC, Flag
from .timeutil import add_local_time, parse_hhmm, utc_offset_s


def snapshots_from_observations(obs: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Distinct observation times as polls (``frozen`` False, ``n_rows`` = rows at that ts)."""
    lf = obs.lazy() if isinstance(obs, pl.DataFrame) else obs
    return (lf.group_by("ts").agg(pl.len().cast(pl.UInt32).alias("n_rows"))
            .with_columns(pl.lit(False).alias("frozen"))
            .select(pl.col("ts").cast(UTC), "n_rows", "frozen").sort("ts").collect())


def _split_by_hour(a: np.ndarray, b: np.ndarray, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """Split [a, b) epoch-second intervals at local hour boundaries -> (piece start, piece length)."""
    starts, lens = [], []
    a = a.astype(np.float64).copy()
    b = b.astype(np.float64)
    keep = b > a
    a, b = a[keep], b[keep]
    while a.size:
        off = utc_offset_s(a, tz)
        nxt = np.floor((a + off) / 3600.0) * 3600.0 + 3600.0 - off
        nxt = np.where(nxt <= a, a + 3600.0, nxt)   # guard (DST edge)
        end = np.minimum(b, nxt)
        starts.append(a.copy())
        lens.append(end - a)
        more = end < b
        a, b = end[more], b[more]
    if not starts:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(starts), np.concatenate(lens)


def _hour_table(t: np.ndarray, value: np.ndarray, name: str, tz: str, sds: str) -> pl.DataFrame:
    df = pl.DataFrame({"ts": (np.floor(t) * 1_000_000).astype(np.int64), name: value}).with_columns(
        pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))
    return (add_local_time(df, tz, sds).group_by("service_date", "hour")
            .agg(pl.col(name).sum()))


def coverage(snapshots: pl.DataFrame | pl.LazyFrame, tz: str = "Europe/Brussels",
             service_day_start: str = "04:00", tick_s: float = 20.0, gap_cap_s: float = 40.0,
             max_frozen_s: float = 300.0, min_coverage: float = 0.8,
             dates: tuple[dt.date, dt.date] | None = None) -> pl.DataFrame:
    """SNAPSHOT -> COVERAGE, one row per service date x hour of the service day (24 rows per date,
    from the service-day start hour, e.g. 4..27), zero-filled where the feed was silent.

    ``dates``: (first, last) service dates of the grid; default: span of the snapshots."""
    df = snapshots.collect() if isinstance(snapshots, pl.LazyFrame) else snapshots
    if "frozen" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("frozen"))
    df = (df.select(pl.col("ts").cast(UTC), pl.col("frozen").cast(pl.Boolean).fill_null(False))
          .sort("ts", "frozen").unique("ts", keep="first", maintain_order=True))
    t = df["ts"].dt.epoch("us").to_numpy().astype(np.float64) / 1e6
    fr = df["frozen"].to_numpy().astype(bool)
    cap = float(gap_cap_s) if gap_cap_s is not None else np.inf

    # polls
    tp = t[~fr]
    gap = np.diff(tp, append=tp[-1] + tick_s if tp.size else np.zeros(0))
    ps, pl_ = _split_by_hour(tp, tp + np.minimum(gap, cap), tz)
    # frozen snapshots
    nxt_any = np.append(t[1:], t[-1] + tick_s) if t.size else t
    tf, gf = t[fr], (nxt_any - t)[fr]
    fs, fl = _split_by_hour(tf, tf + np.minimum(gf, cap), tz)

    # stalls: runs of consecutive frozen snapshots longer than max_frozen_s
    ss, sl = np.zeros(0), np.zeros(0)
    if fr.any():
        span = np.minimum(nxt_any - t, cap)
        edge = np.diff(np.concatenate([[0], fr.astype(np.int8), [0]]))
        r0, r1 = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)     # [r0, r1) frozen runs
        csum = np.concatenate([[0.0], np.cumsum(np.where(fr, span, 0.0))])
        dur = csum[r1] - csum[r0]
        long = dur > max_frozen_s
        if long.any():
            ss, sl = _split_by_hour(t[r0[long]], t[r0[long]] + dur[long], tz)

    sds = service_day_start
    cov = _hour_table(ps, pl_, "covered_s", tz, sds)
    frz = _hour_table(fs, fl, "frozen_s", tz, sds)
    stall = _hour_table(ss, sl, "stall_s", tz, sds) if ss.size else None
    nsn = _hour_table(tp, np.ones_like(tp), "n_snapshots", tz, sds)

    if dates is None:
        if t.size == 0:
            return pl.DataFrame(schema=COVERAGE)
        sd = add_local_time(df.select("ts"), tz, sds)["service_date"]
        dates = (sd.min(), sd.max())
    h0 = parse_hhmm(sds) // 3600
    grid = pl.DataFrame({"service_date": pl.date_range(dates[0], dates[1], "1d", eager=True)}).join(
        pl.DataFrame({"hour": pl.Series(range(h0, h0 + 24), dtype=pl.Int8)}), how="cross")
    out = (grid.join(cov, on=["service_date", "hour"], how="left")
           .join(frz, on=["service_date", "hour"], how="left")
           .join(nsn, on=["service_date", "hour"], how="left")
           .with_columns(pl.col("covered_s", "frozen_s", "n_snapshots").fill_null(0.0)))
    if stall is not None:
        out = out.join(stall, on=["service_date", "hour"], how="left").with_columns(pl.col("stall_s").fill_null(0.0))
    else:
        out = out.with_columns(pl.lit(0.0).alias("stall_s"))
    c = (pl.col("covered_s") / 3600.0).clip(0.0, 1.0)
    flags = (pl.when(c < min_coverage).then(Flag.LOW_COVERAGE).otherwise(0)
             + pl.when(pl.col("stall_s") > 0).then(Flag.FROZEN).otherwise(0))
    out = out.with_columns(c.alias("coverage"), flags.alias("flags"))
    return out.select([pl.col(k).cast(v) for k, v in COVERAGE.items()]).sort("service_date", "hour")


def coverage_from_config(snapshots, cfg, dates=None) -> pl.DataFrame:
    """``coverage`` with the parameters of a ``Config``."""
    return coverage(snapshots, tz=cfg.input.timezone, service_day_start=cfg.input.service_day_start,
                    tick_s=cfg.params.tick_s, gap_cap_s=cfg.params.gap_cap_s,
                    max_frozen_s=cfg.quality.max_frozen_s, min_coverage=cfg.quality.min_hour_coverage,
                    dates=dates)


__all__ = ["coverage", "coverage_from_config", "snapshots_from_observations", "SNAPSHOT"]
