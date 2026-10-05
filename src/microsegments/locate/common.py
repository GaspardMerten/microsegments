"""Shared pieces of the locators: the ``Placed`` result, per-date pattern sets, output framing."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .. import schema
from ..io.timeutil import add_local_time
from ..network.patterns import Network


@dataclass
class Placed:
    """Located observations (``frame``, schema.PLACED + carried extra columns) and what happened.

    ``report``: row counts per outcome / drop reason. ``aliases``: terminus codes resolved to a
    stop (given or learned). ``flen``: per (pattern_uid, link_idx) feed length used for rescaling
    (linear input). Extra frame columns: ``direction_id`` (of the pattern), ``obs_row`` (row of the
    input observation; null for terminus pseudo rows' duplicates is never the case, they keep it),
    and every non-schema column of the input (e.g. ``true_*`` of the simulator)."""
    frame: pl.DataFrame
    report: dict[str, int] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    flen: pl.DataFrame | None = None

    def __len__(self) -> int:
        return self.frame.height

    @property
    def counted(self) -> pl.DataFrame:
        return self.frame.filter(pl.col("count"))


@dataclass
class PatInfo:
    uid: str
    direction: int
    kind: str
    stops: list[str]           # raw GTFS stop ids
    norm: list[str]            # normalised ids (feed side)
    names: list[str | None]
    s: np.ndarray              # along-pattern position of each stop (n_stops)
    lens: np.ndarray           # link lengths (n_links)
    keys: list[str]            # link keys
    parent: str | None = None
    parent_offset: int | None = None

    @property
    def length(self) -> float:
        return float(self.s[-1])


def pattern_infos(net: Network, normalise=None) -> dict[str, PatInfo]:
    """PatInfo of every pattern of the network (kind = pattern-level kind, overridden per date)."""
    out = {}
    links = net.links.sort("pattern_uid", "link_idx")
    by = {k[0]: g for k, g in links.group_by("pattern_uid", maintain_order=True)}
    for r in net.patterns.iter_rows(named=True):
        lk = by.get(r["pattern_uid"])
        if lk is None or not len(lk):
            continue
        s0 = lk["s0_m"].to_numpy().astype(np.float64)
        ln = lk["len_m"].to_numpy().astype(np.float64)
        stops = list(r["stop_ids"])
        out[r["pattern_uid"]] = PatInfo(
            uid=r["pattern_uid"], direction=int(r["direction_id"]), kind=r["kind"], stops=stops,
            norm=[normalise(x) if normalise else x for x in stops], names=list(r["stop_names"]),
            s=np.concatenate([s0, [s0[-1] + ln[-1]]]), lens=ln, keys=lk["link_key"].to_list(),
            parent=r["parent_uid"], parent_offset=r["parent_offset"])
    return out


@dataclass
class Version:
    """The patterns running on a set of dates (same pattern set and kinds)."""
    mains: list[PatInfo]
    others: list[PatInfo]      # detours / other patterns of the day (short workings ride the main)
    shorts: list[PatInfo]
    kinds: dict[str, tuple[str, str | None, int | None]]   # uid -> (kind, parent_uid, parent_offset) that day


def date_versions(net: Network, dates, infos: dict[str, PatInfo]) -> tuple[pl.DataFrame, list[Version], int]:
    """Map each date to a version index. A date the network does not cover uses the nearest covered
    date of the same day type (weekday / Saturday / Sunday), else the nearest one.

    Returns (frame service_date -> _ver, _gdate), versions, number of fallback dates."""
    pd_ = net.pattern_days
    covered = sorted(pd_["service_date"].unique().to_list())
    if not covered:
        raise ValueError("the network has no pattern on any date")
    by_date: dict[dt.date, tuple] = {}
    for (d,), g in pd_.group_by("service_date"):
        g = g.sort("direction_id", "n_trips", "pattern_uid", descending=[False, True, False])
        by_date[d] = tuple((r["pattern_uid"], r["kind"], r["parent_uid"], r["parent_offset"], r["is_main"])
                           for r in g.iter_rows(named=True))

    def daytype(d):
        w = d.weekday()
        return 0 if w < 5 else w

    sig_idx: dict[tuple, int] = {}
    versions: list[Version] = []
    rows, n_fb = [], 0
    for d in sorted({x for x in dates if x is not None}):
        g = d
        if d not in by_date:
            n_fb += 1
            same = [c for c in covered if daytype(c) == daytype(d)] or covered
            g = min(same, key=lambda c: (abs((c - d).days), c))
        sig = by_date[g]
        if sig not in sig_idx:
            mains, others, shorts, kinds = [], [], [], {}
            for uid, kind, parent, off, is_main in sig:
                if uid not in infos:
                    continue
                kinds[uid] = (kind, parent, off)
                (mains if is_main else shorts if kind == "short" else others).append(infos[uid])
            sig_idx[sig] = len(versions)
            versions.append(Version(mains, others, shorts, kinds))
        rows.append((d, sig_idx[sig], g))
    frame = pl.DataFrame(rows, schema={"service_date": pl.Date, "_ver": pl.Int32, "_gdate": pl.Date}, orient="row")
    return frame, versions, n_fb


def ensure_local_time(df: pl.DataFrame, tz: str, service_day_start: str) -> pl.DataFrame:
    if not {"service_date", "hour", "dow"} <= set(df.columns):
        df = add_local_time(df, tz, service_day_start)
    return df


def link_of_s(s: np.ndarray, stops_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(link_idx, pos_m) of along-pattern positions, links half-open (lo, hi]: a position equal to a
    stop's s belongs to the end of the link arriving there; s <= 0 is link 0 at pos 0."""
    n_links = len(stops_s) - 1
    li = np.searchsorted(stops_s, s, side="left") - 1
    li = np.clip(li, 0, n_links - 1)
    pos = np.clip(s - stops_s[li], 0.0, None)
    pos = np.minimum(pos, stops_s[li + 1] - stops_s[li])
    return li.astype(np.int16), pos


def finish(df: pl.DataFrame, extras: list[str]) -> pl.DataFrame:
    """Cast to schema.PLACED order, then the extra columns."""
    out = schema.conform(df, schema.PLACED)
    keep = list(schema.PLACED) + [c for c in extras if c in out.columns and c not in schema.PLACED]
    return out.select(keep)
