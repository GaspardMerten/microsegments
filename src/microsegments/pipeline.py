"""End-to-end run from a :class:`~microsegments.config.Config`:

observations -> coverage -> network -> place -> passages -> segments -> count -> analyse -> hotspots

``prepare(cfg)`` stops after the passages (everything that does not depend on the segmentation), so
``tune`` and ``inspect`` can reuse it; ``run(cfg)`` does the rest.
"""
from __future__ import annotations

import datetime as dt
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .aggregate import count
from .config import Config
from .io import (
    coverage_from_config,
    read_observations,
    read_snapshots,
    snapshots_from_observations,
)
from .io.timeutil import add_local_time
from .metrics import Analysis, analyse
from .network import GtfsSource, build_network
from .segments import segment

from .hotspots import MIN_HOTSPOT_DAYS  # noqa: F401  (re-exported; floor under quality.min_days)


def _locate():
    """``locate.place`` and ``tracks.passages`` (None if those modules are not available)."""
    try:
        from .locate import place
    except ImportError:
        place = None
    try:
        from .tracks import passages
    except ImportError:
        passages = None
    return place, passages


@dataclass
class Prepared:
    """Everything up to the passages (independent of the segmentation)."""
    cfg: Config
    observations: pl.DataFrame
    snapshots: pl.DataFrame
    coverage: pl.DataFrame
    network: Any                       # network.Network
    placed: pl.DataFrame
    passages: pl.DataFrame | None
    events: pl.DataFrame | None = None
    place_report: dict = field(default_factory=dict)
    per_vehicle: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    mode: str | None = None

    def segment_fn(self) -> Callable[..., pl.DataFrame]:
        """``f(length, phase, stop_zone=...)`` for ``tune``: the config's segmentation of this network."""
        p = self.cfg.params

        def f(length: float, phase: float = 0.0, stop_zone=None) -> pl.DataFrame:
            return segment(self.network, length, phase, p.grid, stop_zone or p.stop_zone, geometry=False)
        return f


@dataclass
class RunResult(Prepared):
    segments: pl.DataFrame | None = None
    cube: pl.DataFrame | None = None
    analysis: Analysis | None = None
    hotspots: pl.DataFrame | None = None
    hotspots_status: str = "not_computed"   # "computed" | "too_few_days: ..." | "not_computed"

    @property
    def result(self) -> pl.DataFrame:
        return self.analysis.result

    def contract(self, **kw) -> dict:
        """The page / API JSON (``report.to_contract``)."""
        from .report import to_contract
        kw.setdefault("mode", self.mode)
        kw.setdefault("names", getattr(getattr(self.cfg, "report", None), "names", None))
        return to_contract(self.analysis, self.network, self.segments, self.hotspots, **kw)

    def export(self, path, **kw) -> Path:
        from .html import export
        return export(self, path, **kw)

    def save(self, out_dir: str | Path, *, html: bool = True, png: bool = True, lang: str = "fr",
             title: str | None = None, source: str | None = None) -> dict[str, Path]:
        """Write result.parquet, hotspots.parquet / .geojson, cube.parquet, excluded_days.parquet,
        versions.parquet, analysis.json (contract), report.html and matrix_dir<d>.png."""
        from .hotspots import to_geojson
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        def w(name, df):
            p = out / name
            df.write_parquet(p)
            paths[name] = p
        w("result.parquet", self.result)
        w("cube.parquet", self.cube)
        w("excluded_days.parquet", self.analysis.excluded_days)
        w("versions.parquet", self.analysis.versions)
        w("coverage.parquet", self.coverage)
        hs = self.hotspots if self.hotspots is not None else pl.DataFrame()
        w("hotspots.parquet", hs)
        if hs.height:
            p = out / "hotspots.geojson"
            p.write_text(json.dumps(to_geojson(hs, self.segments), default=str))
            paths["hotspots.geojson"] = p
        c = self.contract(title=title, source=source)
        p = out / "analysis.json"
        p.write_text(json.dumps(c, separators=(",", ":"), ensure_ascii=False, default=str))
        paths["analysis.json"] = p
        if html:
            from .html import export
            paths["report.html"] = export(c, out / "report.html", lang=lang, title=title)
        if png:
            try:
                import matplotlib
                matplotlib.use("Agg", force=False)
                from . import plot
                for q in plot.save_all(self, out):
                    paths[q.name] = q
            except ImportError:
                pass
        return paths


# ------------------------------------------------------------------------------------ stages
def _dates(cfg: Config, obs: pl.DataFrame) -> list[dt.date]:
    rng = cfg.select.date_range()
    if rng is not None:
        n = (rng[1] - rng[0]).days + 1
        ds = [rng[0] + dt.timedelta(days=i) for i in range(n)]
    else:
        ds = sorted(obs["service_date"].unique().to_list()) if obs.height else []
    wd = set(cfg.select.weekdays)
    return [d for d in ds if d.weekday() in wd]


def _events(cfg: Config) -> pl.DataFrame | None:
    path = cfg.input.events
    if not path:
        return None
    from .io.events import read_events, read_stib_punctuality
    route = cfg.select.route
    if cfg.input.kind == "stib":
        try:
            return read_stib_punctuality(path, route=route, id_normaliser=cfg.input.linear.id_normaliser).collect()
        except Exception:  # noqa: BLE001 - not the STIB layout: generic reader below
            pass
    return read_events(path, ts_unit=cfg.input.ts_unit, tz=cfg.input.timezone,
                       service_day_start=cfg.input.service_day_start, route=route).collect()


def network_for(source: GtfsSource, route: str, dates, route_key: str = "route_short_name",
                say: Callable[[str], None] = lambda s: None):
    """``build_network``, plus a fallback for a single feed whose calendar does not cover some dates
    (e.g. a later timetable used for a past period, as the tram 55 prototype did): those dates take
    the patterns of the nearest date of the same weekday that the feed does serve."""
    from .network.calendar import active_services, service_dates
    from .network.patterns import Network
    net = build_network(source, route, dates, route_key=route_key)
    missing = sorted(set(net.missing_dates) & set(dates))
    if not missing or source.path is None:
        return net
    feed = source.feed_for(missing[0], route)
    span = service_dates(feed) if feed is not None else None
    if span is None:
        return net
    proxy = {}
    for d in missing:
        cands = []
        for k in range((span[1] - span[0]).days + 1):
            c = span[0] + dt.timedelta(days=k)
            if c.weekday() == d.weekday():
                cands.append(c)
        for c in sorted(cands, key=lambda c: abs((c - d).days)):
            if active_services(feed, c):
                proxy[d] = c
                break
    if not proxy:
        return net
    net2 = build_network(source, route, sorted(set(proxy.values())), registry=net.registry, route_key=route_key)
    m = pl.DataFrame({"service_date": list(proxy.values()), "_real": list(proxy.keys())},
                     schema={"service_date": pl.Date, "_real": pl.Date})
    pd2 = (net2.pattern_days.join(m, on="service_date").drop("service_date").rename({"_real": "service_date"})
           .select(net2.pattern_days.columns))
    pds = pl.concat([net.pattern_days, pd2]).sort("service_date", "direction_id", "n_trips", descending=[False, False, True])
    pats = pl.concat([net.patterns, net2.patterns]).unique("pattern_uid", keep="first", maintain_order=True)
    links = pl.concat([net.links, net2.links]).unique(["pattern_uid", "link_idx"], keep="first", maintain_order=True)
    stops = pl.concat([net.stops, net2.stops]).unique("stop_id", keep="first", maintain_order=True)
    say(f"GTFS calendar does not cover {len(proxy)} date(s) "
        f"({min(proxy):%Y-%m-%d}..{max(proxy):%Y-%m-%d}): using the same weekday of the feed ({span[0]}..{span[1]})")
    return Network(route=net.route, patterns=pats, pattern_days=pds, links=links, stops=stops,
                   registry=net.registry, missing_dates=sorted(set(net.missing_dates) - set(proxy)))


def _call(fn, *args, **kw):
    """Call ``fn`` with the keyword arguments it accepts."""
    try:
        sig = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kw)
    if any(p.kind == p.VAR_KEYWORD for p in sig.values()):
        return fn(*args, **kw)
    return fn(*args, **{k: v for k, v in kw.items() if k in sig})


def prepare(cfg: Config, *, log: Callable[[str], None] | None = None) -> Prepared:
    """Read, cover, build the network, place and count passages."""
    say = log or (lambda s: None)
    place, passages_fn = _locate()
    if place is None:
        raise ImportError("microsegments.locate.place is not available")
    if not cfg.select.route:
        raise ValueError("select.route is required")
    tm: dict[str, float] = {}
    t = time.perf_counter()
    obs = read_observations(cfg).collect()
    obs = add_local_time(obs, cfg.input.timezone, cfg.input.service_day_start) if "service_date" not in obs.columns else obs
    tm["read"] = time.perf_counter() - t
    say(f"observations: {obs.height:,} rows")
    if obs.height == 0:
        raise ValueError(f"no observation for route {cfg.select.route!r} in {cfg.input.paths}")
    t = time.perf_counter()
    snaps_lf = read_snapshots(cfg)
    snaps = snaps_lf.collect() if snaps_lf is not None else snapshots_from_observations(obs)
    dates = _dates(cfg, obs)
    rng = (min(dates), max(dates)) if dates else None
    cov = coverage_from_config(snaps, cfg, dates=rng)
    tm["coverage"] = time.perf_counter() - t
    t = time.perf_counter()
    source = GtfsSource.from_config(cfg.gtfs)
    net = network_for(source, cfg.select.route, dates, cfg.gtfs.route_key, say)
    tm["network"] = time.perf_counter() - t
    say(f"network: {net.patterns.height} patterns over {len(net.dates)} dates")
    t = time.perf_counter()
    res = place(obs, net, cfg)
    placed = getattr(res, "frame", res)
    if isinstance(placed, pl.LazyFrame):
        placed = placed.collect()
    report = dict(getattr(res, "report", {}) or {})
    tm["place"] = time.perf_counter() - t
    say(f"placed: {placed.height:,} rows")
    t = time.perf_counter()
    events = _events(cfg)
    pas = None
    if passages_fn is not None:
        pas = _call(passages_fn, res, net, events, tz=cfg.input.timezone,
                    id_normaliser=cfg.input.linear.id_normaliser)
    tm["passages"] = time.perf_counter() - t
    per_vehicle = bool(obs["vehicle_id"].null_count() < obs.height / 2) if "vehicle_id" in obs.columns else False
    mode = None
    try:
        from .report import route_mode
        mode = route_mode(source, cfg.select.route, net.dates[0]) if net.dates else None
    except Exception:  # noqa: BLE001
        pass
    return Prepared(cfg=cfg, observations=obs, snapshots=snaps, coverage=cov, network=net, placed=placed,
                    passages=pas, events=events, place_report=report, per_vehicle=per_vehicle, timings=tm, mode=mode)


def run(cfg: Config | str | Path, *, prepared: Prepared | None = None, hotspots: bool = True,
        log: Callable[[str], None] | None = None, hotspot_kw: dict | None = None) -> RunResult:
    """The whole pipeline. ``cfg``: a Config or a TOML path. ``prepared``: reuse a ``prepare`` result
    (e.g. to try other segment lengths)."""
    if not isinstance(cfg, Config):
        cfg = Config.from_toml(cfg)
    say = log or (lambda s: None)
    pre = prepared or prepare(cfg, log=log)
    p = cfg.params
    tm = dict(pre.timings)
    t = time.perf_counter()
    segs = segment(pre.network, p.segment_m, p.phase_m, p.grid, p.stop_zone)
    cube = count(pre.placed, segs, p.tick_s, per_vehicle=pre.per_vehicle, method=p.per_vehicle)
    tm["count"] = time.perf_counter() - t
    t = time.perf_counter()
    an = analyse(cube, pre.coverage, pre.passages, segs, pre.network.pattern_days, cfg.select, p, cfg.quality)
    tm["analyse"] = time.perf_counter() - t
    say(f"analysis: {len(an.days)} days included, {an.excluded_days.height} excluded")
    hs, hs_status = None, "not_computed"
    if hotspots:
        from .hotspots import hotspot_status
        from .hotspots import hotspots as find
        ok, hs_status = hotspot_status(an)
        if not ok:
            say(f"hotspots skipped: {hs_status} (quality.min_days, floor {MIN_HOTSPOT_DAYS})")
        else:
            t = time.perf_counter()
            kw = {"B": p.bootstrap, **(hotspot_kw or {})}
            hs, hs_status = find(an, links=pre.network.links, return_status=True, **kw)
            tm["hotspots"] = time.perf_counter() - t
            say(f"hotspots: {hs.height}")
    base = {f.name: getattr(pre, f.name) for f in Prepared.__dataclass_fields__.values()}
    base["timings"] = tm
    base["cfg"] = cfg
    return RunResult(**base, segments=segs, cube=cube, analysis=an, hotspots=hs, hotspots_status=hs_status)


__all__ = ["Prepared", "RunResult", "prepare", "run"]
