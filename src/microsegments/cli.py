"""Command line: ``microsegments run | tune | hotspots | inspect CONFIG``."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl

from .config import Config


def _cfg(args) -> Config:
    cfg = Config.from_toml(args.config)
    if getattr(args, "segment_m", None):
        cfg.params = replace(cfg.params, segment_m=float(args.segment_m))
    if getattr(args, "dates", None):
        cfg.select = replace(cfg.select, dates=args.dates)
    return cfg


def _log(args):
    return (lambda s: print(s, file=sys.stderr)) if not getattr(args, "quiet", False) else None


def _hotspot_table(hs: pl.DataFrame, tick_s: float, names: dict[str, str] | None = None) -> pl.DataFrame:
    if hs is None or hs.height == 0:
        return pl.DataFrame()
    nm = names or {}
    fix = lambda c: pl.col(c).replace(nm) if nm else pl.col(c)  # noqa: E731
    crit = pl.col("criterion") if "criterion" in hs.columns else pl.lit("peak")
    return hs.select(
        "direction_id", "rank",
        pl.concat_str([fix("from_stop_name"), pl.lit(" -> "), fix("to_stop_name")]).alias("stretch"),
        pl.col("x0_m").round(0), pl.col("x1_m").round(0), "zone", "kind", crit.alias("criterion"),
        pl.col("hours").cast(pl.List(pl.Utf8)).list.join(",").alias("hours"),
        pl.col("excess_per_passage").round(2).alias("excess_obs_per_veh"),
        (pl.col("excess_per_passage") * tick_s).round(0).alias("approx_s_per_veh"),
        pl.col("persistence").round(2))


def cmd_run(args) -> int:
    from .pipeline import run
    cfg = _cfg(args)
    res = run(cfg, log=_log(args), hotspots=not args.no_hotspots)
    paths = res.save(args.out, html=not args.no_html, png=not args.no_png, lang=args.lang, title=args.title,
                     source=args.source)
    for name, p in paths.items():
        print(p)
    if not args.quiet:
        t = " ".join(f"{k} {v:.1f}s" for k, v in res.timings.items())
        print(f"done: {len(res.analysis.days)} days, {res.segments.height} segments, "
              f"{0 if res.hotspots is None else res.hotspots.height} hotspots ({t})", file=sys.stderr)
    return 0


def cmd_hotspots(args) -> int:
    from .pipeline import run
    cfg = _cfg(args)
    res = run(cfg, log=_log(args))
    from .report import name_map
    t = _hotspot_table(res.hotspots, cfg.params.tick_s, name_map(cfg.report.names, res.network))
    if t.height == 0:
        print(f"no hotspot ({res.hotspots_status})")
        return 0
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=200):
        print(t)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".geojson":
            from .hotspots import to_geojson
            out.write_text(json.dumps(to_geojson(res.hotspots, res.segments), default=str))
        elif out.suffix == ".csv":
            t.write_csv(out)
        else:
            res.hotspots.write_parquet(out)
        print(out)
    return 0


def cmd_tune(args) -> int:
    from . import tune
    from .pipeline import prepare
    cfg = _cfg(args)
    pre = prepare(cfg, log=_log(args))
    lengths = [float(x) for x in args.lengths.split(",")]
    kw = dict(coverage=pre.coverage, passages=pre.passages, pattern_days=pre.network.pattern_days,
              select=cfg.select, params=cfg.params, quality=cfg.quality, per_vehicle=pre.per_vehicle)
    tr = tune.segment_length(pre.placed, pre.segment_fn(), lengths, B=args.bootstrap, **kw)
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(tr.table)
    print(f"recommended L = {tr.recommended:g} m ({tr.rule})")
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        tr.table.write_parquet(out / "tune.parquet")
    sr = None
    if args.sensitivity:
        sr = tune.sensitivity(pre.placed, pre.segment_fn(), tr.recommended, B=args.bootstrap, **kw)
        with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200, float_precision=3):
            print(sr.summary)
            print(sr.hotspots)
        print("stable" if sr.stable else "NOT stable")
        if out:
            sr.table.write_parquet(out / "sensitivity_profiles.parquet")
            sr.summary.write_parquet(out / "sensitivity.parquet")
    if out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from . import plot
            plot.save(plot.tune_curves(tr), out / "tune.png")
            if sr is not None:
                plot.save(plot.sensitivity(sr), out / "sensitivity.png")
        except ImportError:
            pass
        print(out)
    return 0


def cmd_inspect(args) -> int:
    from .pipeline import prepare
    cfg = _cfg(args)
    pre = prepare(cfg, log=_log(args))
    cov = pre.coverage
    q = cfg.quality
    print(f"route {cfg.select.route}: {pre.observations.height:,} observations, {pre.snapshots.height:,} polls")
    day = (cov.filter(pl.col("hour").is_between(6, 20))
           .group_by("service_date").agg(pl.col("coverage").mean().alias("mean_cov"),
                                         (pl.col("coverage") < q.min_hour_coverage).sum().alias("low_hours"),
                                         (pl.col("frozen_s") > 0).sum().alias("frozen_hours"))
           .sort("service_date"))
    print("\ncoverage per service date (hours 6-20):")
    with pl.Config(tbl_rows=400, float_precision=2):
        print(day)
    print("\nGTFS versions (main pattern runs):")
    with pl.Config(tbl_rows=50, tbl_cols=12, fmt_str_lengths=40, tbl_width_chars=200):
        print(pre.network.versions().with_columns(pl.col("links_added").list.len(), pl.col("links_removed").list.len()))
    if pre.network.missing_dates:
        print("dates without GTFS service:", ", ".join(d.isoformat() for d in pre.network.missing_dates))
    print("\nplacement:")
    for k, v in (pre.place_report or {}).items():
        print(f"  {k:<36} {v:>12,}" if isinstance(v, (int, float)) else f"  {k:<36} {v}")
    from .schema import Flag
    if "flags" in pre.placed.columns and pre.placed.height:
        fl = pre.placed["flags"].fill_null(0).to_numpy()
        for name in ("LAYOVER", "SHORT_WORKING", "OFF_PATTERN", "AMBIGUOUS", "AT_STOP"):
            n = int(((fl & getattr(Flag, name)) != 0).sum())
            if n:
                print(f"  flag {name.lower():<31} {n:>12,}")
        print(f"  counted rows                         {int(pre.placed['count'].fill_null(True).sum()):>12,}")
    if pre.passages is not None and pre.passages.height:
        p = pre.passages
        print(f"\npassages: {p['n'].sum():,.0f} link passages"
              + (f" (feed {p['n_feed'].sum():,.0f}, events {p['n_events'].sum():,.0f})" if p["n_events"].null_count() < p.height else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="microsegments", description="Vehicle observations per micro-segment of a transit line.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("config", help="TOML configuration")
        p.add_argument("--segment-m", type=float, help="override params.segment_m")
        p.add_argument("--dates", help="override select.dates (YYYY-MM-DD..YYYY-MM-DD)")
        p.add_argument("-q", "--quiet", action="store_true")

    p = sub.add_parser("run", help="full pipeline: parquet + JSON + HTML report + PNG matrices")
    common(p)
    p.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    p.add_argument("--lang", default="fr", choices=["fr", "en"])
    p.add_argument("--title")
    p.add_argument("--source", help="data credit shown in the page footer")
    p.add_argument("--no-html", action="store_true")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--no-hotspots", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("tune", help="choose the segment length (and optionally run the sensitivity suite)")
    common(p)
    p.add_argument("--lengths", default="10,15,20,30,40,50,75,100")
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("-o", "--out", help="directory for tune.parquet / tune.png")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("hotspots", help="print the hotspot table")
    common(p)
    p.add_argument("-o", "--out", help="write .parquet, .csv or .geojson")
    p.set_defaults(fn=cmd_hotspots)

    p = sub.add_parser("inspect", help="coverage, GTFS versions, placement / drop counts")
    common(p)
    p.set_defaults(fn=cmd_inspect)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
