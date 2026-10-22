"""The sensitivity suite over several lines (``sensitivity_table``) and its markdown summary."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import polars as pl

from .sensitivity import sensitivity

SENSITIVITY_COLUMNS = ["line", "mode", "param", "value", "spearman", "jaccard", "link_total_change",
                       "hotspots_stable_share", "n_hotspots", "stable"]


def sensitivity_table(configs, *, length: float | None = None, B: int = 100, dates: str | None = None,
                      log: Callable[[str], None] | None = None, **kw) -> tuple[pl.DataFrame, str]:
    """Run :func:`sensitivity` for every config (``Config`` or TOML path) at its ``params.segment_m``
    (or ``length``), gap-cap variants included (coverage recomputed from the snapshots). Returns one
    tidy table (``SENSITIVITY_COLUMNS``, plus ``days`` and ``baseline_hotspots``) and a markdown summary
    (per line: share of stable variants, the unstable ones; per parameter: worst values)."""
    from ..config import Config
    from ..pipeline import prepare
    say = log or (lambda s: None)
    frames, hs_rows = [], []
    for c in configs:
        cfg = c if isinstance(c, Config) else Config.from_toml(c)
        if dates:
            cfg = replace(cfg, select=replace(cfg.select, dates=dates))
        pre = prepare(cfg, log=log)
        L = float(length or cfg.params.segment_m)
        sr = sensitivity(pre.placed, pre.segment_fn(), L, coverage=pre.coverage, coverage_fn=pre.coverage_fn(),
                         passages=pre.passages, pattern_days=pre.network.pattern_days, select=cfg.select,
                         params=replace(cfg.params, segment_m=L), quality=cfg.quality, per_vehicle=pre.per_vehicle,
                         B=B, **kw)
        n_days = sr.n_days
        line = str(cfg.select.route)
        say(f"line {line}: {sr.summary.height} variants, {'stable' if sr.stable else 'NOT stable'}")
        if sr.summary.height:
            frames.append(sr.summary.with_columns(pl.lit(line).alias("line"), pl.lit(pre.mode).alias("mode"),
                                                  pl.lit(sr.baseline_hotspots.height).alias("baseline_hotspots"),
                                                  pl.lit(n_days).alias("days")))
        hs_rows.append((line, pre.mode, sr.hotspots.height, int(sr.hotspots["stable"].sum()) if sr.hotspots.height else 0))
    table = (pl.concat(frames, how="diagonal_relaxed").select(SENSITIVITY_COLUMNS + ["days", "baseline_hotspots"])
             if frames else pl.DataFrame(schema={c: pl.Utf8 for c in SENSITIVITY_COLUMNS}))
    return table, _markdown(table, hs_rows)


def _fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _markdown(t: pl.DataFrame, hs_rows) -> str:
    out = ["# Sensitivity", "",
           "A variant is *stable* when Spearman (day-band profile, 5 m grid) >= 0.9, hotspot Jaccard >= 0.7 "
           "and link totals within 5 % of the baseline.", ""]
    if not t.height:
        return "\n".join(out + ["No variant ran."])
    out += ["| line | mode | days | variants stable | top hotspots stable (boot >= 80 %, >= 3 phases) | unstable variants |",
            "|---|---|---|---|---|---|"]
    tops = {r[0]: r for r in hs_rows}
    for (line,), g in t.group_by(["line"], maintain_order=True):
        bad = g.filter(~pl.col("stable"))
        bad_s = ", ".join(f"{p}={v}" for p, v in bad.select("param", "value").iter_rows()) or "none"
        h = tops.get(line, (line, None, 0, 0))
        out.append(f"| {line} | {g['mode'][0] or ''} | {g['days'][0]} | {int(g['stable'].sum())}/{g.height} | "
                   f"{h[3]}/{h[2]} | {bad_s} |")
    out += ["", "## Per variant", "", "| line | param | value | Spearman | Jaccard | link total change | baseline hotspots found | stable |",
            "|---|---|---|---|---|---|---|---|"]
    for r in t.iter_rows(named=True):
        out.append(f"| {r['line']} | {r['param']} | {r['value']} | {_fmt(r['spearman'], 3)} | {_fmt(r['jaccard'])} | "
                   f"{_fmt(100 * r['link_total_change'], 1)} % | {_fmt(100 * r['hotspots_stable_share'], 0)} % | "
                   f"{'yes' if r['stable'] else 'no'} |")
    return "\n".join(out) + "\n"
