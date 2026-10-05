"""report.to_contract, html.export, plot smoke tests (synthetic T4 data, no locate needed)."""
import json
import re

import polars as pl
import pytest
from t4_fixtures import hotspot_line, simulate

from microsegments import aggregate, metrics
from microsegments.contract import CONTRACT_VERSION
from microsegments.hotspots import hotspots
from microsegments.html import export, render
from microsegments.report import to_contract

TOP = {"v", "line", "route_id", "mode", "params", "period", "hours", "coverage", "versions", "dirs", "hotspots", "ctx"}
DIR = {"dir", "pattern_uid", "from", "to", "stops", "links", "seg", "wk"}
SEG = {"key", "link", "x0", "len", "zone", "c", "flags"}
WK = {"obs", "cov", "p", "days"}


@pytest.fixture(scope="module")
def an_hs():
    ll, hots = hotspot_line(6)
    sim = simulate(link_len=ll, hots=hots, n_days=12, seed=3, cut_days=(10, 11))
    segs = sim.segment_fn(30)
    an = metrics.analyse(aggregate.count(sim.placed, segs), sim.coverage, sim.passages, segs, sim.pattern_days)
    return an, hotspots(an, B=50)


def test_contract_keys(an_hs):
    an, hs = an_hs
    c = to_contract(an, hotspots=hs, line="T1", mode="tram")
    assert TOP <= set(c) and c["v"] == CONTRACT_VERSION
    assert {"segment_m", "phase_m", "grid", "tick_s", "gap_cap_s", "stop_zone", "reference_hours"} <= set(c["params"])
    assert {"first", "last", "days", "per_dow", "excluded"} <= set(c["period"])
    assert len(c["period"]["per_dow"]) == 7
    assert {"dates", "c"} <= set(c["coverage"]) and len(c["coverage"]["c"]) == len(c["coverage"]["dates"])
    for d in c["dirs"]:
        assert DIR <= set(d) and SEG <= set(d["seg"]) and WK <= set(d["wk"])
        n = len(d["seg"]["key"])
        assert all(len(d["seg"][k]) == n for k in SEG)
        assert len(d["wk"]["obs"]) == 7 and len(d["wk"]["obs"][0]) == len(c["hours"]) and len(d["wk"]["obs"][0][0]) == n
        assert len(d["wk"]["cov"][0][0]) == len(d["links"]) == len(d["stops"]) - 1
        assert max(d["seg"]["link"]) == len(d["links"]) - 1
        # every observation of the display pattern is in wk
        tot = an.result.filter((pl.col("direction_id") == d["dir"]) & (pl.col("pattern_uid") == d["pattern_uid"])
                               & (pl.col("hour").is_in(c["hours"])))["obs"].sum()
        assert sum(sum(sum(r) for r in w) for w in d["wk"]["obs"]) == tot
    # the cut version is offered on its own
    assert c["versions"] and any(not v["display"] for v in c["versions"])
    alt = c["dirs"][0]["alt"]
    assert alt and len(alt[0]["seg"]["key"]) < len(c["dirs"][0]["seg"]["key"])
    assert {"from_seg", "to_seg", "x0_m", "x1_m", "rank", "hours"} <= set(c["hotspots"][0])
    json.dumps(c, allow_nan=False)


def test_export_html(an_hs, tmp_path):
    an, hs = an_hs
    p = export(an, tmp_path / "r.html", title="Ligne <T1>", hotspots=hs, line="T1")
    s = p.read_text()
    assert s.startswith("<!doctype html>") and s.rstrip().endswith("</html>")
    for tag in ("<html", "<head>", "</head>", "<body>", "</body>"):
        assert tag in s
    assert not re.findall(r"__[A-Z]+__", s)
    assert "Ligne &lt;T1&gt;" in s and ".leaflet-pane" in s
    m = re.search(r'<script id="ms-data" type="application/json">(.*?)</script>', s, re.DOTALL)
    data = json.loads(m.group(1))
    assert data["title"] == "Ligne <T1>" and data["dirs"]
    assert "</script" not in m.group(1)
    # contract dict input, English, fragment
    c = to_contract(an, hotspots=hs)
    s2 = render(c, lang="en", scales={"oph": 3})
    assert '<html lang="en">' in s2 and '"scales":{"oph":3}' in s2
    frag = export(c, tmp_path / "f.html", standalone=False).read_text()
    assert "<body>" not in frag and "ms-data" in frag


def test_plots(an_hs, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from microsegments import plot
    an, hs = an_hs
    figs = [plot.matrix(an, 0, hotspots=hs), plot.matrix(an, 0, "ex", days=[0, 1], stops=False, vmax=2),
            plot.profile(an, 0, 8, metric="opp"), plot.profile(an, 0, band="pm", metric="obs_per_h_10m"),
            plot.map(an, 0, "am", hotspots=hs)]
    for i, f in enumerate(figs):
        assert plot.save(f, tmp_path / f"f{i}.png").stat().st_size > 1000
    assert plot.auto_vmax(an, "obs_per_h") > 0
    with pytest.raises(ValueError):
        plot.matrix(an, 0, "nope")
    paths = plot.save_all(an, tmp_path / "m")
    assert len(paths) == len(an.dirs)


def test_tune_plots(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from microsegments import plot, tune
    ll, hots = hotspot_line(4)
    sim = simulate(link_len=ll, hots=hots, n_days=10, seed=1)
    tr = tune.segment_length(sim.placed, sim.segment_fn, (20, 40), coverage=sim.coverage, passages=sim.passages,
                             pattern_days=sim.pattern_days, B=20, n_splits=3, n_jaccard=1)
    plot.save(plot.tune_curves(tr), tmp_path / "t.png")
    sr = tune.sensitivity(sim.placed, sim.segment_fn, 30, coverage=sim.coverage, passages=sim.passages,
                          pattern_days=sim.pattern_days, B=20, phases=(0, 0.5), stop_zones=((15, 15),),
                          references=(("evening", (20, 23)),), passage_sources=("n",))
    plot.save(plot.sensitivity(sr), tmp_path / "s.png")


def test_lazy_plot_import():
    import subprocess
    import sys
    code = "import sys, microsegments as ms; assert 'matplotlib' not in sys.modules; ms.run; ms.export; ms.simulate"
    subprocess.run([sys.executable, "-c", code], check=True)
