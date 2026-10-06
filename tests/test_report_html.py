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
    with pytest.warns(DeprecationWarning):
        s2 = render(c, lang="en", scales={"oph": 3})
    assert '<html lang="en">' in s2 and '"scales"' not in s2
    frag = export(c, tmp_path / "f.html", standalone=False).read_text()
    assert "<body>" not in frag and "ms-data" in frag


def test_plots(an_hs, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from microsegments import plot
    an, hs = an_hs
    figs = [plot.matrix(an, 0, hotspots=hs), plot.matrix(an, 0, "ex", days=[0, 1], stops=False),
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


def _page_scale(html: str) -> dict:
    """The page's colour-scale constants (``const SC = {...}``) as a dict."""
    m = re.search(r"const SC = (\{.*?\});", html)
    assert m, "no SC constant in the page"
    return json.loads(re.sub(r"(\w+):", r'"\1":', m.group(1)))


def test_scale_fixed_whatever_the_data(an_hs):
    """The colour scale and legend constants do not depend on the data: two different analyses give the
    same page scale, equal to microsegments.scale; nothing in the page computes a scale from values."""
    from microsegments import scale
    an, hs = an_hs
    ll, hots = hotspot_line(4)
    sim = simulate(link_len=ll, hots=hots, n_days=7, seed=11)
    segs = sim.segment_fn(30)
    an2 = metrics.analyse(aggregate.count(sim.placed, segs), sim.coverage, sim.passages, segs, sim.pattern_days)
    p1, p2 = render(to_contract(an, hotspots=hs)), render(to_contract(an2), lang="en")
    assert _page_scale(p1) == _page_scale(p2) == scale.CONSTANTS
    script = p1[p1.index('<script>\n"use strict"'):]
    for pat in ("xs.length * .98", "quantile", "D.scales", "SCALES"):
        assert pat not in script
    assert "scales" not in json.loads(re.search(r'<script id="ms-data" type="application/json">(.*?)</script>', p1, re.S).group(1))
    # the page ramp is the same as the Python one
    ramp = re.search(r"const RAMP = (\[.*?\]);", p1).group(1)
    assert [tuple(x) for x in json.loads(ramp.replace("'", '"'))] == [(float(a), b) for a, b in scale.RAMP]


def test_scale_functions():
    import numpy as np

    from microsegments import scale
    # observations per passage: 1 obs (20 s) in a 30 m segment = 20 s; 15 m segment = 40 s per 30 m
    assert scale.to_seconds("obs_per_passage", [1.0, 1.0], [30.0, 15.0]).tolist() == [20.0, 40.0]
    # both excesses share one scale; negative excess = fluid
    assert scale.to_seconds("excess_per_passage", 0.5, 30.0) == scale.to_seconds("excess_line_per_passage", 0.5, 30.0) == 13.0
    assert scale.to_seconds("excess_per_passage", -2.0, 30.0) == scale.BASE_S
    # the 10 m unit does not change the colour: same T for the same observations
    assert np.isclose(scale.to_seconds("obs_per_h", 6.0, 30.0), scale.to_seconds("obs_per_h_10m", 2.0, 30.0))
    # legend labels at the fixed ticks (per 30 m; obs_per_h_10m per 10 m)
    t = scale.T_TICKS
    assert scale.from_seconds("obs_per_passage", t).tolist() == [3, 6, 9, 12, 15, 18, 21, 24]
    assert scale.from_seconds("excess_per_passage", t).tolist() == [0, 3, 6, 9, 12, 15, 18, 21]
    assert scale.from_seconds("obs_per_h", t).tolist() == [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12]
    assert scale.from_seconds("obs_per_h_10m", t).tolist() == [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    assert scale.position(scale.T_MIN) == 0 and scale.position(1e9) == 1
    assert np.isclose(scale.kmh(4.0), 27.0)
    # the legend axis is linear in T: ticks every 3 s, evenly spaced; km/h under them
    assert np.allclose(np.diff(scale.position(np.array(t, dtype=float))), 1 / 7)
    assert np.isclose(scale.position(13.5), 0.5)
    assert np.allclose(scale.kmh(np.array(t, dtype=float))[[0, -1]], [36.0, 4.5])
    # the comparison legend is linear too: -10, -5, 0, 5, 10 s evenly spaced
    assert scale.CMP_TICKS == (-10, -5, 0, 5, 10)
    assert np.allclose(scale.cmp_position(np.array(scale.CMP_TICKS, dtype=float)), [0, .25, .5, .75, 1])


def test_plot_scale_fixed(an_hs):
    import numpy as np
    pytest.importorskip("matplotlib")
    from microsegments import plot, scale
    an, _ = an_hs
    n = plot.norm()
    assert (n.vmin, n.vmax) == (scale.T_MIN, scale.T_MAX)
    # linear norm, as the page's legend bar
    assert np.isclose(float(n(13.5)), 0.5)
    assert plot.auto_vmax(an, "opp") == plot.auto_vmax(None, "opp") == scale.T_MAX
    assert plot.auto_vmax(an, "ex") == scale.T_MAX - scale.BASE_S


def test_page_linear_legend_and_fixed_30m(an_hs):
    """The page's legend axis is linear in T (tPos), the per-10 m toggle is gone, the default measure is the
    time per vehicle, and the hour ticks sit at their true place on the slider."""
    an, hs = an_hs
    p = render(to_contract(an, hotspots=hs))
    script = p[p.index('<script>\n"use strict"'):]
    assert "const tPos = T => (Math.min(SC.T1, Math.max(SC.T0, T)) - SC.T0) / (SC.T1 - SC.T0);" in script
    assert "Math.log(" not in script.split("const tPos")[1].split("\n")[0]
    assert 'id="unit"' not in p and "per10" not in script and "10 m" not in script
    assert "metric: 'opp'" in script
    assert "var(--th) / 2 +" in script
    # big numbers in words, no k / M abbreviation
    assert "' k'" not in script and "' M'" not in script and "millions" in script


def _node():
    import shutil
    n = shutil.which("node")
    if not n:
        pytest.skip("node not installed")
    return n


def _run_js(code: str):
    import subprocess
    out = subprocess.run([_node(), "-e", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_page_stop_zone_is_one_block(an_hs):
    """A shown stop zone is one block: its micro-segments are summed (total time per vehicle, each on its own
    link's passages) and coloured by that total brought to 30 m of the zone's nominal length; running
    micro-segments stay one by one; the zone's tooltip gives the total seconds."""
    an, hs = an_hs
    p = render(to_contract(an, hotspots=hs))
    script = p[p.index('<script>\n"use strict"'):]
    block = script[script.index("// <zones>"):script.index("// </zones>")]
    # stop zone, 30 m explanation, legend swatch, tooltips: FR and EN
    for s in ("stopsHelp:", "why30: `Pourquoi 30 m ? À cette taille", "why30: `Why 30 m?", "zone: 'zone d\\'arrêt (un seul bloc)'",
              "zone: 'stop zone (one block)'", "montée et descente comprises", "boarding included", 'id="mszone"', 'id="stopsQ"'):
        assert s in p, s
    E = {"seg": {"key": list("abcdefgh"), "zone": ["stop", "running", "running", "stop", "stop", "stop", "running", "stop"],
                 "x0": [0, 30, 60, 90, 120, 150, 180, 210], "len": [30] * 8, "link": [0, 0, 0, 0, 1, 1, 1, 1]},
         "stops": [{"x": 0, "name": "A"}, {"x": 120, "name": "B"}, {"x": 240, "name": "C"}]}
    # time per vehicle per micro-segment, in observations per passage (tick 20 s)
    opp = [0.5, 0.2, 0.2, 1.0, 0.1, 0.4, 0.2, 0.6]
    js = ("const P = {stop_zone: [30, 60]}, TICK = 20, SC = {L: 30};"
          "function stopBefore(E, x) { let s = E.stops[0]; for (const t of E.stops) if (t.x <= x + .5) s = t; return s; }"
          + block +
          f"const E = {json.dumps(E)}, opp = {json.dumps(opp)};"
          "const v = opp.map((o, i) => ({i, l: E.seg.link[i], x0: E.seg.x0[i], len: 30, stop: E.seg.zone[i] === 'stop', flags: 0, nd: 10, obs: o,"
          "  opp: o, ex: i === 7 ? null : o - .2, exl: o - .1, oph: o * 10, pph: 8}));"
          "const zv = zvals(E, v), it = items(v, E, zv, true), raw = items(v, E, zv, false);"
          "const c = opp.map((o, i) => ({absent: false, d: .1, dx: .1, lo: i === 3 ? .05 : -.01, hi: .3, oa: o, ob: o + .1, fa: 8, fb: 9}));"
          "const cz = czvals(E, c);"
          "console.log(JSON.stringify({Z: zonesOf(E).Z.map(z => [z.idx, z.name, z.len, z.nom]), zv: zv.map(z => [z.tot.opp, z.s30.opp, z.tot.ex, z.l]),"
          "  items: it.map(b => b.z ? 'Z' + b.zone : b.i), raw: raw.length, cz: cz.map(z => [z.d, z.sig, z.oa])}));")
    r = _run_js(js)
    # three blocks: the terminus (one-sided, nominal 30 m), B (90 m), the end (30 m)
    assert r["Z"] == [[[0], "A", 30, 30], [[3, 4, 5], "B", 90, 90], [[7], "C", 30, 30]]
    tot, s30, ex, link = r["zv"][1]
    assert tot == pytest.approx(30.0) and s30 == pytest.approx(10.0)      # 30 s per vehicle over 90 m = 10 s per 30 m
    assert ex == pytest.approx(30.0 - 3 * 0.2 * 20) and link == 1        # tooltip direction: the outgoing link
    assert r["zv"][2][2] is None                                          # one micro-segment without value: no zone value
    assert r["items"] == ["Z0", 1, 2, "Z1", 6, "Z2"] and r["raw"] == 8
    # comparison: summed differences; "sure" only when the summed bounds exclude zero
    assert r["cz"][1][0] == pytest.approx(.3) and r["cz"][1][1] is True and r["cz"][0][1] is False
    assert r["cz"][1][2] == pytest.approx(1.5)


def test_page_first_of_month_in_french():
    """French dates write the first of the month '1er juillet'; English stays '1 July'."""
    p = render({"v": 1, "dirs": [], "hours": [8], "period": {}, "params": {}})
    script = p[p.index('<script>\n"use strict"'):]
    i = script.index("const fmtDate")
    fmt = script[i:script.index("};", i) + 2]
    js = ("const LOC = {fr: 'fr-BE', en: 'en-GB'}, st = {lang: 'fr'};" + fmt +
          "const a = fmtDate('2026-07-01'), b = fmtDate('2026-07-02'), c = fmtDate('2026-07-01', {weekday: 'long', day: 'numeric', month: 'long', year: 'numeric'}),"
          " s = fmtDate('2026-07-01', {day: '2-digit', month: '2-digit', year: '2-digit'}); st.lang = 'en'; const e = fmtDate('2026-07-01');"
          "console.log(JSON.stringify([a, b, c, s, e]));")
    a, b, c, s, e = _run_js(js)
    assert a == "1er juillet 2026" and b == "2 juillet 2026" and c == "mercredi 1er juillet 2026"
    assert s == "01/07/26" and e == "1 July 2026"


def test_page_stop_zone_nominal_length_of_close_stops(an_hs):
    """Two stops closer than a stop zone share one block; its nominal length is the union of their
    [stop - 30, stop + 60] inside the run (150 m here), not a single 90 m."""
    an, hs = an_hs
    p = render(to_contract(an, hotspots=hs))
    script = p[p.index('<script>\n"use strict"'):]
    block = script[script.index("// <zones>"):script.index("// </zones>")]
    E = {"seg": {"key": list("abcdefg"), "zone": ["running", "stop", "stop", "stop", "stop", "stop", "running"],
                 "x0": [0, 30, 60, 90, 120, 150, 180], "len": [30] * 7, "link": [0] * 7},
         "stops": [{"x": 60, "name": "A"}, {"x": 120, "name": "B"}]}
    js = ("const P = {stop_zone: [30, 60]}, TICK = 20, SC = {L: 30};"
          "function stopBefore(E, x) { let s = E.stops[0]; for (const t of E.stops) if (t.x <= x + .5) s = t; return s; }"
          + block + f"const E = {json.dumps(E)};"
          "console.log(JSON.stringify(zonesOf(E).Z.map(z => [z.idx, z.name, z.len, z.nom])));")
    assert _run_js(js) == [[[1, 2, 3, 4, 5], "A / B", 150, 150]]
