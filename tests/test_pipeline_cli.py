"""End-to-end: simulate -> config -> run / CLI (needs microsegments.locate)."""
import json
import re
from pathlib import Path

import pytest

try:
    from microsegments.locate import place  # noqa: F401
    HAVE_LOCATE = True
except ImportError:
    HAVE_LOCATE = False

pytestmark = pytest.mark.skipif(not HAVE_LOCATE, reason="microsegments.locate.place not available")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIX = Path(__file__).resolve().parent / "fixtures" / "stib55"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    import importlib.util
    d = tmp_path_factory.mktemp("ex")
    spec = importlib.util.spec_from_file_location("make_synth", EXAMPLES / "make_synth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(d / "data", days=14)
    toml = (EXAMPLES / "synth.toml").read_text().replace("2025-03-23", "2025-03-16")
    (d / "synth.toml").write_text(toml)
    return d / "synth.toml"


def test_run_api(synth):
    import microsegments as ms
    res = ms.run(synth)
    assert res.result.height and len(res.analysis.days) >= 8
    assert res.hotspots is not None and res.hotspots.height >= 1
    v = res.analysis.versions
    assert v.filter(v["direction_id"] == 0).height == 2      # the GTFS cut after 2/3 of the period
    c = res.contract()
    assert c["dirs"] and c["mode"] == "tram" and c["line"] == "S1"


def test_cli_run(synth, tmp_path):
    from microsegments.cli import main
    out = tmp_path / "out"
    assert main(["run", str(synth), "-o", str(out), "-q"]) == 0
    for f in ("result.parquet", "analysis.json", "report.html", "matrix_dir0.png", "matrix_dir1.png", "hotspots.parquet"):
        assert (out / f).exists(), f
    html = (out / "report.html").read_text()
    assert html.startswith("<!doctype html>") and not re.findall(r"__[A-Z]+__", html)
    json.loads((out / "analysis.json").read_text())


def test_cli_inspect_hotspots(synth, capsys):
    from microsegments.cli import main
    assert main(["inspect", str(synth), "-q"]) == 0
    assert main(["hotspots", str(synth), "-q"]) == 0
    out = capsys.readouterr().out
    assert "coverage per service date" in out and "GTFS versions" in out


def test_stib55_example(tmp_path):
    if not (FIX / "vd55").exists():
        pytest.skip("no stib55 fixture")
    import microsegments as ms
    res = ms.run(EXAMPLES / "stib55.toml")
    assert len(res.analysis.days) == 2 and res.hotspots is None    # too few days for hotspots
    p = ms.export(res, tmp_path / "55.html")
    assert "Rogier".upper() in p.read_text().upper()
