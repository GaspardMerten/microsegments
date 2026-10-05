"""Hotspot criteria end to end on the simulated STIB-like line (microsegments.simulate):

* (a) peak excess over the evening: the peak-hour queue;
* (b) persistent lingering above the typical running level: the all-day signal, which the
  evening reference hides (present in the evening too);
* the evening holding at a stop is a stop-zone effect, never a running-zone hotspot;
* a clean line (no injected hotspot) lists none (false-positive control)."""
import polars as pl
import pytest

from microsegments.config import Config
from microsegments.hotspots import hotspots
from microsegments.pipeline import run
from microsegments.simulate import simulate


def _run(tmp, hot: bool, days: int = 21, seed: int = 0):
    sim = simulate(seed=seed, days=days, start="2025-03-03", mode="stib", out_dir=tmp / "gtfs",
                   change_day=days * 2 // 3, hotspots=hot)
    sim.write_stib(tmp / "vd")
    sim.stop_events.write_parquet(tmp / "events.parquet")
    cfg = Config.from_dict({
        "input": {"paths": [str(tmp / "vd" / "*.parquet")], "kind": "stib", "events": str(tmp / "events.parquet")},
        "gtfs": {"path": str(tmp / "gtfs")},
        "select": {"route": "S1", "dates": f"2025-03-03..2025-03-{2 + days:02d}"},
        "params": {"segment_m": 30, "bootstrap": 200},
    })
    return sim, run(cfg)


@pytest.fixture(scope="module")
def hot(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("hot"), True)


def _over(hs, d, a, b):
    return hs.filter((pl.col("direction_id") == d) & (pl.col("x0_m") <= b) & (pl.col("x1_m") >= a))


def test_allday_signal_listed_both_directions(hot):
    sim, res = hot
    hs = res.hotspots
    sig = [h for h in sim.hotspots if h["kind"] == "signal"]
    assert len(sig) == 2
    for h in sig:
        m = _over(hs, h["direction_id"], h["s0_m"], h["s1_m"])
        assert m.height == 1, (h, hs)
        r = m.row(0, named=True)
        assert r["criterion"] in ("persistent", "both") and r["zone"] == "running"
        assert r["kind"] == "infrastructure" if r["criterion"] == "persistent" else r["kind"] == "mixed"
        # 15 s of mean red wait per vehicle ~ 0.75 observation per vehicle at 20 s
        assert 0.4 <= r["allday_excess_per_passage"] <= 1.1, r
        assert abs(r["excess_s_per_passage"] - r["excess_per_passage"] * 20) < 1e-3
        assert r["rank"] <= 2


def test_peak_queue_listed(hot):
    sim, res = hot
    q = next(h for h in sim.hotspots if h["kind"] == "queue")
    m = _over(res.hotspots, q["direction_id"], q["s0_m"], q["s1_m"])
    assert m.height == 1
    r = m.row(0, named=True)
    assert r["criterion"] in ("peak", "both") and r["zone"] == "running" and r["kind"] == "congestion"
    assert r["peak_hour"] in (7, 8, 9, 16, 17, 18)
    assert r["peak_excess_per_passage"] * 20 > 15


def test_evening_holding_is_a_stop_effect(hot):
    sim, res = hot
    h = next(h for h in sim.hotspots if h["kind"] == "holding")
    near = _over(res.hotspots, h["direction_id"], h["s0_m"] - 30, h["s0_m"] + 60)
    assert (near["zone"] == "stop").all(), near
    running = res.hotspots.filter(pl.col("zone") == "running")     # the list with stop zones hidden
    assert _over(running, h["direction_id"], h["s0_m"] - 30, h["s0_m"] + 60).height == 0


def test_clean_line_no_hotspot(tmp_path):
    """No running-zone hotspot and no persistent one. (A stop-zone peak stretch may remain: short
    workings really do dwell longer at their first stop at peak hours in the simulation.)"""
    _, res = _run(tmp_path, False, seed=3)
    running = res.hotspots.filter(pl.col("zone") == "running")
    assert running.height == 0, running
    assert res.hotspots.filter(pl.col("criterion") != "peak").height == 0


def test_hotspots_honour_min_days(hot):
    _, res = hot
    an = res.analysis.subset(res.analysis.days[:4])
    hs, status = hotspots(an, B=20, return_status=True)
    assert hs.height == 0 and status.startswith("too_few_days")
    assert "criterion" in hs.columns
    assert res.hotspots_status == "computed"
