"""Two-period comparison (compare.py) on the T4 synthetic generator."""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from microsegments.compare import compare
from microsegments.config import Params, Select
from microsegments.metrics import analyse

from t4_fixtures import INFRA_HOURS, PEAK, Hot, simulate

LINKS = (450, 380, 520, 410, 480)
HOT = Hot(2, 200.0, 260.0, 30.0, INFRA_HOURS, "infrastructure")


def _an(sim, segs=None, **kw):
    segs = sim.segment_fn(30.0) if segs is None else segs
    from microsegments.aggregate import count
    cube = count(sim.placed, segs, 20.0, per_vehicle=False)
    return analyse(cube, sim.coverage, sim.passages, segs, sim.pattern_days, Select(), Params(), **kw)


def _rekey(sim, old, new):
    """Same link under another link_key (a new GTFS version with a jittered shape)."""
    f = lambda df: df.with_columns(pl.col("link_key").replace(old, new))  # noqa: E731
    sim.placed, sim.passages = f(sim.placed), f(sim.passages)
    sim.link_keys = [new if k == old else k for k in sim.link_keys]
    return sim


def test_hotspot_removed_is_detected():
    a = simulate(link_len=LINKS, n_days=20, seed=1, hots=(HOT,))
    b = simulate(link_len=LINKS, n_days=20, seed=2, hots=(), start=dt.date(2025, 4, 7))
    cmp = compare(_an(a), _an(b), B=100)
    st = cmp.stretches
    assert st.height >= 1
    top = st.row(0, named=True)
    x0 = float(a.link_x0[2]) + 200
    assert top["sign"] == -1                                  # faster after
    assert top["x0_m"] <= x0 + 30 and top["x1_m"] >= x0 + 30  # at the right place
    assert top["delta_s"] == pytest.approx(-30, abs=10)
    assert top["veh_time_s_per_day"] < 0
    assert not top["frequency_changed"]
    # nothing else of consequence
    assert st.filter(pl.col("rank") > 1)["veh_time_s_per_day"].abs().sum() < 0.2 * abs(top["veh_time_s_per_day"])
    c = cmp.to_contract()
    assert c["changes"][0]["sign"] == -1 and len(c["dirs"]) == 1
    assert len(c["dirs"][0]["d"]) == len(c["cols"])


def test_no_false_positive_same_truth():
    a = simulate(link_len=LINKS, n_days=20, seed=3, hots=(HOT,))
    b = simulate(link_len=LINKS, n_days=20, seed=4, hots=(HOT,), start=dt.date(2025, 4, 7))
    cmp = compare(_an(a), _an(b), B=100)
    assert cmp.stretches.height == 0
    assert cmp.bins["significant"].mean() < 0.01


def test_frequency_change_flagged():
    vph = {h: 15 if h in PEAK else 6 if h < 6 else 9 if h >= 20 else 12 for h in range(5, 24)}   # 1.5 x the default
    a = simulate(link_len=LINKS, n_days=20, seed=5, hots=(HOT,))
    b = simulate(link_len=LINKS, n_days=20, seed=6, hots=(), veh_per_h=vph, start=dt.date(2025, 4, 7))
    cmp = compare(_an(a), _an(b), B=100)
    top = cmp.stretches.row(0, named=True)
    assert top["sign"] == -1 and top["frequency_changed"] and top["freq_change"] > 0.2
    f = cmp.frequency.filter(pl.col("hour") == 12)
    assert f["freq_change"][0] == pytest.approx(0.5, abs=0.1)


def test_different_gtfs_versions():
    a = simulate(link_len=LINKS, n_days=20, seed=7, hots=(HOT,), cut_days=range(20))   # last link never run
    b = _rekey(simulate(link_len=LINKS, n_days=20, seed=8, hots=(), start=dt.date(2025, 4, 7)), "S1-S2", "S1-S2~v2")
    cmp = compare(_an(a), _an(b), B=100)
    um = cmp.unmatched
    assert um.filter(pl.col("period") == "b")["link_key"].unique().to_list() == ["S4-S5"]
    assert um.filter(pl.col("period") == "a").height == 0
    bins = cmp.bins
    assert set(bins.filter(pl.col("link_key") == "S1-S2~v2")["match"].unique()) == {"stop_pair"}
    assert bins.filter(pl.col("link_key") == "S4-S5")["delta"].null_count() == bins.filter(pl.col("link_key") == "S4-S5").height
    top = cmp.stretches.row(0, named=True)
    assert top["sign"] == -1 and top["x0_m"] <= float(a.link_x0[2]) + 230 <= top["x1_m"]
