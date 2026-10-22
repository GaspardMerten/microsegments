import numpy as np
import polars as pl
import pytest

from microsegments import aggregate, metrics, tune
from microsegments.config import Params
from t4_fixtures import hotspot_line, simulate

LENGTHS = (10, 15, 20, 30, 40, 50, 75, 100)


@pytest.fixture(scope="module")
def sharp():
    """Sharp ~60 m features (hotspots), small stop dwell, sparse data: the best L is not the smallest."""
    ll, hots = hotspot_line(6)
    return simulate(link_len=ll, hots=hots, n_days=15, seed=2, stop_dwell_s=5.0, veh_per_h={h: 6 for h in range(5, 24)})


def _oracle(sim):
    """L minimising the error of the hour x segment matrix (6-20 h) against the true density."""
    g = np.arange(0.5, sim.link_len.sum(), 1.0)
    err = []
    for L in LENGTHS:
        segs = sim.segment_fn(L)
        an = metrics.analyse(aggregate.count(sim.placed, segs), sim.coverage, sim.passages, segs, sim.pattern_days)
        dd = an.dirs[0]
        ks = dd.pattern_segs["P0"]
        e = an.estimate(0)["obs_per_passage"][0]
        j = np.clip(np.searchsorted(dd.pattern_x0["P0"], g, side="right") - 1, 0, len(ks) - 1)
        err.append(np.mean([np.nanmean((e[i][ks][j] / dd.seg_len[ks][j] - sim.dens(g, h)) ** 2)
                            for i, h in enumerate(dd.cols) if 6 <= h <= 20]))
    return LENGTHS[int(np.argmin(err))]


def test_segment_length_matches_truth(sharp):
    res = tune.segment_length(sharp.placed, sharp.segment_fn, LENGTHS, coverage=sharp.coverage,
                              passages=sharp.passages, pattern_days=sharp.pattern_days, n_splits=8, B=60, n_jaccard=1,
                              params=Params(reference_hours=(20, 23)))   # the simulator's quiet evening ends at 23 h
    t = res.table
    assert t.height == len(LENGTHS)
    assert (t["link_total_dev"] == 0).all()
    assert t["reliability"].is_not_null().all() and t["cv_deviance"].is_not_null().all()
    # coarser bins: more reliable
    assert t["reliability"][-1] > t["reliability"][0]
    # the gate is the reliability of the time lost per vehicle; Jaccard and the CV 1-SE rule are diagnostics
    assert t["reliability_excess"].is_not_null().all()
    assert {"cv_within_1se", "jaccard_ok", "eligible"} <= set(t.columns)
    ok = (t["reliability_excess"] >= 0.8) & (t["loc_spread_m"].is_null() | t["loc_spread_m"].is_nan()
                                              | (t["loc_spread_m"] <= 30))
    assert t["eligible"].to_list() == ok.to_list()
    assert res.recommended == t.filter(pl.col("eligible"))["L"].min()
    assert "time lost per vehicle" in res.rule
    oracle = _oracle(sharp)
    assert abs(LENGTHS.index(res.recommended) - LENGTHS.index(oracle)) <= 1, (res.recommended, oracle, t)


def test_segment_length_fallback_keeps_default(sharp):
    """No length passes the gates: the default params.segment_m is kept, not the CV minimum."""
    res = tune.segment_length(sharp.placed, sharp.segment_fn, (10, 15, 50), coverage=sharp.coverage,
                              passages=sharp.passages, pattern_days=sharp.pattern_days, n_splits=3, B=20,
                              n_jaccard=1, min_reliability=1.01)
    assert not res.table["eligible"].any()
    assert res.recommended == 30.0
    assert "no length passed" in res.rule and "default segment_m=30" in res.rule


def test_sensitivity():
    ll, hots = hotspot_line(4)
    sim = simulate(link_len=ll, hots=hots, n_days=25, seed=3)

    def cov_fn(cap):
        return sim.coverage
    r = tune.sensitivity(sim.placed, sim.segment_fn, 30, coverage=sim.coverage, coverage_fn=cov_fn,
                         passages=sim.passages, pattern_days=sim.pattern_days, B=100)
    s = r.summary
    assert set(s["param"]) == {"phase", "gap_cap_s", "stop_zone", "reference", "passages", "segment_m"}
    sm = s.filter(pl.col("param") == "segment_m")
    assert set(sm["value"]) == {"15.0", "60.0"} and (sm["link_total_change"] < 1e-9).all()
    assert (s.filter(pl.col("param") == "phase")["link_total_change"] < 1e-9).all()
    assert (s.filter(pl.col("param") == "gap_cap_s")["spearman"] > 0.999).all()
    assert (s["hotspots_stable_share"] >= 0.75).all()
    assert r.hotspots.height == 4 and r.hotspots["stable"].all()
    assert {"variant", "x_m", "obs_per_h_m"} <= set(r.table.columns)
    assert isinstance(r.stable, bool)


def test_stop_zone_from_profile():
    sim = simulate(n_days=10, seed=4, stop_dwell_s=30.0)
    up, down = tune.stop_zone_from_profile(sim.placed, sim.segment_fn(30))
    # dwell bump: Gaussian sd 5 m centred 6 m before the stop
    assert 10 <= up <= 30 and down <= 10


def test_score_sums_deviance_per_date_across_directions(monkeypatch):
    """Per-day CV deviance: each direction's days added on their own dates, not by position."""
    from types import SimpleNamespace as NS
    from microsegments.tune import length as T
    d = lambda *s: np.array(s, dtype="datetime64[D]")  # noqa: E731
    dirs = {0: NS(days=d("2025-03-03", "2025-03-04", "2025-03-05"), pattern_segs={"u": [0]}, display_uid="u"),
            1: NS(days=d("2025-03-04", "2025-03-05"), pattern_segs={"u": [0]}, display_uid="u")}
    dev = {0: np.array([1.0, 2.0, 3.0]), 1: np.array([10.0, 20.0])}
    for f, v in {"_overlap": lambda an, k, fine: (None,), "_fine_counts": lambda *a: None,
                 "_loo_deviance": lambda an, k, Y, ov: dev[k], "_split_half": lambda *a: (1.0, 1.0, 1.0),
                 "_ss_cost": lambda *a: 0.0, "bin_stats": lambda *a, **k: None, "stretches": lambda st: [],
                 "_loc_spread": lambda *a: np.nan, "_split_jaccard": lambda *a: 1.0}.items():
        monkeypatch.setattr(T, f, v)
    row, per_day = T._score(NS(dirs=dirs), None, None, None, n_splits=1, B=1, n_jaccard=1, seed=0, hotspot_kw={})
    assert per_day.tolist() == [1.0, 12.0, 23.0]
    assert row["cv_deviance"] == pytest.approx(12.0)
