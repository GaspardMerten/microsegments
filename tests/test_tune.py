import numpy as np
import polars as pl
import pytest

from microsegments import aggregate, metrics, tune
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
                              passages=sharp.passages, pattern_days=sharp.pattern_days, n_splits=8, B=60, n_jaccard=1)
    t = res.table
    assert t.height == len(LENGTHS)
    assert (t["link_total_dev"] == 0).all()
    assert t["reliability"].is_not_null().all() and t["cv_deviance"].is_not_null().all()
    # coarser bins: more reliable
    assert t["reliability"][-1] > t["reliability"][0]
    oracle = _oracle(sharp)
    assert abs(LENGTHS.index(res.recommended) - LENGTHS.index(oracle)) <= 1, (res.recommended, oracle, t)


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
