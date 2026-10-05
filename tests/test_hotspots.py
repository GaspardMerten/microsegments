import json

import numpy as np
import polars as pl
import pytest

from microsegments import aggregate, metrics
from microsegments.hotspots import bin_stats, bins_frame, bootstrap_weights, hotspots, to_geojson
from microsegments.schema import HOTSPOT, validate
from t4_fixtures import Hot, hotspot_line, simulate


def _analyse(sim, L=30):
    segs = sim.segment_fn(L)
    return metrics.analyse(aggregate.count(sim.placed, segs), sim.coverage, sim.passages, segs, sim.pattern_days), segs


@pytest.fixture(scope="module")
def hot_sim():
    ll, hots = hotspot_line(10)
    return simulate(link_len=ll, hots=hots, n_days=40, seed=1)


def test_injected_hotspots_recovered(hot_sim):
    an, segs = _analyse(hot_sim)
    hs = hotspots(an, B=200)
    validate(hs.select(list(HOTSPOT)), HOTSPOT)
    x0 = hot_sim.link_x0
    truth = [(x0[h.link] + h.lo, x0[h.link] + h.hi, h.kind) for h in hot_sim.hots]
    found = list(hs.select("x0_m", "x1_m", "kind").iter_rows())
    match = lambda a, b: a[0] < b[1] and b[0] < a[1]
    tp = [f for f in found if any(match(f, t) for t in truth)]
    precision = len(tp) / max(len(found), 1)
    recall = np.mean([any(match(f, t) for f in found) for t in truth])
    assert precision >= 0.9 and recall >= 0.9, (precision, recall, found)
    for f in tp:
        t = next(t for t in truth if match(f, t))
        assert f[2] == t[2], (f, t)
    assert (hs["zone"] == "running").all()
    assert (hs["ci_lo"] * 20 > 2).all() and (hs["persistence"] >= 0.6).all()
    peaks = hs.filter(pl.col("kind") == "congestion")["peak_hour"].to_list()
    assert all(h in (7, 8, 9, 16, 17, 18) for h in peaks)
    gj = to_geojson(hs, segs)
    json.dumps(gj)
    assert len(gj["features"]) == hs.height and all(f["geometry"]["type"] == "LineString" for f in gj["features"])


def test_no_hotspot_on_null_line():
    an, _ = _analyse(simulate(n_days=30, seed=7))
    assert hotspots(an, B=200).height <= 1


def test_bootstrap_weights_stratified():
    dow = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    W = bootstrap_weights(dow, 50, np.random.default_rng(0))
    for w in np.unique(dow):
        np.testing.assert_allclose(W[:, dow == w].sum(1), (dow == w).sum(), rtol=1e-5)


def test_bootstrap_ci_coverage():
    """95 % percentile CIs of excess_per_passage cover the truth in 90-98 % of bins (repeated sims)."""
    hots = [Hot(1, 150, 210, 14, tuple(range(6, 20)), "infrastructure"), Hot(3, 120, 180, 20, (7, 8, 17, 18), "congestion")]
    hit, n = 0, 0
    for seed in range(4):
        sim = simulate(n_days=30, seed=100 + seed, hots=hots)
        an, segs = _analyse(sim)
        st = bin_stats(an, 0, B=200, seed=seed)
        bf = bins_frame(st).filter(pl.col("hour").is_between(6, 19))
        for h in range(6, 20):
            tr = sim.truth_seg(segs, h)["excess"]
            b = bf.filter(pl.col("hour") == h).sort("seg_idx")
            lo, hi = b["ci_lo"].to_numpy(), b["ci_hi"].to_numpy()
            ok = np.isfinite(lo)
            hit += int(((lo <= tr) & (tr <= hi))[ok].sum())
            n += int(ok.sum())
    cov = hit / n
    assert 0.90 <= cov <= 0.98, cov
