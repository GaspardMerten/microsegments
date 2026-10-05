import datetime as dt
import time

import numpy as np
import polars as pl
import pytest

from microsegments import aggregate, metrics
from microsegments.hotspots import bin_stats
from microsegments.schema import CUBE, PLACED, conform, validate
from t4_fixtures import make_segments, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(n_days=12, seed=5, extra_false_rows=500)


def test_conservation(sim):
    cube = aggregate.count(sim.placed, sim.segment_fn(30))
    validate(cube, CUBE)
    assert cube["obs"].sum() == sim.placed["count"].sum()


def test_link_totals_invariant_to_length_and_phase(sim):
    ref = None
    for L, ph in [(30, 0), (30, 7.5), (30, 15), (10, 0), (75, 20), (100, 0)]:
        segs = sim.segment_fn(L, ph)
        t = aggregate.link_totals(aggregate.count(sim.placed, segs), segs)
        if ref is None:
            ref = t
        else:
            assert t.equals(ref)


def test_row_order_invariance(sim):
    segs = sim.segment_fn(20)
    a = aggregate.count(sim.placed, segs)
    b = aggregate.count(sim.placed.sample(fraction=1.0, shuffle=True, seed=1), segs)
    assert a.equals(b)


def _rows(pos, link, track=None, ts=None):
    n = len(pos)
    base = dt.datetime(2025, 3, 3, 8, tzinfo=dt.timezone.utc)
    return conform(pl.DataFrame({
        "ts": ts or [base] * n, "service_date": [dt.date(2025, 3, 3)] * n, "hour": [8] * n,
        "pattern_uid": ["P0"] * n, "link_idx": link, "pos_m": pos, "track_id": track or ["a"] * n, "count": [True] * n,
    }), PLACED)


def test_assign_bounds():
    segs = make_segments(np.array([90.0, 60.0]), ["A", "B"], "P0", 30)   # link 0: 0,30,60,90; link 1: 0,30,60
    df = aggregate.assign(_rows([30.0, 30.01, 90.0, 120.0, 0.0, -2.0, 0.0, 1e-3], [0, 0, 0, 0, 1, 1, 0, 1]), segs)
    assert df["seg_idx"].to_list() == [0, 1, 2, 2, 2, 2, 0, 3]


def test_unknown_link_dropped():
    segs = make_segments(np.array([90.0]), ["A"], "P0", 30)
    cube = aggregate.count(_rows([10.0, 10.0], [0, 5]), segs)
    assert cube["obs"].sum() == 1


def test_per_vehicle_thinning():
    base = dt.datetime(2025, 3, 3, 8, tzinfo=dt.timezone.utc)
    ts = [base + dt.timedelta(seconds=s) for s in (0, 5, 19, 21, 0, 3)]
    tr = ["a", "a", "a", "a", "b", None]
    segs = make_segments(np.array([90.0]), ["A"], "P0", 30)
    df = _rows([5.0, 35.0, 65.0, 70.0, 10.0, 10.0], [0] * 6, tr, ts)
    cube = aggregate.count(df, segs, tick_s=20, per_vehicle=True)
    assert cube["obs"].sum() == 4           # a: ticks 0 and 1, b: 1, anonymous: 1
    kept = aggregate.thin(df, 20)
    assert 65.0 in kept.filter(pl.col("track_id") == "a")["pos_m"].to_list()   # last fix of tick 0
    assert aggregate.count(df, segs, per_vehicle=False)["obs"].sum() == 6


def test_performance_3m_rows():
    """3 M placed rows (3 months x 1 line): count + analyse < 1 s + bootstrap B=200 < 10 s."""
    rng = np.random.default_rng(0)
    n = 3_000_000
    link_len = np.array([450.0, 380, 520, 410, 480, 500, 430, 470, 520, 400] * 2)
    keys = [f"S{i}" for i in range(len(link_len))]
    segs = make_segments(link_len, keys, "P0", 30)
    days = np.datetime64("2025-02-03") + np.arange(91)
    d = days[rng.integers(0, len(days), n)]
    li = rng.integers(0, len(link_len), n)
    placed = pl.DataFrame({
        "ts": (d.astype("datetime64[s]") + rng.integers(5 * 3600, 24 * 3600, n).astype("timedelta64[s]")).astype("datetime64[us]"),
        "service_date": d, "pattern_uid": np.full(n, "P0"), "link_idx": li.astype(np.int16),
        "pos_m": (rng.random(n) * link_len[li]).astype(np.float32), "track_id": None, "count": True,
    }).with_columns(pl.col("ts").dt.replace_time_zone("UTC"), pl.col("ts").dt.hour().cast(pl.Int8).alias("hour"))
    placed = conform(placed, PLACED)
    t = time.perf_counter()
    cube = aggregate.count(placed, segs)
    t_count = time.perf_counter() - t
    assert cube["obs"].sum() == n
    t = time.perf_counter()
    an = metrics.analyse(cube, None, None, segs, None)
    t_an = time.perf_counter() - t
    t = time.perf_counter()
    bin_stats(an, 0, B=200)
    t_boot = time.perf_counter() - t
    print(f"count {t_count:.2f}s analyse {t_an:.2f}s bootstrap {t_boot:.2f}s")
    assert t_count < 5 and t_an < 1.0 and t_boot < 10
