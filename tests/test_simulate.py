import time

import numpy as np
import polars as pl
import pytest

from microsegments import schema
from microsegments.io import add_local_time, coverage
from microsegments.simulate import simulate


@pytest.fixture(scope="module")
def stib(tmp_path_factory):
    return simulate(seed=0, days=3, mode="stib", out_dir=tmp_path_factory.mktemp("s"), change_day=2,
                    outages=[(0, "08:10", 20)], frozen=[(1, "10:00", 8)], terminus_alias={1: "9943"})


def test_determinism(tmp_path):
    a = simulate(seed=7, days=2, out_dir=tmp_path / "a")
    b = simulate(seed=7, days=2, out_dir=tmp_path / "b")
    c = simulate(seed=8, days=2, out_dir=tmp_path / "c")
    assert a.observations.equals(b.observations) and a.truth_seconds.equals(b.truth_seconds)
    assert (tmp_path / "a/stop_times.txt").read_text() == (tmp_path / "b/stop_times.txt").read_text()
    assert not a.observations.equals(c.observations)


def test_speed(tmp_path):
    t = time.time()
    simulate(seed=0, days=10, out_dir=tmp_path)
    assert time.time() - t < 10


def test_schemas_and_gtfs(stib):
    schema.validate(stib.observations.select(list(schema.OBSERVATION)), schema.OBSERVATION)
    schema.validate(stib.snapshots, schema.SNAPSHOT)
    schema.validate(stib.stop_events, schema.STOP_EVENT)
    for f in ("stops", "routes", "trips", "stop_times", "shapes", "calendar"):
        assert (stib.gtfs_dir / f"{f}.txt").exists()
    cal = pl.read_csv(stib.gtfs_dir / "calendar.txt")
    assert cal["service_id"].to_list() == ["BASE", "CUT"]
    assert set(stib.patterns["name"]) == {"d0_main", "d1_main", "d0_short", "d1_short", "d0_cut", "d1_cut"}
    assert {h["kind"] for h in stib.hotspots} == {"signal", "queue", "holding"}
    links = pl.read_csv(stib.gtfs_dir / "stops.txt")
    assert len(links) == 18


def test_truth_consistency(stib):
    ev = stib.stop_events.sort("trip_key", "stop_sequence").with_columns(t=pl.coalesce("departure", "arrival"))
    ev = ev.with_columns(nt=pl.col("t").shift(-1).over("trip_key")).drop_nulls("nt")
    link_time = (ev["nt"] - ev["t"]).dt.total_microseconds().sum() / 1e6
    assert abs(stib.truth_seconds["seconds"].sum() - link_time) < 1e-3 * link_time
    # per link: truth seconds over its 1-m bins == sum of (departure B - departure A)
    tl = stib.truth_link_seconds().group_by("direction_id", "link_idx").agg(pl.col("seconds").sum())
    trips = stib.trips.select("trip_key", "direction_id")
    ev = ev.join(trips, on="trip_key")
    stops = stib.stops.select("direction_id", "stop_id", pl.col("idx").alias("link_idx"))
    per = (ev.join(stops, on=["direction_id", "stop_id"])
           .group_by("direction_id", "link_idx")
           .agg(((pl.col("nt") - pl.col("t")).dt.total_microseconds() / 1e6).sum().alias("ev")))
    j = tl.join(per, on=["direction_id", "link_idx"])
    assert len(j) == 16
    assert ((j["seconds"] - j["ev"]).abs() / j["ev"]).max() < 1e-6
    # passages: one per trip and link
    n_links = stib.passages.group_by("direction_id", "link_idx").agg(pl.col("n").sum())
    assert n_links["n"].sum() == len(ev)


def test_hotspots_visible_in_truth(stib):
    t = stib.truth_seconds
    sig = next(h for h in stib.hotspots if h["kind"] == "signal" and h["direction_id"] == 0)
    s = int(sig["s0_m"])
    near = t.filter((pl.col("direction_id") == 0) & (pl.col("s_bin") == s - 1))["seconds"].sum()
    far = t.filter((pl.col("direction_id") == 0) & (pl.col("s_bin") == s - 40))["seconds"].sum()
    assert near > 3 * far
    q = next(h for h in stib.hotspots if h["kind"] == "queue")
    mid = int((q["s0_m"] + q["s1_m"]) / 2)
    qq = t.filter((pl.col("direction_id") == 0) & (pl.col("s_bin") == mid))
    peak = qq.filter(pl.col("hour") == 8)["seconds"].sum() / stib.passages.filter(
        (pl.col("direction_id") == 0) & (pl.col("link_idx") == q["link_idx"]) & (pl.col("hour") == 8))["n"].sum()
    off = qq.filter(pl.col("hour") == 12)["seconds"].sum() / stib.passages.filter(
        (pl.col("direction_id") == 0) & (pl.col("link_idx") == q["link_idx"]) & (pl.col("hour") == 12))["n"].sum()
    assert peak > 2 * off


def test_stib_rows_match_truth(stib):
    o = stib.observations.join(stib.stops.select("direction_id", "stop_id", "s_m").rename(
        {"direction_id": "true_direction_id"}), on=["true_direction_id", "stop_id"], how="left")
    assert o["s_m"].null_count() == 0
    est = o["s_m"] + o["dist_m"] / 1.03
    err = (est - o["true_s_m"]).abs()
    assert float(err.quantile(0.99)) < 20
    assert set(o.filter(pl.col("true_direction_id") == 1)["terminus_id"]) == {"9943"}
    assert o["vehicle_id"].is_null().all()


def test_stib_outage_and_frozen(stib):
    c = coverage(stib.snapshots, stib.tz, gap_cap_s=40)
    d0 = c.filter((pl.col("service_date") == stib.dates[0]) & (pl.col("hour") == 8)).row(0, named=True)
    assert 3600 - 1200 - 30 < d0["covered_s"] < 3600 - 1200 + 60
    d1 = c.filter((pl.col("service_date") == stib.dates[1]) & (pl.col("hour") == 10)).row(0, named=True)
    assert abs(d1["frozen_s"] - 480) < 30 and d1["flags"] & schema.Flag.FROZEN
    # no observation during the outage nor at frozen polls
    fr = stib.snapshots.filter(pl.col("frozen"))["ts"]
    assert not stib.observations["ts"].is_in(fr.implode()).any()
    a, b = stib.outages[0]
    e = stib.observations["ts"].dt.epoch("s").to_numpy()
    assert not ((e >= a) & (e < b)).any()


def test_gtfsrt_mode(tmp_path):
    r = simulate(seed=1, days=1, mode="gtfsrt", out_dir=tmp_path, gtfsrt_interval=(1, 5), outages=[], frozen=[])
    o = r.observations
    assert o["vehicle_id"].null_count() == 0 and o["lat"].null_count() == 0
    gaps = o.sort("vehicle_id", "ts").with_columns(
        g=pl.col("ts").diff().over("vehicle_id").dt.total_seconds()).drop_nulls("g")["g"]
    assert gaps.median() <= 5
    # each vehicle runs one trip at a time
    per = o.group_by("vehicle_id", "ts").agg(pl.col("trip_id").n_unique())
    assert per["trip_id"].max() == 1
    loc = add_local_time(o, r.tz)
    assert loc["hour"].min() >= 4 and loc["hour"].max() <= 25
