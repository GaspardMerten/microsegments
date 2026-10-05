"""Anonymous tracking and passages against simulator truth."""
import polars as pl
import pytest

from microsegments.config import Config
from microsegments.locate import place, place_latlon
from microsegments.network import build_network
from microsegments.simulate import simulate
from microsegments.tracks import passages, track_anonymous


def _link_map(net, sim):
    st = sim.stops.select("direction_id", pl.col("stop_id").alias("from_stop"), pl.col("idx").cast(pl.Int16).alias("full_link"))
    lk = net.links.join(net.patterns.select("pattern_uid", "direction_id"), on="pattern_uid").join(st, on=["direction_id", "from_stop"])
    return lk.select("link_key", "direction_id", "full_link").unique()


def _per_link(P, net, sim, col):
    ours = P.join(_link_map(net, sim), on="link_key").group_by("direction_id", "full_link").agg(pl.col(col).sum())
    tr = sim.passages.group_by("direction_id", pl.col("link_idx").alias("full_link")).agg(pl.col("n").sum().alias("truth"))
    return tr.join(ours, on=["direction_id", "full_link"], how="left").fill_null(0)


@pytest.fixture(scope="module")
def stib():
    sim = simulate(seed=0, days=3, mode="stib")
    net = build_network(str(sim.gtfs_dir), "S1", sim.dates)
    res = place(sim.observations, net, Config.from_dict({"input": {"kind": "stib", "timezone": sim.tz}}))
    return sim, net, res


def test_anonymous_tracks(stib):
    sim, net, res = stib
    f = res.frame
    assert f["track_id"].null_count() == 0
    # a track never mixes two true vehicles for long: dominant vehicle share per track
    if "true_vehicle_id" in f.columns:   # counted rows (terminus layover rows are interchangeable)
        share = (f.filter("count").group_by("track_id", "true_vehicle_id").len()
                 .group_by("track_id").agg(pl.col("len").max().alias("top"), pl.col("len").sum().alias("all")))
        assert share["top"].sum() / share["all"].sum() >= 0.95


def test_track_anonymous_idempotent_shape(stib):
    sim, net, res = stib
    again = track_anonymous(res.frame.with_columns(pl.lit(None, pl.Utf8).alias("track_id")))
    assert again.height == res.frame.height
    assert again["track_id"].null_count() == 0


def test_anonymous_feed_passages_within_3pct(stib):
    sim, net, res = stib
    P = passages(res, net, tz=sim.tz, mode="feed")
    t = _per_link(P, net, sim, "n_feed")
    rel = (t["n_feed"] / t["truth"] - 1).abs()
    assert rel.max() <= 0.03, t.with_columns(rel=rel)
    assert (P["n"] == P["n_feed"]).all()


def test_event_passages_exact_and_max(stib):
    sim, net, res = stib
    P = passages(res, net, events=sim.stop_events, tz=sim.tz)
    t = _per_link(P, net, sim, "n_events")
    assert (t["n_events"] == t["truth"]).all()
    assert P.select((pl.col("n") == pl.max_horizontal(pl.col("n_feed").fill_null(0),
                                                       pl.col("n_events").fill_null(0))).all()).item()
    assert P.select(pl.struct("service_date", "hour", "link_key").is_unique().all()).item()


def test_vehicle_id_passages():
    # no feed outage: a trip run entirely inside one is not observable
    sim = simulate(seed=1, days=2, mode="gtfsrt", outages=[])
    net = build_network(str(sim.gtfs_dir), "S1", sim.dates)
    res = place_latlon(sim.observations, net, tz=sim.tz)
    P = passages(res, net, tz=sim.tz, mode="feed")
    t = _per_link(P, net, sim, "n_feed")
    rel = (t["n_feed"] / t["truth"] - 1).abs()
    assert rel.max() <= 0.03, t.with_columns(rel=rel)
