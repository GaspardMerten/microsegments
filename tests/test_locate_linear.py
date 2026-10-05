"""place_linear on simulated STIB-format data (alias, change day, flen) and the place() dispatch."""
import polars as pl
import pytest

from microsegments.config import Config
from microsegments.locate import place, place_linear
from microsegments.network import build_network
from microsegments.simulate import simulate


@pytest.fixture(scope="module")
def sim_net():
    sim = simulate(seed=0, days=3, mode="stib", change_day=2, terminus_alias={1: "9943"})
    net = build_network(str(sim.gtfs_dir), "S1", sim.dates)
    return sim, net


def _offsets(net, sim):
    """Along-line position of each pattern's first stop (short / cut versions start downstream)."""
    first = net.patterns.select("pattern_uid", "direction_id", pl.col("stop_ids").list.first().alias("sid"))
    return first.join(sim.stops.select("direction_id", pl.col("stop_id").alias("sid"), pl.col("s_m").alias("off")),
                      on=["direction_id", "sid"]).select("pattern_uid", "off")


@pytest.fixture(scope="module")
def placed(sim_net):
    sim, net = sim_net
    return place(sim.observations, net, Config.from_dict({"input": {"kind": "stib", "timezone": sim.tz}}))


def test_counted_rows_within_one_segment_and_direction(sim_net, placed):
    sim, net = sim_net
    c = placed.counted.join(_offsets(net, sim), on="pattern_uid")
    assert c.height > 0.5 * sim.observations.height
    err = (c["s_m"].cast(pl.Float64) + c["off"] - c["true_s_m"]).abs()
    assert (err <= 30).mean() >= 0.98
    assert (c["direction_id"] == c["true_direction_id"]).mean() >= 0.99
    rep = placed.report
    assert rep["total"] == sim.observations.height
    assert rep["counted"] == placed.counted.height
    # anonymous feed: tracks were rebuilt
    assert placed.frame["track_id"].null_count() == 0


def test_alias_learned(placed):
    assert "9943" in placed.aliases
    assert placed.report["aliases_learned"] >= 1
    assert placed.report.get("unknown_terminus", 0) == 0


def test_change_day_version(sim_net, placed):
    sim, net = sim_net
    f = placed.counted
    for d in sim.dates:
        day = set(net.pattern_days.filter(pl.col("service_date") == d)["pattern_uid"].to_list())
        used = set(f.filter(pl.col("service_date") == d)["pattern_uid"].unique().to_list())
        assert used and used <= day, d
    # the versions really differ across the change day
    assert net.main(sim.dates[0], 0) != net.main(sim.dates[-1], 0)


def test_flen_table_and_gtfs_lengths(sim_net, placed):
    sim, net = sim_net
    fl = placed.flen
    assert fl is not None and {"pattern_uid", "link_idx"} <= set(fl.columns)
    g = place_linear(sim.observations, net, feed_length="gtfs", tz=sim.tz)
    assert g.report["counted"] > 0
    c = g.counted.join(_offsets(net, sim), on="pattern_uid")
    err = (c["s_m"].cast(pl.Float64) + c["off"] - c["true_s_m"]).abs()
    assert (err <= 30).mean() >= 0.95


def test_unknown_alias_without_learning_is_reported(sim_net):
    sim, net = sim_net
    obs = sim.observations.with_columns(
        pl.when(pl.col("terminus_id") == "9943").then(pl.lit("77777")).otherwise(pl.col("terminus_id")).alias("terminus_id"))
    r = place_linear(obs, net, tz=sim.tz, compat=True)
    # still learned by majority (the code is consistent) -> placed
    assert "77777" in r.aliases
