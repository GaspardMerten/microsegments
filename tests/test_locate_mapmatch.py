"""place_latlon on simulated GTFS-RT data, and STIB-format vs GTFS-RT-format counts."""
import polars as pl
import pytest

from microsegments.config import Config
from microsegments.locate import place, place_latlon
from microsegments.locate.resample import resample
from microsegments.network import build_network
from microsegments.simulate import simulate

NULLS = {"trip_id": pl.Utf8, "direction_id": pl.Int8, "bearing": pl.Float32, "vehicle_id": pl.Utf8}


@pytest.fixture(scope="module")
def sim_net():
    sim = simulate(seed=1, days=3, mode="gtfsrt", change_day=2)
    net = build_network(str(sim.gtfs_dir), "S1", sim.dates)
    return sim, net


def _strip(obs, *cols):
    return obs.with_columns([pl.lit(None, NULLS[c]).alias(c) for c in cols])


def _check(sim, net, r):
    first = net.patterns.select("pattern_uid", "direction_id", pl.col("stop_ids").list.first().alias("sid")).join(
        sim.stops.select("direction_id", pl.col("stop_id").alias("sid"), pl.col("s_m").alias("off")),
        on=["direction_id", "sid"])
    c = r.counted.join(first.select("pattern_uid", "off"), on="pattern_uid")
    err = (c["s_m"].cast(pl.Float64) + c["off"] - c["true_s_m"]).abs()
    return c.height, (err <= 30).mean(), (c["direction_id"] == c["true_direction_id"]).mean()


@pytest.mark.parametrize("strip", [(), ("trip_id",), ("trip_id", "direction_id", "bearing")],
                         ids=["full", "no_trip", "vehicle_only"])
def test_placement(sim_net, strip):
    sim, net = sim_net
    r = place_latlon(_strip(sim.observations, *strip), net, tz=sim.tz)
    n, within, direction = _check(sim, net, r)
    assert n > 0.5 * sim.observations.height
    assert within >= 0.98
    assert direction >= 0.99
    assert r.report["total"] == sim.observations.height
    if strip:   # without trip / direction, unresolved rows are reported, not guessed
        assert r.report["by_trip"] == 0


def test_nothing_known_is_ambiguous(sim_net):
    sim, net = sim_net
    r = place_latlon(_strip(sim.observations, *NULLS), net, tz=sim.tz)
    assert r.report["ambiguous"] > 0.9 * sim.observations.height
    assert r.report["counted"] < 0.05 * sim.observations.height


def test_change_day_version(sim_net):
    sim, net = sim_net
    r = place(sim.observations, net, Config.from_dict({"input": {"kind": "latlon", "timezone": sim.tz}}))
    for d in sim.dates:
        day = set(net.pattern_days.filter(pl.col("service_date") == d)["pattern_uid"].to_list())
        used = set(r.counted.filter(pl.col("service_date") == d)["pattern_uid"].unique().to_list())
        assert used and used <= day, d


def test_stib_vs_gtfsrt_link_totals():
    """Same service, two feed formats: per-link counted totals agree within 3% once the per-vehicle
    feed is resampled onto the 20 s poll grid. (Short workings are off: the STIB rules for them,
    e.g. no layover drop under 100 s, differ by design.)"""
    kw = dict(seed=3, days=3, outages=[], frozen=[], drop_rate=0.0, short_working=False)
    a = simulate(mode="stib", **kw)
    b = simulate(mode="gtfsrt", gtfsrt_interval=(5.0, 10.0), **kw)
    net = build_network(str(a.gtfs_dir), "S1", a.dates)
    ra = place(a.observations, net, Config.from_dict({"input": {"kind": "stib", "timezone": a.tz}}))
    rb = place(b.observations, net, Config.from_dict({"input": {"kind": "latlon", "timezone": b.tz}}))
    fb = resample(rb, 20.0)

    def per_link(f):
        return f.filter("count").group_by("link_key").agg(pl.len().alias("n"))

    j = per_link(ra.frame).join(per_link(fb), on="link_key", suffix="_rt", how="full", coalesce=True).fill_null(0)
    rel = (j["n_rt"] / j["n"] - 1).abs()
    assert rel.max() <= 0.03, j.with_columns(rel=rel).sort("rel")
