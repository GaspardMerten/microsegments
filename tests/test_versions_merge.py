"""Versions: patterns that differ only by small shape edits (same link keys) are one version."""
import datetime as dt

import polars as pl

from microsegments.aggregate import count
from microsegments.config import Params, Select
from microsegments.metrics import analyse
from microsegments.network.keys import KeyRegistry
from microsegments.network.patterns import Network
from microsegments.report import to_contract
from microsegments.schema import PATTERN_DAY, conform

from t4_fixtures import simulate


def _days(n, start=dt.date(2025, 3, 3)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def test_network_versions_merge_same_link_sequence():
    days = _days(12)
    # P0 / P0b / P0c: same links (shape edits); P1: the last link cut
    main = ["P0"] * 4 + ["P0b"] * 3 + ["P0c"] * 2 + ["P1"] * 3
    pd = conform(pl.DataFrame([(d, "R", 0, u, 100, True) for d, u in zip(days, main)], schema=list(PATTERN_DAY),
                              orient="row"), PATTERN_DAY)
    keys = ["A>B~1", "B>C~1", "C>D~1"]
    links = pl.DataFrame([(u, i, k) for u in ("P0", "P0b", "P0c") for i, k in enumerate(keys)]
                         + [("P1", i, k) for i, k in enumerate(keys[:2])],
                         schema={"pattern_uid": pl.Utf8, "link_idx": pl.Int16, "link_key": pl.Utf8}, orient="row")
    net = Network(route="R", patterns=pl.DataFrame(), pattern_days=pd, links=links, stops=pl.DataFrame(),
                  registry=KeyRegistry())
    v = net.versions()
    assert v.height == 2
    assert v["n_days"].to_list() == [9, 3]
    assert v["display"].to_list() == [True, False]
    assert v["pattern_uid"][0] == "P0" and v["links_removed"][1].to_list() == ["C>D~1"]
    assert net.display_pattern(0) == "P0"


def test_analysis_versions_and_contract_merge():
    sim = simulate(n_days=12, seed=0, cut_days=range(9, 12))
    segs = sim.segment_fn(30.0)
    # days 0-5 main P0, days 6-8 main P0b (same links), days 9-11 P1 (cut)
    segs = pl.concat([segs, segs.filter(pl.col("pattern_uid") == "P0").with_columns(pl.lit("P0b").alias("pattern_uid"))])
    b_days = set(sim.days[6:9])
    pd = sim.pattern_days.with_columns(pl.when(pl.col("service_date").is_in(b_days) & (pl.col("pattern_uid") == "P0"))
                                       .then(pl.lit("P0b")).otherwise(pl.col("pattern_uid")).alias("pattern_uid"))
    cube = count(sim.placed, segs, 20.0, per_vehicle=False)
    an = analyse(cube, sim.coverage, sim.passages, segs, pd, Select(), Params())
    assert an.versions.height == 2
    disp = an.versions.filter(pl.col("display"))
    assert disp["n_days"][0] == 9 and disp["pattern_uid"][0] == "P0"
    c = to_contract(an)
    assert c["dirs"][0]["versions"] == ["P0", "P1"] and len(c["dirs"][0]["alt"]) == 1
