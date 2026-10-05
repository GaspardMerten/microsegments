import datetime as dt
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from gtfs_fixtures import (
    CUT,
    FULL,
    STOPS,
    street,
    two_versions,
    v1_patterns,
    write_feed,
)

from microsegments import schema
from microsegments.network import GtfsSource, KeyRegistry, build_network, classify
from microsegments.network.calendar import active_services, service_dates

D1, D2, D3 = dt.date(2025, 1, 1), dt.date(2025, 1, 11), dt.date(2025, 1, 20)


def days(a, b):
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


@pytest.fixture
def net2(tmp_path):
    src = GtfsSource(feeds=two_versions(tmp_path))
    return build_network(src, "55", days(D1, D3))


def test_schemas(net2):
    schema.validate(net2.patterns, schema.PATTERN)
    schema.validate(net2.pattern_days, schema.PATTERN_DAY)
    schema.validate(net2.links, schema.LINK)


def test_versions_and_dates(net2):
    v = net2.versions(0)
    assert len(v) == 2
    a, b = v.row(0, named=True), v.row(1, named=True)
    assert (a["first"], a["last"], a["n_days"]) == (D1, D2 - dt.timedelta(days=1), 10)
    assert (b["first"], b["last"], b["n_days"]) == (D2, D3, 10)
    # equal days: the latest version is displayed
    assert b["display"] and not a["display"]
    assert net2.pattern(a["pattern_uid"])["stop_ids"] == FULL
    assert net2.pattern(b["pattern_uid"])["stop_ids"] == CUT
    removed = {k.split("~")[0] for k in a["links_removed"]}
    added = {k.split("~")[0] for k in a["links_added"]}
    assert added == {"C>D", "E>F"} and removed == {"C>D"}
    assert len(net2.versions()) == 4
    assert net2.main(D1, 1) == net2.versions(1)["pattern_uid"][0]


def test_keys_stable_and_reroute(net2):
    v = net2.versions(0)
    k1 = {r["from_stop"] + r["to_stop"]: r["link_key"] for r in net2.links_of(v["pattern_uid"][0]).iter_rows(named=True)}
    k2 = {r["from_stop"] + r["to_stop"]: r["link_key"] for r in net2.links_of(v["pattern_uid"][1]).iter_rows(named=True)}
    for pair in ("AB", "BC", "DE"):
        assert k1[pair] == k2[pair]
    assert k1["CD"] != k2["CD"]
    assert k1["CD"].startswith("C>D~") and k2["CD"].startswith("C>D~")
    l2 = net2.links_of(v["pattern_uid"][1]).filter(pl.col("from_stop") == "C")["len_m"][0]
    assert l2 == pytest.approx(250 + 240 - 8, abs=1)   # stops sit 4 m up the side legs


def test_short_and_detour(net2):
    p = net2.patterns.filter(pl.col("direction_id") == 0)
    main = net2.versions(0)["pattern_uid"][0]
    short = p.filter(pl.col("stop_ids").list.len() == 4).row(0, named=True)
    assert short["kind"] == "short" and short["parent_uid"] == main and short["parent_offset"] == 0
    det = p.filter(pl.col("stop_ids").list.first() == "A").filter(pl.col("stop_ids").list.get(1) == "C").row(0, named=True)
    assert det["kind"] == "detour"
    # the short working shares its links' keys with the main pattern
    km = set(net2.links_of(main)["link_key"])
    assert set(net2.links_of(short["pattern_uid"])["link_key"]) <= km
    d = net2.patterns_on(D1)
    assert d.filter(pl.col("is_main"))["pattern_uid"].n_unique() == 2
    assert d.filter(pl.col("pattern_uid") == short["pattern_uid"])["n_trips"][0] == 4


def test_classify():
    m = list("ABCDEF")
    assert classify(m, m) == ("main", None)
    assert classify(list("CDE"), m) == ("short", 2)
    assert classify(list("ACDEF"), m) == ("detour", None)
    assert classify(list("XYZ"), m) == ("other", None)


def test_link_days(net2):
    ld = net2.link_days()
    cd = ld.filter(pl.col("link_key").str.starts_with("E>F") & (pl.col("direction_id") == 0))
    assert sorted(cd["service_date"].to_list()) == days(D1, D2 - dt.timedelta(days=1))
    ab = ld.filter(pl.col("link_key").str.starts_with("A>B"))
    assert ab["service_date"].n_unique() == 20


def test_reissued_feed_keeps_keys(tmp_path):
    p1 = write_feed(tmp_path / "a", v1_patterns(), start=D1, end=D3)
    p2 = write_feed(tmp_path / "b", v1_patterns(), start=D1, end=D3)
    n1 = build_network(str(p1), "55", [D1])
    # no shared registry: keys come from geometry hashes, identical for an identical feed
    n2 = build_network(str(p2), "55", [D1])
    assert n1.links["link_key"].to_list() == n2.links["link_key"].to_list()
    assert n1.patterns["pattern_uid"].to_list() == n2.patterns["pattern_uid"].to_list()


def test_registry_jitter_and_parquet(tmp_path):
    reg = KeyRegistry()
    pats = v1_patterns()
    n1 = build_network(str(write_feed(tmp_path / "a", pats, start=D1, end=D3)), "55", [D1], registry=reg)
    # same streets, shapes shifted 5 m north: same physical links -> same keys via the registry
    jit = []
    for p in pats:
        q = dict(p)
        q["shape"] = [(lon, lat + 5 / 111195.0) for lon, lat in p["shape"]]
        jit.append(q)
    f = tmp_path / "reg.parquet"
    reg.to_parquet(f)
    reg2 = KeyRegistry.from_parquet(f)
    assert len(reg2) == len(reg)
    n2 = build_network(str(write_feed(tmp_path / "b", jit, start=D1, end=D3)), "55", [D1], registry=reg2)
    assert sorted(n1.links["link_key"].unique()) == sorted(n2.links["link_key"].unique())
    # without the registry, the jittered geometry hashes differently
    n3 = build_network(str(tmp_path / "b"), "55", [D1])
    assert set(n3.links["link_key"]) != set(n1.links["link_key"])


def test_registry_thresholds():
    reg = KeyRegistry()
    base = street(list("AB"))
    k = reg.key_for("A", "B", base)
    assert reg.key_for("A", "B", base) == k
    # 15 m longer but same corridor -> within max(5 %, 20 m) and Hausdorff 20 m
    longer = base + [(base[-1][0] + 15 / 70200, base[-1][1])]
    assert reg.key_for("A", "B", longer) == k
    # 60 m detour north: Hausdorff 60 m -> new key
    from gtfs_fixtures import densify
    detour = densify([(0, 0), (150, 60), (310, 0)])
    assert reg.key_for("A", "B", detour) != k


def test_calendar_dates_only_and_route_id(tmp_path):
    p = write_feed(tmp_path / "cd", v1_patterns(), start=D1, end=D1 + dt.timedelta(days=4), calendar=False)
    src = GtfsSource(p, route_key="route_id")
    feed = src.feed_for(D1, "R55")
    assert feed.calendar is None and active_services(feed, D1) == {"WK"}
    assert service_dates(feed) == (D1, D1 + dt.timedelta(days=4))
    assert len(src.active_trips(D1, "R55")) == 46
    assert len(src.active_trips(D1 + dt.timedelta(days=9), "R55")) == 0
    # route_key fallback: a short name works too
    net = build_network(src, "55", days(D1, D1 + dt.timedelta(days=9)))
    assert len(net.dates) == 5 and len(net.missing_dates) == 5


def test_removed_date_and_dated_template(tmp_path):
    p = write_feed(tmp_path / "2025-01-01", v1_patterns(), start=D1, end=D3, extra_dates=[(D1 + dt.timedelta(days=2), 2)])
    src = GtfsSource(dated=str(tmp_path / "{date}"))
    assert src.feed_path(D1 + dt.timedelta(days=5)) == str(p)      # falls back to the latest snapshot
    assert src.feed_path(D1 - dt.timedelta(days=1)) is None
    net = build_network(src, "55", days(D1, D1 + dt.timedelta(days=4)))
    assert D1 + dt.timedelta(days=2) in net.missing_dates


def test_missing_shapes(tmp_path):
    p = write_feed(tmp_path / "ns", v1_patterns(), start=D1, end=D3, shapes=False)
    net = build_network(str(p), "55", [D1])
    main = net.main(D1, 0)
    lk = net.links_of(main)
    exp = np.diff([STOPS[s] for s in FULL])
    assert np.allclose(lk["len_m"].to_numpy(), exp, atol=0.5)


# ---------------------------------------------------------------- real data (prototype)
PROTO = Path(os.environ.get("MS_PROTOTYPE_DATA", Path.home() / "Documents/Dev/CoDE/BrusselsTransitStuck"))
GTFS = PROTO / "data2025/gtfs"


def digits(stop_id):
    m = re.match(r"0*(\d+)", str(stop_id))
    return int(m.group(1)) if m else None


@pytest.mark.skipif(not (GTFS / "trips.parquet").exists(), reason="prototype GTFS not available")
def test_route55_matches_prototype():
    src = GtfsSource(str(GTFS))
    feed = src.load(str(GTFS), "55")
    lo, hi = service_dates(feed)
    net = build_network(src, "55", days(lo, hi))
    # prototype: most frequent digits() stop sequence over all trips, per direction_id
    st = feed.stop_times.sort("trip_id", "stop_sequence")
    seqs = {t: tuple(digits(s) for s in g["stop_id"]) for (t,), g in st.group_by("trip_id")}
    proto_out = PROTO / "out/speed55_2025.json"
    proto = json.loads(proto_out.read_text()) if proto_out.exists() else None
    for d in (0, 1):
        trips = feed.trips.filter(pl.col("direction_id") == d)["trip_id"].to_list()
        expected = Counter(seqs[t] for t in trips if t in seqs).most_common(1)[0][0]
        uid = net.display_pattern(d)
        assert tuple(digits(s) for s in net.pattern(uid)["stop_ids"]) == expected
        if proto:
            got = net.links_of(uid)["len_m"].to_numpy()
            exp = np.array([lk["len"] for lk in proto["dirs"][d]["links"]])
            assert np.abs(got - exp).max() <= 1.0      # json lengths are rounded to the metre
