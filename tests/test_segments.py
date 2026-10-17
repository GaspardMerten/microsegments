import datetime as dt

import numpy as np
import polars as pl
import pytest
from gtfs_fixtures import v1_patterns, write_feed
from hypothesis import given, settings
from hypothesis import strategies as st

from microsegments import schema
from microsegments.network import build_network
from microsegments.segments import (
    adaptive_breaks,
    assign,
    assign_frame,
    link_edges,
    segment,
)

D1 = dt.date(2025, 1, 1)


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    p = write_feed(tmp_path_factory.mktemp("g") / "f", v1_patterns(), start=D1, end=D1)
    return build_network(str(p), "55", [D1])


@settings(max_examples=300, deadline=None)
@given(st.floats(0.5, 2000), st.sampled_from([10, 15, 20, 30, 50, 100]), st.floats(0, 1),
       st.sampled_from(["equal", "fixed"]))
def test_edges_partition(len_m, L, frac, grid):
    e = link_edges(len_m, L, frac * L, grid)
    assert e[0] == 0 and e[-1] == len_m
    assert np.all(np.diff(e) > 0)
    w = np.diff(e)
    if len(w) > 1:
        assert w.min() >= 1.0 - 1e-9
    if grid == "fixed":
        assert w.max() <= L + 1.0 + 1e-6      # a merged sliver adds < 1 m
    else:
        n = max(1, round(len_m / L))
        assert w.max() <= len_m / n + 1.0 + 1e-6


def test_equal_and_fixed_grids():
    e = link_edges(253.0, 30, 0, "equal")
    assert len(e) - 1 == 8 and np.allclose(np.diff(e), 253 / 8)
    # phase shifts the grid, remainder at both ends
    e = link_edges(253.0, 30, 15, "equal")
    w = np.diff(e)
    assert len(w) == 9 and w[0] == pytest.approx(253 / 16) and w[-1] == pytest.approx(253 / 16)
    # fixed = prototype buckets: arange(0, L, 30) + [L], a sliver < 1 m joins the previous bucket
    for L in (253.0, 240.5, 300.0, 15.0):
        proto = list(np.arange(0, L, 30.0)) + [L]
        if len(proto) > 2 and proto[-1] - proto[-2] < 1:
            proto.pop(-2)
        assert np.allclose(link_edges(L, 30, 0, "fixed"), proto)


@pytest.mark.parametrize("grid", ["equal", "fixed"])
@pytest.mark.parametrize("L,phase", [(30, 0), (30, 7.5), (30, 15), (20, 13), (100, 50)])
def test_segment_lengths_sum(net, grid, L, phase):
    seg = segment(net, L, phase, grid)
    schema.validate(seg, schema.SEGMENT)
    tot = seg.group_by("pattern_uid", "link_idx").agg(pl.col("len_m").sum())
    j = tot.join(net.links.select("pattern_uid", "link_idx", "len_m"), on=["pattern_uid", "link_idx"], suffix="_l")
    assert len(j) == len(net.links)
    assert np.allclose(j["len_m"], j["len_m_l"], atol=1e-6)
    # x0 continuity along each pattern, seg_idx contiguous
    for (_,), g in seg.sort("seg_idx").group_by("pattern_uid"):
        g = g.sort("seg_idx")
        assert g["seg_idx"].to_list() == list(range(len(g)))
        assert np.allclose(g["x0_m"][1:].to_numpy(), (g["x0_m"] + g["len_m"])[:-1].to_numpy(), atol=1e-6)
    k = seg["seg_key"][0]
    assert k.startswith(seg["link_key"][0] + "/0@") and k.endswith(f"@{L:g}:{phase:g}" + ("f" if grid == "fixed" else ""))
    assert seg["seg_key"].n_unique() == seg.select("link_key", "k").n_unique()


def test_geometry_cut(net):
    seg = segment(net, 30)
    from microsegments.network.geometry import length
    lens = np.array([length(g) for g in seg["geometry"].to_list()])
    assert np.allclose(lens, seg["len_m"].to_numpy(), atol=0.05)


def test_zone_tagging():
    links = pl.DataFrame({
        "pattern_uid": ["p", "p"], "link_idx": [0, 1], "link_key": ["A>B~x", "B>C~y"],
        "from_stop": ["A", "B"], "to_stop": ["B", "C"], "from_name": ["A", "B"], "to_name": ["B", "C"],
        "s0_m": [0.0, 300.0], "len_m": [300.0, 300.0], "geometry": [None, None]}, schema=schema.LINK)
    seg = segment(links, 30, 0, "equal", stop_zone=(30, 60))
    z = seg.filter(pl.col("link_idx") == 0)["zone"].to_list()
    # [0, 60] after A -> bins 0, 1; [270, 300] before B -> bin 9
    assert z == ["stop", "stop"] + ["running"] * 7 + ["stop"]
    z1 = seg.filter(pl.col("link_idx") == 1)["zone"].to_list()
    assert z1 == ["stop", "stop"] + ["running"] * 7 + ["stop"]
    seg = segment(links, 30, 0, "equal", stop_zone=(45, 0))
    assert seg.filter(pl.col("link_idx") == 0)["zone"].to_list() == ["running"] * 8 + ["stop"] * 2
    # a short link entirely in the zone
    short = links.with_columns(len_m=pl.Series([50.0, 300.0]), s0_m=pl.Series([0.0, 50.0]))
    assert set(segment(short, 30)["zone"].head(2)) == {"stop"}


def test_stops_grid():
    links = pl.DataFrame({
        "pattern_uid": ["p", "p", "p"], "link_idx": [0, 1, 2], "link_key": ["A>B~x", "B>C~y", "C>D~z"],
        "from_stop": ["A", "B", "C"], "to_stop": ["B", "C", "D"], "from_name": ["A", "B", "C"], "to_name": ["B", "C", "D"],
        "s0_m": [0.0, 300.0, 370.0], "len_m": [300.0, 70.0, 200.0], "geometry": [None, None, None]}, schema=schema.LINK)
    seg = segment(links, 30, 0, "stops", stop_zone=(36, 38))
    l0 = seg.filter(pl.col("link_idx") == 0)
    # [0, 38] zone after A, 226 m running in 8 bins of 28.25 m, [264, 300] zone before B
    assert l0["zone"].to_list() == ["stop"] + ["running"] * 8 + ["stop"]
    assert np.allclose(l0["len_m"].to_numpy(), [38] + [226 / 8] * 8 + [36])
    assert l0["seg_key"][0] == "A>B~x/0@30:0s38_36"
    # 70 m < 38 + 36: one stop bin
    assert seg.filter(pl.col("link_idx") == 1)["zone"].to_list() == ["stop"]
    # per-link zones: the pattern's own row first, then the link's row, then the defaults
    z = pl.DataFrame({"pattern_uid": ["p", "q"], "link_key": ["C>D~z", "A>B~x"], "head_m": [44.0, 50.0]})
    seg = segment(links, 30, 0, "stops", stop_zone=(36, 38), zones=z)
    assert seg.filter(pl.col("link_idx") == 2)["len_m"][0] == pytest.approx(44)
    assert seg.filter(pl.col("link_idx") == 0)["len_m"][0] == pytest.approx(50)
    with pytest.raises(ValueError):
        segment(links, 30, 0, "equal", zones=z)
    # no zone at all: plain equal bins
    assert segment(links, 30, 0, "stops", stop_zone=(0, 0))["zone"].unique().to_list() == ["running"]


def test_approach():
    from microsegments.locate.linear import approach
    from microsegments.schema import Flag
    links = pl.DataFrame({"pattern_uid": ["p", "p"], "link_idx": [0, 1], "len_m": [300.0, 200.0]})
    placed = pl.DataFrame({"pattern_uid": ["p", "p", "p", "p"], "link_idx": pl.Series([0, 0, 0, 1], dtype=pl.Int16),
                           "pos_m": pl.Series([150.0, 300.0, 300.0, 100.0], dtype=pl.Float32),
                           "s_m": pl.Series([150.0, 300.0, 300.0, 400.0], dtype=pl.Float32),
                           "flags": pl.Series([0, 0, int(Flag.AT_STOP), 0], dtype=pl.UInt16)})
    out = approach(placed, links, 36.0)
    assert np.allclose(out["pos_m"].to_numpy(), [132.0, 264.0, 300.0, 82.0])
    assert np.allclose(out["s_m"].to_numpy(), [132.0, 264.0, 300.0, 382.0])
    assert out.schema == placed.schema
    assert approach(placed, links, 0).equals(placed)


def test_assign():
    links = pl.DataFrame({
        "pattern_uid": ["p", "p"], "link_idx": [0, 1], "link_key": ["A>B~x", "B>C~y"],
        "from_stop": ["A", "B"], "to_stop": ["B", "C"], "from_name": ["A", "B"], "to_name": ["B", "C"],
        "s0_m": [0.0, 90.0], "len_m": [90.0, 60.0], "geometry": [None, None]}, schema=schema.LINK)
    seg = segment(links, 30, 0, "fixed")     # link0: (0,30] (30,60] (60,90]; link1: (0,30] (30,60]
    pos = np.array([0.0, 0.1, 30.0, 30.01, 90.0, 120.0, 0.0, 30.0, 60.0, 75.0, np.nan])
    li = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert assign(pos, li, seg).tolist() == [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, -1]
    df = pl.DataFrame({"pattern_uid": ["p", "p", "q"], "link_idx": [1, 0, 0], "pos_m": [45.0, 61.0, 3.0]})
    out = assign_frame(df, seg)
    assert out["seg_idx"].to_list() == [4, 2, -1]
    assert out["seg_key"].to_list() == ["B>C~y/1@30:0f", "A>B~x/2@30:0f", None]


def test_adaptive_breaks_and_segment():
    # one link of 400 m, rate 1 per 10 m bin then 8 from 200 m to 260 m, then 1
    rng = np.random.default_rng(0)
    lam = np.ones(40)
    lam[20:26] = 8
    counts = rng.poisson(lam * 50)
    b = adaptive_breaks(counts, [0.0, 400.0])
    assert b[0] == 0 and b[-1] == 400
    w = np.diff(b)
    assert w.min() >= 20 - 1e-9 and w.max() <= 200 + 1e-9
    assert np.min(np.abs(b - 200)) <= 10 and np.min(np.abs(b - 260)) <= 10
    # stops are forced breaks
    b2 = adaptive_breaks(counts, [0.0, 155.0, 400.0])
    assert 155.0 in b2
    links = pl.DataFrame({
        "pattern_uid": ["p"], "link_idx": [0], "link_key": ["A>B~x"], "from_stop": ["A"], "to_stop": ["B"],
        "from_name": ["A"], "to_name": ["B"], "s0_m": [0.0], "len_m": [400.0], "geometry": [None]}, schema=schema.LINK)
    seg = segment(links, grid="adaptive", breaks={"p": b})
    assert seg["len_m"].sum() == pytest.approx(400) and len(seg) == len(b) - 1
    assert "@a:" in seg["seg_key"][0]
