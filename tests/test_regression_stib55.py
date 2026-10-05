"""Regression against the prototype (time30m_2025.py) on STIB line 55, 2025-03-18/19
(tests/fixtures/stib55, golden made by tests/regression/make_golden.py).

``compat=True`` reproduces the prototype's rules. The fixture GTFS calendar does not cover March
2025: the network is built on 2025-04-15 (same pattern set) and placement falls back to it.
Residual cell mismatch (~1.5 %) is geometric: the package's link lengths differ from the
prototype's by up to 0.33 m (stop projection vs vertex snap), which moves a few rows across 30 m
bucket edges; with the prototype's link lengths every cell matches."""
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from microsegments.config import Config
from microsegments.io.events import read_stib_punctuality
from microsegments.io.stib import read_compact
from microsegments.locate import place
from microsegments.network import build_network
from microsegments.segments import assign, segment
from microsegments.tracks import passages

F = Path(__file__).parent / "fixtures" / "stib55"
G = F / "golden"
DAY = dt.date(2025, 4, 15)

pytestmark = pytest.mark.skipif(not (F / "vd55").exists(), reason="stib55 fixture missing")


@pytest.fixture(scope="module")
def run():
    obs, _ = read_compact(str(F / "vd55"), route_id="55")
    net = build_network(str(F / "gtfs"), "55", [DAY])
    cfg = Config.from_dict({"input": {"kind": "stib"},
                            "params": {"grid": "fixed", "segment_m": 30, "stop_zone": [15, 15], "layover_m": 10}})
    res = place(obs.collect(), net, cfg, compat=True)
    return net, res


@pytest.fixture(scope="module")
def cells(run):
    """(golden, ours) counted rows per (dir, link, bucket k, dow, hour)."""
    net, res = run
    fr = res.counted
    seg = segment(net, 30, grid="fixed", stop_zone=(15, 15), geometry=False)
    rows = []
    for d in (0, 1):
        uid = net.main(DAY, d)
        s = seg.filter(pl.col("pattern_uid") == uid)
        x = fr.filter(pl.col("pattern_uid") == uid)
        si = assign(x["pos_m"].to_numpy(), x["link_idx"].to_numpy(), s)
        x = x.with_columns(pl.Series("seg_idx", si)).join(s.select("seg_idx", "link_idx", "k"), on="seg_idx", suffix="_s")
        rows.append(x.select(pl.lit(d).alias("dir"), pl.col("link_idx_s").cast(pl.Int64).alias("link"),
                             pl.col("k").cast(pl.Int64), "dow", (pl.col("hour") % 24).cast(pl.Int64).alias("hour")))
    ours = pl.concat(rows).group_by("dir", "link", "k", "dow", "hour").agg(pl.len().alias("n_ours"))
    gb = pl.read_parquet(G / "buckets.parquet").with_columns((pl.col("lo") / 30).round().cast(pl.Int64).alias("k"))
    gold = (pl.read_parquet(G / "obs_dow.parquet").join(gb.select("dir", "bucket", "link", "k"), on=["dir", "bucket"])
            .select("dir", "link", "k", "dow", pl.col("hour").cast(pl.Int64), "n_obs"))
    return gold.join(ours, on=["dir", "link", "k", "dow", "hour"], how="full", coalesce=True).fill_null(0)


def test_drop_counts_equal_golden(run):
    _, res = run
    g = dict(pl.read_parquet(G / "drops.parquet").filter(pl.col("what") == "row_drop").select("reason", "n").iter_rows())
    r = res.report
    off = next(v for k, v in g.items() if k.startswith("off_pattern"))
    assert r["off_pattern"] + r.get("ambiguous", 0) + r.get("unknown_terminus", 0) == off
    for k in ("layover_first_stop", "layover_short_working_end", "layover_short_working_start", "total"):
        assert r[k] == g[k], k
    assert r["counted"] == g["kept"]
    assert res.aliases.get("9943") is not None


def test_bucket_cells(cells):
    j = cells
    assert j["n_ours"].sum() == j["n_obs"].sum()
    exact = (j["n_obs"] == j["n_ours"]).mean()
    assert exact >= 0.98      # 0.985 (geometry, see module docstring)
    lt = j.group_by("dir", "link").agg(pl.col("n_obs").sum(), pl.col("n_ours").sum())
    assert ((lt["n_ours"] / lt["n_obs"] - 1).abs() <= 0.005).all()


def test_bucket_cells_with_prototype_link_lengths(run):
    """Diagnostic: rescaling positions to the prototype's link lengths gives the golden cells."""
    net, res = run
    fr = res.counted
    fl = pl.read_parquet(G / "flen.parquet")
    gb = pl.read_parquet(G / "buckets.parquet")
    rows = []
    for d in (0, 1):
        uid = net.main(DAY, d)
        lks = (net.links_of(uid).select(pl.col("link_idx").cast(pl.Int64), "len_m")
               .with_columns(pl.lit(d).cast(pl.Int64).alias("dir"))
               .join(fl.select("dir", pl.col("link").alias("link_idx"), "glen"), on=["dir", "link_idx"]))
        x = fr.filter(pl.col("pattern_uid") == uid).with_columns(pl.col("link_idx").cast(pl.Int64)).join(lks, on="link_idx")
        p = (x["pos_m"].cast(pl.Float64) * x["glen"] / x["len_m"]).to_numpy()
        k = np.ceil(p / 30).astype(int) - 1
        nb = gb.filter(pl.col("dir") == d).group_by("link").agg(pl.len().alias("nb")).sort("link")["nb"].to_numpy()
        li = x["link_idx"].to_numpy().copy()
        k = np.minimum(k, nb[li] - 1)
        z = k < 0
        li[z] -= 1
        k[z] = nb[li[z]] - 1
        rows.append(pl.DataFrame({"dir": d, "link": li, "k": k, "dow": x["dow"].cast(pl.Int64),
                                  "hour": (x["hour"] % 24).cast(pl.Int64)}))
    ours = pl.concat(rows).group_by("dir", "link", "k", "dow", "hour").agg(pl.len().alias("n_ours"))
    gb = gb.with_columns((pl.col("lo") / 30).round().cast(pl.Int64).alias("k"))
    gold = (pl.read_parquet(G / "obs_dow.parquet").join(gb.select("dir", "bucket", "link", "k"), on=["dir", "bucket"])
            .select("dir", "link", "k", "dow", pl.col("hour").cast(pl.Int64), "n_obs"))
    j = gold.join(ours, on=["dir", "link", "k", "dow", "hour"], how="full", coalesce=True).fill_null(0)
    assert (j["n_obs"] == j["n_ours"]).mean() >= 0.99


def test_passages(run):
    net, res = run
    ev = read_stib_punctuality(str(F / "punctuality"), route="55").collect()
    P = passages(res, net, events=ev, compat=True)
    lk = pl.concat([net.links_of(net.main(DAY, d)).select(pl.lit(d).alias("dir"), pl.col("link_idx").cast(pl.Int64).alias("link"), "link_key")
                    for d in (0, 1)])
    P = (P.join(lk, on="link_key")
         .with_columns(dow=(pl.col("service_date").dt.weekday() - 1).cast(pl.Int64), hour=(pl.col("hour") % 24).cast(pl.Int64))
         .group_by("dir", "link", "dow", "hour").agg(pl.col("n_feed").sum(), pl.col("n_events").sum(), pl.col("n").sum()))
    gp = pl.read_parquet(G / "passages.parquet").filter(pl.col("dow").is_in([1, 2]))
    pj = gp.join(P, on=["dir", "link", "dow", "hour"], how="left").fill_null(0)
    assert (pj["n_feed"] == pj["n_feed_right"]).mean() >= 0.99
    assert (pj["n_punct"] == pj["n_events"]).mean() >= 0.99
    tot = pj.group_by("dir", "link").agg([pl.col(c).sum() for c in ("n_feed", "n_feed_right", "n_punct", "n_events", "n_max", "n")])
    for ours, gold in (("n_feed_right", "n_feed"), ("n_events", "n_punct"), ("n", "n_max")):
        assert ((tot[ours] / tot[gold] - 1).abs() <= 0.03).all(), (ours, tot)
