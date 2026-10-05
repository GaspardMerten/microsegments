import datetime as dt
import gzip
import json
import os
from pathlib import Path

import polars as pl
import pytest

from microsegments import Config, schema
from microsegments.io import coverage, read_observations, read_snapshots
from microsegments.io.stib import (parse_snapshots, read_compact, read_json_snapshots,
                                   observations_to_compact, ts_from_filename)
from microsegments.simulate import simulate

VD55 = Path(os.environ.get("MS_PROTOTYPE_DATA",
                           Path.home() / "Documents/Dev/CoDE/BrusselsTransitStuck/data2025")) / "stib/vd55"


def _snap(rows):
    return [{"lineId": l, "directionId": d, "pointId": p, "distanceFromPoint": m} for l, d, p, m in rows]


def _write_snapshots(tmp_path):
    t0 = dt.datetime(2025, 3, 3, 7, 0, 0, tzinfo=dt.timezone.utc)
    snaps = [
        _snap([("55", "9943", "05766", 95), ("55", "0529", "5298", 0), ("10", "5800", "5766", 150)]),
        _snap([("55", "9943", "05766", 120), ("10", "5800", "5766", 160)]),
        _snap([("55", "9943", "05766", 120), ("10", "5800", "5766", 160)]),   # frozen copy
        {"data": _snap([("55", "9943", "05767", 3)])},
    ]
    paths = []
    for i, s in enumerate(snaps):
        name = (t0 + dt.timedelta(seconds=20 * i)).strftime("%Y-%m-%dT%H-%M-%SZ") + ".json"
        p = tmp_path / name
        if i == 3:
            p = tmp_path / (name + ".gz")
            p.write_bytes(gzip.compress(json.dumps(s).encode()))
        else:
            p.write_text(json.dumps(s))
        paths.append(p)
    return t0, paths


def test_json_snapshots(tmp_path):
    t0, paths = _write_snapshots(tmp_path)
    assert ts_from_filename(paths[0]) == t0.timestamp()
    obs, sn = read_json_snapshots(tmp_path / "*.json*", route="55")
    schema.validate(obs.select(list(schema.OBSERVATION)), schema.OBSERVATION)
    schema.validate(sn, schema.SNAPSHOT)
    assert sn["frozen"].to_list() == [False, False, True, False]
    assert sn["n_rows"].to_list() == [3, 2, 2, 1]
    assert len(obs) == 2 + 1 + 1    # frozen snapshot adds no row
    assert obs["stop_id"].to_list() == ["5766", "5298", "5766", "5767"]
    assert obs["terminus_id"].to_list() == ["9943", "529", "9943", "9943"]
    assert obs["dist_m"].to_list() == [95, 0, 120, 3]
    assert obs["ts"][2] == t0 + dt.timedelta(seconds=20)
    assert obs["vehicle_id"].is_null().all() and (obs["route_id"] == "55").all()
    # same via the config entry point
    got = read_observations(Config.from_dict({"input": {"paths": [str(tmp_path / "*")], "kind": "stib"},
                                              "select": {"route": "55"}})).collect()
    assert got.equals(obs)
    # coverage: frozen poll not counted
    c = coverage(sn, gap_cap_s=40)
    assert c.filter(pl.col("hour") == 8)["n_snapshots"][0] == 3


def test_parse_snapshots_all_lines():
    obs, sn = parse_snapshots([(1_700_000_000, _snap([("1", "2", "3", 4), ("5", "6", "7", 8)]))], route=None,
                              id_normaliser="none")
    assert obs["route_id"].to_list() == ["1", "5"] and len(sn) == 1


def test_compact_roundtrip_from_simulator(tmp_path):
    r = simulate(seed=4, days=2, mode="stib", out_dir=tmp_path / "gtfs")
    r.write_stib(tmp_path / "vd")
    obs, snaps = read_compact(tmp_path / "vd")
    obs = obs.collect().sort("ts", "stop_id", "dist_m")
    ref = r.observations.select(list(schema.OBSERVATION)).sort("ts", "stop_id", "dist_m")
    assert obs.select(list(schema.OBSERVATION)).equals(ref)
    snaps = snaps.collect()
    schema.validate(snaps, schema.SNAPSHOT)
    assert snaps["frozen"].sum() == r.snapshots["frozen"].sum()
    assert len(snaps) == len(r.snapshots)
    # config path: kind stib, snaps found next to the rows
    cfg = Config.from_dict({"input": {"paths": [str(tmp_path / "vd")], "kind": "stib"}, "select": {"route": "S1"}})
    assert read_observations(cfg).collect().height == len(ref)
    assert read_snapshots(cfg).collect().height == len(snaps)
    # compact writer
    back = observations_to_compact(r.observations)
    assert back.columns == ["ts", "line", "dir", "point", "dist"]


def test_prototype_layout_needs_route(tmp_path):
    pl.DataFrame({"ts": [1739763459, 1739763479], "dir": ["529", "529"], "point": ["5298", "5298"],
                  "dist": [0, 81]}).write_parquet(tmp_path / "20250217.parquet")
    pl.DataFrame({"ts": [1739763459, 1739763479, 1739763499], "same": [False, False, True]}).write_parquet(
        tmp_path / "20250217.snaps.parquet")
    with pytest.raises(ValueError):
        read_compact(tmp_path)[0].collect()
    obs, sn = read_compact(tmp_path, route_id="55")
    assert obs.collect()["route_id"].to_list() == ["55", "55"]
    assert sn.collect()["frozen"].to_list() == [False, False, True]


@pytest.mark.skipif(not (VD55 / "20250217.parquet").exists(), reason="prototype data absent")
def test_real_vd55_day():
    obs, sn = read_compact([VD55 / "20250217.parquet", VD55 / "20250217.snaps.parquet"], route_id="55")
    obs = obs.collect()
    sn = sn.collect()
    assert len(obs) > 10_000 and obs["stop_id"].null_count() == 0
    assert obs["ts"].min() >= sn["ts"].min()
    c = coverage(sn)
    assert c.filter(pl.col("hour").is_between(6, 21))["coverage"].mean() > 0.9
