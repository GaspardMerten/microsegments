import datetime as dt

import polars as pl
import pytest

from microsegments import Config, schema
from microsegments.io import read_observations
from microsegments.io.ids import normalise_expr, normalise_py
from microsegments.simulate import simulate

TZ = "Europe/Brussels"


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    return simulate(seed=2, days=2, mode="gtfsrt", out_dir=tmp_path_factory.mktemp("g"), outages=[], frozen=[])


def _cfg(paths, **inp):
    select = inp.pop("select", {})
    return Config.from_dict({"input": {"paths": [str(p) for p in paths], **inp}, "select": select})


def _check(got: pl.DataFrame, ref: pl.DataFrame):
    schema.validate(got.select(list(schema.OBSERVATION)), schema.OBSERVATION)
    got, ref = got.sort("ts", "vehicle_id"), ref.sort("ts", "vehicle_id")
    assert len(got) == len(ref)
    assert (got["ts"] == ref["ts"]).all()
    assert (got["lat"] - ref["lat"]).abs().max() < 1e-9
    assert (got["vehicle_id"] == ref["vehicle_id"]).all()


def test_csv_latlon_iso_local(sim, tmp_path):
    ref = sim.observations
    src = ref.select(
        pl.col("ts").dt.convert_time_zone(TZ).dt.strftime("%Y-%m-%d %H:%M:%S").alias("time"),
        pl.col("lat").alias("latitude"), pl.col("lon").alias("longitude"),
        pl.col("vehicle_id").alias("veh"), pl.col("route_id").alias("line"),
        pl.col("trip_id"), pl.col("direction_id"))
    p = tmp_path / "pos.csv"
    src.write_csv(p)
    cfg = _cfg([p], ts_unit="iso", columns={"ts": "time", "lat": "latitude", "lon": "longitude",
                                            "vehicle_id": "veh", "route_id": "line"})
    _check(read_observations(cfg).collect(), ref)


def test_csv_iso_with_offset_and_autodetect(sim, tmp_path):
    ref = sim.observations
    src = ref.select(
        pl.col("ts").dt.strftime("%Y-%m-%dT%H:%M:%SZ").alias("vehicle.timestamp"),
        pl.col("lat").alias("vehicle.position.latitude"), pl.col("lon").alias("vehicle.position.longitude"),
        pl.col("vehicle_id").alias("vehicle.vehicle.id"), pl.col("route_id").alias("vehicle.trip.route_id"),
        pl.col("trip_id").alias("vehicle.trip.trip_id"), pl.col("bearing").alias("vehicle.position.bearing"))
    p = tmp_path / "flat.csv"
    src.write_csv(p)
    got = read_observations(_cfg([p], ts_unit="iso")).collect()   # kind latlon, no mapping: detected
    _check(got, ref)
    assert (got.sort("ts", "vehicle_id")["trip_id"] == ref.sort("ts", "vehicle_id")["trip_id"]).all()


def test_parquet_epoch_ms_route_filter_and_glob(sim, tmp_path):
    ref = sim.observations
    src = ref.select(pl.col("ts").dt.epoch("ms").alias("t_ms"), "lat", "lon", "vehicle_id", "route_id")
    other = src.with_columns(pl.lit("OTHER").alias("route_id"))
    src.write_parquet(tmp_path / "a.parquet")
    other.write_parquet(tmp_path / "b.parquet")
    cfg = _cfg([tmp_path / "*.parquet"], ts_unit="ms", columns={"ts": "t_ms"}, select={"route": "S1"})
    _check(read_observations(cfg).collect(), ref)


def test_route_filled_and_date_filter(sim, tmp_path):
    ref = sim.observations
    p = tmp_path / "x.parquet"
    ref.select(pl.col("ts").dt.epoch("s").alias("ts"), "lat", "lon", "vehicle_id").write_parquet(p)
    d0 = sim.dates[0].isoformat()
    cfg = _cfg([p], select={"route": "S1", "dates": f"{d0}..{d0}"})
    got = read_observations(cfg).collect()
    assert set(got["route_id"]) == {"S1"}
    from microsegments.io import add_local_time
    exp = add_local_time(ref, TZ).filter(pl.col("service_date") == sim.dates[0])
    assert len(got) == len(exp) > 0
    with pytest.raises(ValueError):
        read_observations(_cfg([p])).collect()   # no route column, no select.route


def test_id_normalisers():
    s = pl.DataFrame({"x": ["0626", "5272F", "0", "T92", None]})
    assert s.select(normalise_expr(pl.col("x"), "leading_digits"))["x"].to_list() == ["626", "5272", "0", None, None]
    assert s.select(normalise_expr(pl.col("x"), "none"))["x"].to_list() == ["0626", "5272F", "0", "T92", None]
    assert s.select(normalise_expr(pl.col("x"), r"(\d+)F?$"))["x"].to_list() == ["0626", "5272", "0", "92", None]
    assert normalise_py("0626") == "626" and normalise_py("T92") is None


def test_linear_csv_columns(tmp_path):
    p = tmp_path / "lin.csv"
    pl.DataFrame({"t": [1739763459, 1739763479], "ln": ["55", "55"], "term": ["9943", "0529"],
                  "pt": ["05766", "5298F"], "m": [95, 0]}).write_csv(p)
    cfg = _cfg([p], kind="linear", columns={"ts": "t", "route_id": "ln", "terminus_id": "term",
                                            "stop_id": "pt", "dist_m": "m"})
    got = read_observations(cfg).collect()
    assert got["stop_id"].to_list() == ["5766", "5298"] and got["terminus_id"].to_list() == ["9943", "529"]
    assert got["ts"][0] == dt.datetime(2025, 2, 17, 3, 37, 39, tzinfo=dt.timezone.utc)
    assert got["dist_m"].dtype == pl.Float32
