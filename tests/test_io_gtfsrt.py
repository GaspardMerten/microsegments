import polars as pl
import pytest

from microsegments import schema
from microsegments.io import gtfsrt
from microsegments.simulate import simulate


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    return simulate(seed=5, days=1, mode="gtfsrt", out_dir=tmp_path_factory.mktemp("g"), outages=[], frozen=[])


def test_detect_columns():
    names = ["header.timestamp", "entity.id", "vehicle.timestamp", "vehicle.position.latitude",
             "vehicle.position.longitude", "vehicle.position.bearing", "vehicle.trip.trip_id",
             "vehicle.trip.route_id", "vehicle.trip.direction_id", "vehicle.vehicle.id", "vehicle.vehicle.label"]
    m = gtfsrt.detect_columns(names)
    assert m == {"ts": "vehicle.timestamp", "lat": "vehicle.position.latitude",
                 "lon": "vehicle.position.longitude", "bearing": "vehicle.position.bearing",
                 "trip_id": "vehicle.trip.trip_id", "route_id": "vehicle.trip.route_id",
                 "direction_id": "vehicle.trip.direction_id", "vehicle_id": "vehicle.vehicle.id"}
    m = gtfsrt.detect_columns(["timestamp", "latitude", "longitude", "tripId", "routeId", "vehicleId", "bearing"])
    assert m["trip_id"] == "tripId" and m["vehicle_id"] == "vehicleId" and m["route_id"] == "routeId"
    m = gtfsrt.detect_columns(["lat", "lon", "ts", "vehicle_id", "direction_id"])
    assert set(m) == {"lat", "lon", "ts", "vehicle_id", "direction_id"}


def test_read_flat_epoch_guess(sim, tmp_path):
    ref = sim.observations
    src = ref.select(pl.col("ts").dt.epoch("ms").alias("timestamp"), pl.col("lat").alias("latitude"),
                     pl.col("lon").alias("longitude"), "vehicle_id", "trip_id", "route_id", "direction_id")
    src.write_parquet(tmp_path / "p.parquet")
    got = gtfsrt.read_flat(tmp_path / "p.parquet").collect()
    schema.validate(got.select(list(schema.OBSERVATION)), schema.OBSERVATION)
    assert (got["ts"] == ref["ts"]).all() and (got["direction_id"] == ref["direction_id"]).all()


def test_pb_roundtrip(sim, tmp_path):
    pytest.importorskip("google.transit.gtfs_realtime_pb2")
    ref = sim.observations.head(200)
    chunks = ref.with_columns((pl.col("ts").dt.epoch("s") // 60).alias("m")).partition_by("m", include_key=False)
    for i, ch in enumerate(chunks):
        (tmp_path / f"f{i:03d}.pb").write_bytes(gtfsrt.build_feed(ch))
    (tmp_path / "dup.pb").write_bytes(gtfsrt.build_feed(chunks[0]))   # repeated poll
    got = gtfsrt.read_pb(tmp_path / "*.pb", route="S1")
    assert len(got) == len(ref)
    got, ref = got.sort("ts", "vehicle_id"), ref.sort("ts", "vehicle_id")
    assert (got["vehicle_id"] == ref["vehicle_id"]).all() and (got["trip_id"] == ref["trip_id"]).all()
    assert (got["lat"] - ref["lat"]).abs().max() < 1e-5   # float32 in protobuf
