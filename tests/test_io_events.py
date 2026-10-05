import datetime as dt
import glob
import os
from pathlib import Path

import polars as pl
import pytest

from microsegments import schema
from microsegments.io.events import read_events, read_stib_punctuality

PUNCT = Path(os.environ.get("MS_PROTOTYPE_DATA",
                            Path.home() / "Documents/Dev/CoDE/BrusselsTransitStuck/data2025")) / "stib/punctuality"


def test_stib_punctuality(tmp_path):
    t = dt.datetime(2025, 1, 2, 7, 0, 0)   # naive UTC as in the feed
    df = pl.DataFrame({
        "trip_id": ["T1"] * 3 + ["T2"], "journey_id": ["J1"] * 3 + ["J2"],
        "start_date": ["20250102"] * 4, "stop_sequence": [1, 2, 3, 1],
        "stop_id": ["5272F", "05223", "5225", "5272F"], "route_id": ["55", "55", "55", "10"],
        "arrival_time": [None, t + dt.timedelta(seconds=60), t + dt.timedelta(seconds=120), t],
        "departure_time": [t, t + dt.timedelta(seconds=80), None, t],
        "observed": [True, False, True, True],
    }).with_columns(pl.col("arrival_time", "departure_time").cast(pl.Datetime("ns")))
    df.write_parquet(tmp_path / "2025-01-01T23-00-00Z.parquet")
    ev = read_stib_punctuality(tmp_path, route="55").collect()
    schema.validate(ev, schema.STOP_EVENT)
    assert ev["stop_id"].to_list() == ["5272", "5223", "5225"]
    assert ev["trip_key"][0] == "J1:T1" and ev["service_date"][0] == dt.date(2025, 1, 2)
    assert ev["departure"][0] == t.replace(tzinfo=dt.timezone.utc)
    assert ev["observed"].to_list() == [True, False, True]
    assert read_stib_punctuality(tmp_path, route="55", only_observed=True).collect().height == 2


def test_generic_events_csv(tmp_path):
    p = tmp_path / "ev.csv"
    pl.DataFrame({"run": ["a", "a", "b"], "stop": ["S1", "S2", "S1"],
                  "arr": ["2025-03-18 01:00:00", "2025-03-18 01:05:00", "2025-03-18 08:00:00"],
                  "dep": ["2025-03-18 01:01:00", None, "2025-03-18 08:00:30"]}).write_csv(p)
    ev = read_events(p, columns={"trip_key": "run", "stop_id": "stop", "arrival": "arr", "departure": "dep"},
                     ts_unit="iso", route="R").collect()
    schema.validate(ev, schema.STOP_EVENT)
    # 01:00 local belongs to the previous service day
    assert ev["service_date"].to_list() == [dt.date(2025, 3, 17)] * 2 + [dt.date(2025, 3, 18)]
    assert ev["stop_sequence"].to_list() == [1, 2, 1]
    assert ev["arrival"][0] == dt.datetime(2025, 3, 18, 0, 0, tzinfo=dt.timezone.utc)
    assert (ev["route_id"] == "R").all() and ev["observed"].all()


@pytest.mark.skipif(not glob.glob(str(PUNCT / "*.parquet")), reason="prototype data absent")
def test_real_punctuality_day():
    f = sorted(glob.glob(str(PUNCT / "*.parquet")))[0]
    ev = read_stib_punctuality(f, route="55").collect()
    schema.validate(ev, schema.STOP_EVENT)
    assert len(ev) > 1000 and ev["stop_id"].null_count() == 0
