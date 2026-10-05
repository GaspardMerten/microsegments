import polars as pl

import microsegments as ms
from microsegments import schema


def test_config_roundtrip(tmp_path):
    p = tmp_path / "ms.toml"
    p.write_text('[input]\npaths=["a.csv"]\nkind="latlon"\n[input.columns]\nts="time"\n'
                 '[params]\nsegment_m=20\nstop_zone=[15,15]\n[select]\nroute="55"\n')
    cfg = ms.Config.from_toml(p)
    assert cfg.params.segment_m == 20 and cfg.params.stop_zone == (15, 15)
    assert cfg.input.columns == {"ts": "time"} and cfg.input.paths[0].endswith("a.csv")


def test_conform_adds_missing():
    df = pl.DataFrame({"ts": [0], "route_id": ["1"], "lat": [50.0], "lon": [4.0], "x": [1]})
    df = df.with_columns(pl.from_epoch("ts", "s").dt.replace_time_zone("UTC"))
    out = schema.conform(df, schema.OBSERVATION, schema.OBSERVATION_REQUIRED)
    assert out["stop_id"].dtype == pl.Utf8 and "x" in out.columns
    schema.validate(out.select(list(schema.OBSERVATION)), schema.OBSERVATION)


def test_flags():
    assert ms.Flag.names(ms.Flag.LOW_COVERAGE | ms.Flag.FEW_DAYS) == ["low_coverage", "few_days"]
