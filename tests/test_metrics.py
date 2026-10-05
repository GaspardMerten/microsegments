import numpy as np
import polars as pl
import pytest

from microsegments import aggregate, metrics
from microsegments.config import Params, Quality, Select
from microsegments.schema import RESULT, Flag, validate
from t4_fixtures import Hot, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(n_days=30, seed=11, hots=[Hot(2, 200, 260, 16, tuple(range(7, 10)), "congestion")])


def _run(sim, L=30, **kw):
    segs = sim.segment_fn(L)
    cube = aggregate.count(sim.placed, segs)
    return metrics.analyse(cube, kw.pop("coverage", sim.coverage), sim.passages, segs, sim.pattern_days, **kw)


def test_result_schema_and_bands(sim):
    an = _run(sim)
    validate(an.result, RESULT)
    hrs = set(an.result["hour"].unique().to_list())
    assert {-1, -2, -3, -4} <= hrs and set(range(5, 24)) <= hrs
    r = an.result.filter(pl.col("hour") == -3)
    # ratio of sums
    np.testing.assert_allclose(r["obs_per_h"], r["obs"].cast(float) / r["covered_h"], rtol=1e-4)
    np.testing.assert_allclose(r["excess_per_passage"], r["obs_per_passage"] - r["ref_obs_per_passage"], atol=1e-5)
    np.testing.assert_allclose(r["log2_ratio"], np.log2((r["obs_per_passage"] + .5) / (r["ref_obs_per_passage"] + .5)), atol=1e-5)
    np.testing.assert_allclose(r["excess_obs_per_h"],
                               (r["obs"].cast(float) - r["ref_obs_per_passage"] * r["passages"]) / r["covered_h"], rtol=1e-3, atol=1e-4)
    # band day = sum of hours 6..20
    h = an.result.filter(pl.col("hour").is_between(6, 20)).group_by("seg_key").agg(pl.col("obs").sum())
    j = r.join(h, on="seg_key")
    assert (j["obs"] == j["obs_right"]).all()
    # the injected peak queue shows in the am band only
    top = an.result.filter(pl.col("hour") == -1).sort("excess_per_passage", descending=True)["x0_m"][0]
    x0 = sim.link_x0[2]
    assert x0 + 170 <= top <= x0 + 270


def _degrade(sim, share, c, seed=0):
    """Random day-hours get coverage c and keep a share c of their rows (missing polls)."""
    rng = np.random.default_rng(seed)
    out = [(d, h) for d in sim.days for h in sim.hours if rng.random() < share]
    key = pl.DataFrame(out, schema={"service_date": pl.Date, "hour": pl.Int8}, orient="row").with_columns(pl.lit(True).alias("_o"))
    drop = sim.placed.join(key, on=["service_date", "hour"], how="left")
    keep = drop.filter(pl.col("_o").is_null() | (pl.Series(rng.random(drop.height)) < c)).drop("_o")
    cov = sim.coverage.join(key, on=["service_date", "hour"], how="left").with_columns(
        pl.when(pl.col("_o")).then(c).otherwise(pl.col("coverage")).cast(pl.Float32).alias("coverage")).drop("_o")
    return keep, cov


def _line_rate(an, h):
    r = an.result.filter(pl.col("hour") == h)
    return r["obs"].sum() / r["covered_h"][0], r["covered_h"][0]


@pytest.mark.parametrize("c,hour", [(0.85, 8), (0.85, -3), (0.5, -3)])
def test_exposure_invariance(sim, c, hour):
    """Partial polling (kept, exposure-normalised) or hours removed through low coverage (c=0.5)
    change the line's obs_per_h by < 2 %."""
    keep, cov = _degrade(sim, 0.15, c)
    segs = sim.segment_fn(30)
    a = metrics.analyse(aggregate.count(sim.placed, segs), sim.coverage, sim.passages, segs, sim.pattern_days)
    b = metrics.analyse(aggregate.count(keep, segs), cov, sim.passages, segs, sim.pattern_days)
    ra, ca = _line_rate(a, hour)
    rb, cb = _line_rate(b, hour)
    assert abs(rb / ra - 1) < 0.02
    assert cb < ca


def test_day_exclusion(sim):
    d_low, d_frz = sim.days[3], sim.days[4]
    cov = sim.coverage.with_columns(
        pl.when((pl.col("service_date") == d_low) & pl.col("hour").is_between(6, 13)).then(0.3)
        .otherwise(pl.col("coverage")).cast(pl.Float32).alias("coverage"),
        pl.when((pl.col("service_date") == d_frz) & pl.col("hour").is_between(9, 20)).then(Flag.FROZEN)
        .otherwise(pl.col("flags")).cast(pl.UInt16).alias("flags"))
    an = _run(sim, coverage=cov, select=Select(exclude_dates=[str(sim.days[5])]))
    ex = dict(an.excluded_days.iter_rows())
    assert ex == {d_low: "low_coverage", d_frz: "frozen", sim.days[5]: "excluded"}
    assert len(an.days) == len(sim.days) - 3
    assert an.result["n_days"].max() == len(sim.days) - 3
    # a few bad hours only: day kept, those hours dropped
    cov2 = sim.coverage.with_columns(pl.when((pl.col("service_date") == d_low) & pl.col("hour").is_between(6, 8))
                                     .then(0.3).otherwise(pl.col("coverage")).cast(pl.Float32).alias("coverage"))
    an2 = _run(sim, coverage=cov2)
    assert an2.excluded_days.height == 0
    dh = an2.day_hours.filter((pl.col("service_date") == d_low) & pl.col("hour").is_between(6, 8))
    assert not dh["included"].any()


def test_per_version_aggregation():
    full = simulate(n_days=30, seed=21)
    cut = simulate(n_days=30, seed=21, cut_days=range(20, 30))
    a, b = _run(full), _run(cut)
    assert b.versions.height == 2
    v = dict(zip(b.versions["pattern_uid"], b.versions["n_days"]))
    assert v == {"P0": 20, "P1": 10}
    last = cut.link_keys[-1]
    rb = b.result.filter((pl.col("pattern_uid") == "P0") & (pl.col("hour") == -3))
    seg_last = rb.filter(pl.col("link_key") == last)
    seg_trunk = rb.filter(pl.col("link_key") != last)
    assert (seg_last["n_days"] == 20).all() and (seg_last["flags"] & Flag.PARTIAL_VERSIONS > 0).all()
    assert (seg_trunk["n_days"] == 30).all() and (seg_trunk["flags"] & Flag.PARTIAL_VERSIONS == 0).all()
    # cut segments are aggregated over their valid days only: rates not diluted by the cut days
    ra = a.result.filter((pl.col("hour") == -3) & (pl.col("link_key") == last))
    ratio = seg_last["obs_per_h"].sum() / ra["obs_per_h"].sum()
    assert 0.9 < ratio < 1.1
    np.testing.assert_allclose(seg_last["covered_h"], ra["covered_h"] * 20 / 30, rtol=1e-4)
    # P1 rows exist and share the trunk seg_keys values
    p1 = b.result.filter((pl.col("pattern_uid") == "P1") & (pl.col("hour") == -3))
    assert p1.height == seg_trunk.height
    assert p1.join(seg_trunk, on="seg_key").select((pl.col("obs") == pl.col("obs_right")).all()).item()
    # FEW_DAYS
    c = _run(cut, quality=Quality(min_days=25))
    r = c.result.filter((pl.col("hour") == -3) & (pl.col("link_key") == last))
    assert (r["flags"] & Flag.FEW_DAYS > 0).all()


def test_contract_wk(sim):
    an = _run(sim)
    wk = an.to_contract_wk(0)
    nseg = an.result.filter((pl.col("pattern_uid") == "P0") & (pl.col("hour") == 8)).height
    assert len(wk["obs"]) == 7 and len(wk["obs"][0]) == len(an.hours) and len(wk["obs"][0][0]) == nseg
    assert len(wk["cov"][0][0]) == len(sim.link_len) and len(wk["days"][0]) == len(sim.link_len)
    hi = an.hours.index(8)
    tot = np.sum([wk["obs"][w][hi] for w in range(7)], axis=0)
    r = an.result.filter((pl.col("pattern_uid") == "P0") & (pl.col("hour") == 8)).sort("seg_idx")
    assert (tot == r["obs"].to_numpy()).all()
    assert sum(wk["days"][w][0] for w in range(7)) == len(sim.days)
    cov = sum(wk["cov"][w][hi][0] for w in range(7))
    assert abs(cov - r["covered_h"][0]) < 1e-3


def test_reference_and_stratify(sim):
    a = _run(sim)
    f = _run(sim, params=Params(reference="freeflow"))
    s = metrics.analyse(aggregate.count(sim.placed, sim.segment_fn(30)), sim.coverage, sim.passages, sim.segment_fn(30),
                        sim.pattern_days, stratify=True)
    ra, rf, rs = (x.result.filter(pl.col("hour") == -3).sort("seg_idx") for x in (a, f, s))
    assert rf["ref_obs_per_passage"].is_not_null().all()
    # weekdays are balanced in the fixture: stratified close to pooled
    np.testing.assert_allclose(rs["obs_per_h"], ra["obs_per_h"], rtol=0.05)
    np.testing.assert_allclose(rf["ref_obs_per_passage"], ra["ref_obs_per_passage"], rtol=0.35, atol=0.05)


def test_min_passages_nulls(sim):
    an = _run(sim, quality=Quality(min_passages_per_day=7))
    r = an.result.filter(pl.col("hour") == 5)          # 4 vehicles / h at 5 h
    assert r["obs_per_passage"].is_null().all()
    assert an.result.filter(pl.col("hour") == 8)["obs_per_passage"].is_not_null().all()
