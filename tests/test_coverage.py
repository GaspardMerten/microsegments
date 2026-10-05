import datetime as dt

import numpy as np
import polars as pl

from microsegments.io.coverage import coverage, snapshots_from_observations
from microsegments.io.timeutil import add_local_time, local_epoch
from microsegments.schema import COVERAGE, Flag, validate

TZ = "Europe/Brussels"
DAY = dt.date(2025, 3, 4)


def _snaps(t, frozen=None):
    t = np.asarray(t, dtype=np.float64)
    return pl.DataFrame({
        "ts": (t * 1e6).astype(np.int64),
        "frozen": np.zeros(t.size, bool) if frozen is None else np.asarray(frozen, bool),
    }).with_columns(pl.col("ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"))


def _day_polls(step=20.0):
    a = local_epoch(DAY, 4 * 3600, TZ)
    return a + np.arange(0, 86400, step)


def _brute(t, frozen, cap, tick, tz=TZ):
    """Independent per-second reference: covered seconds per local hour (integer times only)."""
    t = np.asarray(t, dtype=np.int64)
    live = t[~frozen]
    gaps = np.diff(live, append=live[-1] + tick)
    a0 = int(local_epoch(DAY, 4 * 3600, tz))
    cov = np.zeros(86400 + 7200, bool)
    for ti, g in zip(live, np.minimum(gaps, cap).astype(int)):
        cov[ti - a0: ti - a0 + g] = True
    return {4 + h: int(cov[h * 3600:(h + 1) * 3600].sum()) for h in range(24)}


def test_full_day_is_fully_covered():
    c = coverage(_snaps(_day_polls()), TZ)
    validate(c, COVERAGE)
    assert len(c) == 24 and c["hour"].to_list() == list(range(4, 28))
    assert (c["coverage"] == 1.0).all() and (c["flags"] == 0).all()
    assert c["n_snapshots"].to_list() == [180] * 24


def test_outage_exact():
    t = _day_polls()
    o1, o2 = local_epoch(DAY, 8 * 3600 + 600, TZ), local_epoch(DAY, 8 * 3600 + 1800, TZ)
    t = t[(t < o1) | (t >= o2)]
    c = coverage(_snaps(t), TZ, gap_cap_s=40)
    row = c.filter(pl.col("hour") == 8).row(0, named=True)
    # last poll 08:09:40 covers 40 s, next poll 08:30:00: 20 min - 20 s lost
    assert abs(row["covered_s"] - (3600 - 1180)) <= 1
    assert row["flags"] & Flag.LOW_COVERAGE
    assert c.filter(pl.col("hour") != 8)["coverage"].min() == 1.0


def test_random_outages_match_brute_force():
    rng = np.random.default_rng(3)
    t = _day_polls() + rng.integers(-2, 3, 4320)
    t = np.unique(t)
    keep = rng.random(t.size) > 0.05
    for s in rng.uniform(0, 86000, 6):
        a = t[0] + s
        keep &= ~((t >= a) & (t < a + rng.uniform(60, 3000)))
    t = t[keep]
    frozen = np.zeros(t.size, bool)
    ref = _brute(t, frozen, 40, 20)
    c = coverage(_snaps(t), TZ, gap_cap_s=40, tick_s=20)
    got = dict(zip(c["hour"].to_list(), c["covered_s"].to_list()))
    for h in range(4, 28):
        assert abs(got[h] - ref[h]) <= 1, (h, got[h], ref[h])


def test_interval_split_across_hours_and_infinite_cap():
    a = local_epoch(DAY, 8 * 3600 + 3590, TZ)   # 08:59:50
    c = coverage(_snaps([a, a + 30]), TZ, gap_cap_s=40, tick_s=20)
    got = dict(zip(c["hour"].to_list(), c["covered_s"].to_list()))
    assert got[8] == 10 and got[9] == 20 + 20
    # infinite cap: a 3 h gap is fully credited, hour by hour
    c = coverage(_snaps([a, a + 3 * 3600]), TZ, gap_cap_s=float("inf"), tick_s=20)
    got = dict(zip(c["hour"].to_list(), c["covered_s"].to_list()))
    # second poll at 11:59:50 covers its tick (20 s) across 12:00
    assert got[8] == 10 and got[9] == got[10] == got[11] == 3600 and got[12] == 10


def test_frozen_not_counted():
    t = _day_polls()
    f1, f2 = local_epoch(DAY, 10 * 3600, TZ), local_epoch(DAY, 10 * 3600 + 600, TZ)
    frozen = (t >= f1) & (t < f2)
    c = coverage(_snaps(t, frozen), TZ, gap_cap_s=40, max_frozen_s=300)
    row = c.filter(pl.col("hour") == 10).row(0, named=True)
    assert row["n_snapshots"] == 180 - 30
    assert abs(row["frozen_s"] - 600) <= 1
    # last live poll before (09:59:40) covers 40 s into 10:00:20; 10:10 poll resumes
    assert abs(row["covered_s"] - (3600 - 580)) <= 1
    assert row["flags"] & Flag.FROZEN and not row["flags"] & Flag.LOW_COVERAGE
    # same result as if the frozen polls were absent, for covered_s
    c2 = coverage(_snaps(t[~frozen]), TZ, gap_cap_s=40)
    assert c2.filter(pl.col("hour") == 10)["covered_s"][0] == row["covered_s"]


def test_scattered_repeats_do_not_flag_frozen():
    """One poll in four repeating the previous one (a feed refreshed more slowly than polled) is not a
    stall: ~900 s of frozen snapshots in the hour, no run longer than 20 s."""
    t = _day_polls()
    frozen = (np.arange(t.size) % 4) == 3
    c = coverage(_snaps(t, frozen), TZ, gap_cap_s=40, max_frozen_s=300)
    row = c.filter(pl.col("hour") == 10).row(0, named=True)
    assert row["frozen_s"] > 300
    assert not row["flags"] & Flag.FROZEN and not row["flags"] & Flag.LOW_COVERAGE


def test_stall_across_hours_flags_both():
    t = _day_polls()
    f1, f2 = local_epoch(DAY, 10 * 3600 + 3400, TZ), local_epoch(DAY, 11 * 3600 + 200, TZ)   # 400 s stall
    c = coverage(_snaps(t, (t >= f1) & (t < f2)), TZ, gap_cap_s=40, max_frozen_s=300)
    fl = dict(zip(c["hour"].to_list(), c["flags"].to_list()))
    assert fl[10] & Flag.FROZEN and fl[11] & Flag.FROZEN and not fl[12] & Flag.FROZEN
    c = coverage(_snaps(t, (t >= f1) & (t < f1 + 200)), TZ, gap_cap_s=40, max_frozen_s=300)   # 200 s: fine
    assert not any(f & Flag.FROZEN for f in c["flags"].to_list())


def test_silent_hours_are_zero_rows():
    t = _day_polls()
    t = t[(t < local_epoch(DAY, 12 * 3600, TZ)) | (t >= local_epoch(DAY, 14 * 3600, TZ))]
    c = coverage(_snaps(t), TZ)
    assert c.filter(pl.col("hour") == 13)["coverage"][0] == 0
    assert c.filter(pl.col("hour") == 13)["n_snapshots"][0] == 0


def test_service_day_and_hour_boundaries():
    ts = [dt.datetime(2025, 3, 18, 0, 30, tzinfo=dt.timezone.utc),   # 01:30 local -> 17th, hour 25
          dt.datetime(2025, 3, 18, 3, 0, tzinfo=dt.timezone.utc),    # 04:00 local -> 18th, hour 4
          dt.datetime(2025, 3, 18, 2, 59, tzinfo=dt.timezone.utc),   # 03:59 local -> 17th, hour 27
          dt.datetime(2025, 3, 30, 1, 30, tzinfo=dt.timezone.utc),   # DST night: 03:30 CEST -> 29th, hour 27
          dt.datetime(2025, 3, 30, 2, 30, tzinfo=dt.timezone.utc)]   # 04:30 CEST -> 30th (Sunday), hour 4
    df = add_local_time(pl.DataFrame({"ts": ts}).with_columns(pl.col("ts").dt.cast_time_unit("us")), TZ, "04:00")
    assert df["service_date"].to_list() == [dt.date(2025, 3, 17), dt.date(2025, 3, 18), dt.date(2025, 3, 17),
                                            dt.date(2025, 3, 29), dt.date(2025, 3, 30)]
    assert df["hour"].to_list() == [25, 4, 27, 27, 4]
    assert df["dow"].to_list() == [0, 1, 0, 5, 6]
    assert df["hour"].dtype == pl.Int8 and df["dow"].dtype == pl.Int8


def test_snapshots_from_observations():
    obs = _snaps([5, 5, 25, 45]).drop("frozen").with_columns(pl.lit("1").alias("route_id"))
    s = snapshots_from_observations(obs)
    assert s["n_rows"].to_list() == [2, 1, 1] and not s["frozen"].any()
