"""Golden reference data from the tram 55 prototype (BrusselsTransitStuck/time30m_2025.py).

The `microsegments` package must reproduce, on the same input, what the prototype counts. This
script imports the prototype's own functions (geometry, load_feed, place, track, passages), runs
them exactly as `time30m_2025.main()` does, and dumps the intermediate tables the package will be
regression-tested against. It never writes into the prototype directory (its out/time30m_2025.json
is only read, for the self-check).

Run it with the prototype's environment (plain python3 with pandas/numpy/pyarrow), not the
package's:

    cd ~/Documents/Dev/CoDE/BrusselsTransitStuck
    PYTHONPATH=. python3 ~/Documents/Dev/CoDE/microsegments/tests/regression/make_golden.py
        # full period (all weekdays with both vd55 and punctuality, ~3 min)
        # -> microsegments/tests/regression/golden/, checked against out/time30m_2025.json

    PYTHONPATH=. python3 .../make_golden.py --days 20250318,20250319 \
        --write-fixture .../microsegments/tests/fixtures/stib55
        # cut the committed fixture (vd55 days, route-55 GTFS subset, route-55 punctuality,
        # validation reference) from the prototype data, then

    PYTHONPATH=. python3 .../make_golden.py --days 20250318,20250319 \
        --data .../microsegments/tests/fixtures/stib55 --out .../tests/fixtures/stib55/golden
        # golden of the 2 fixture days, computed from the fixture itself

Options: --proto DIR (prototype dir, default ~/Documents/Dev/CoDE/BrusselsTransitStuck),
--data DIR (read a fixture instead of <proto>/data2025), --days D1,D2 (subset of days),
--out DIR, --write-fixture DIR.

Outputs (parquet, plus summary.json):
  buckets.parquet      dir, dir_idx, bucket, link, a, b, lo, hi, x0, len: the 30 m buckets of each
                       direction (link-anchored from stop A, last bucket keeps its true length, a
                       sliver < 1 m merges into the previous one).
  obs_dow.parquet      dir, bucket, dow, hour -> n_obs (counted placed rows = vehicle observations,
                       the package's indicator), n_obs_run (outside +-15 m of stops), ts / ts_run
                       (the prototype's tram-seconds, credit capped at 40 s). Totals over the days
                       of that weekday. All local hours 0-23 are kept (the prototype shows 5-23).
  obs_day.parquet      same per service day (non-zero cells only).
  passages.parquet     dir, link, a, b, dow, hour -> n_feed (tracker, median over counting lines,
                       may be x.5), n_punct (stib/punctuality), n_max = max of the two (the
                       prototype's passage count). Totals over the days of that weekday.
  flen.parquet         dir, link, a, b, glen (GTFS metres), flen (p99.5 of feed metres after A).
  validation.parquet   per link, band 'jour' (6-20 h): buckets_s (sum of bucket s/tram) vs ]A-B]
                       of speed55_2025.json (ref_s, trimmed) and the plain mean (ref_mean_s), with
                       passages per day (max / punct / feed). Same as the prototype's 'val', unrounded.
  drops.parquet        row drop reasons (place + track) and feed stats (snapshots, capped gaps).
  summary.json         days, counts, headline numbers.

Semantics replicated here (see the prototype for the code):
  - Service day D = D 04:00 -> D+1 03:00 Europe/Brussels (the vd55 file). dow = weekday of the
    service day; hour = local clock hour of the snapshot (so 00:30 is hour 0 of the previous dow).
  - Frozen snapshots were never fetched (vd55 holds rows of non-identical snapshots only); each row
    is credited with the gap to the next non-identical snapshot of the feed (any line), capped at
    40 s, the last snapshot of the day gets 20 s. One row = one observation, whatever its credit.
  - Dropped: rows on a point outside the main pattern or at/past the vehicle's own terminus; rows
    < 10 m after the first stop (layover); short workings in the last quarter of the link that
    leads to their own terminus; track starts standing >= 100 s at an intermediate stop.
  - Feed metres are rescaled per link to GTFS metres via flen (p99.5 over the run's days, so it
    depends on the day set); dist < 10 m snaps to the stop, and a row at a stop counts in the last
    bucket of the link arriving there (half-open buckets (lo, hi]).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_PROTO = Path.home() / "Documents/Dev/CoDE/BrusselsTransitStuck"


def args_():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proto", default=str(DEFAULT_PROTO))
    ap.add_argument("--data", default=None, help="fixture dir (vd55/, gtfs/, punctuality/, speed55_ref.json)")
    ap.add_argument("--days", default=None, help="comma-separated YYYYMMDD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-fixture", default=None)
    return ap.parse_args()


def import_proto(proto):
    """Import the prototype modules (analysis.py reads data/gtfs at import: cwd must be proto)."""
    os.chdir(proto)
    sys.path.insert(0, str(proto))
    import speed55_2025 as S
    import time30m_2025 as T
    return S, T


def fixture_load(S, root):
    """speed55_2025.load() on fixture punctuality: same columns, filters and sort, without the
    prototype's file selection (size >= 100 kB, first start_date in FIRST..LAST), which only
    skips broken files and files outside the period."""
    cols = ["journey_id", "trip_id", "start_date", "stop_sequence", "stop_id", "arrival_time",
            "departure_time", "observed"]

    def load():
        fr = [pd.read_parquet(f, columns=cols, filters=[("route_id", "==", S.LINE)])
              for f in sorted(glob.glob(f"{root}/punctuality/*.parquet"))]
        p = pd.concat(fr, ignore_index=True)
        p = p[pd.to_datetime(p["start_date"]).dt.dayofweek.lt(5) & ~p["start_date"].isin(S.HOLIDAYS)]
        return p.sort_values(["start_date", "journey_id", "trip_id", "stop_sequence"])
    return load


def write_fixture(S, T, days, dest):
    """Cut a small fixture for `days` from the prototype data (cwd = proto)."""
    from analysis import digits
    dest = Path(dest)
    for sub in ("vd55", "gtfs", "punctuality"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    pts = set()
    for d in days:
        for suf in (".parquet", ".snaps.parquet"):
            shutil.copyfile(f"data2025/stib/vd55/{d}{suf}", dest / "vd55" / f"{d}{suf}")
        x = pd.read_parquet(f"data2025/stib/vd55/{d}.parquet")
        pts |= {digits(v) for v in x["point"]} | {digits(v) for v in x["dir"]}
    p = S.load()
    p = p[p["start_date"].isin(days)].copy()
    p["route_id"] = S.LINE
    p.reset_index(drop=True).to_parquet(dest / "punctuality" / "route55.parquet", compression="zstd")
    G = S.G
    r = pd.read_parquet(f"{G}/routes.parquet")
    r = r[r.route_short_name == S.LINE]
    t = pd.read_parquet(f"{G}/trips.parquet")
    t = t[t.route_id.isin(r.route_id)]
    st = pd.read_parquet(f"{G}/stop_times.parquet")
    st = st[st.trip_id.isin(t.trip_id)]
    pts |= {digits(v) for v in st.stop_id}
    stops = pd.read_parquet(f"{G}/stops.parquet")
    stops = stops[stops.stop_id.map(digits).isin(pts)]
    sh = pd.read_parquet(f"{G}/shapes.parquet")
    sh = sh[sh.shape_id.isin(t.shape_id)]
    tabs = {"routes": r, "trips": t, "stop_times": st, "stops": stops, "shapes": sh,
            "agency": pd.read_parquet(f"{G}/agency.parquet")}
    for n in ("calendar", "calendar_dates"):
        c = pd.read_parquet(f"{G}/{n}.parquet")
        tabs[n] = c[c.service_id.isin(t.service_id)]
    for n, df in tabs.items():
        df.reset_index(drop=True).to_parquet(dest / "gtfs" / f"{n}.parquet", compression="zstd")
    # Validation reference (]A-B] trimmed mean, band 'jour', full period) from out/speed55_2025.json.
    S_ = json.load(open("out/speed55_2025.json"))
    ref = {"source": "BrusselsTransitStuck/out/speed55_2025.json (full period 20250217-20250516)",
           "dirs": [{"dir": s["dir"], "links": [{"from": l["from"], "to": l["to"], "jour": l["jour"]}
                                                for l in s["links"]]} for s in S_["dirs"]]}
    json.dump(ref, open(dest / "speed55_ref.json", "w"), indent=0)
    tot = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file() and "golden" not in f.parts)
    print(f"fixture written to {dest}: {tot / 1e3:.0f} kB")


def run(S, T, days_arg, ref_path):
    load = T.load
    all_days = sorted(set(load()["start_date"].unique()) & {Path(f).stem for f in glob.glob("data2025/stib/vd55/*[0-9].parquet")})
    if days_arg:
        missing = sorted(set(days_arg) - set(all_days))
        if missing:
            raise SystemExit(f"days without both vd55 and punctuality: {missing}")
        days = sorted(days_arg)
    else:
        days = all_days
    nd = len(days)
    # --- exactly time30m_2025.main() up to the per-direction loop ---
    dirs, names, ll = T.geometry()
    x, stats = T.load_feed(days)
    x, flen, why = T.place(x, dirs, names)
    npass, drop = T.track(x, dirs, nd)
    why["layover_short_working_start"] = len(drop)
    x = x.drop(index=drop)
    x = x[x["credit"]]
    why["kept"] = len(x)
    npunct, refmean = T.passages(days)
    Sref = json.load(open(ref_path))
    BANDS, HOURS, STEP, STOPZONE = T.BANDS, T.HOURS, T.STEP, T.STOPZONE

    buckets, obs, pas_rows, flen_rows, val_rows, wk_check, headline = [], [], [], [], [], {}, {}
    for k, D in enumerate(dirs):
        seq = D["seq"]
        nl = len(seq) - 1
        B = []
        for i in range(nl):
            L = D["len"][i]
            e = list(np.arange(0, L, STEP)) + [L]
            if e[-1] - e[-2] < 1:
                e.pop(-2)
            for lo, hi in zip(e[:-1], e[1:]):
                B.append((i, lo, hi))
        b_link = np.array([b[0] for b in B])
        b_x0 = np.array([D["cum"][b[0]] + b[1] for b in B])
        b_len = np.array([b[2] - b[1] for b in B])
        nb = len(B)
        for j, (i, lo, hi) in enumerate(B):
            buckets.append({"dir": D["dir"], "dir_idx": k, "bucket": j, "link": i, "a": seq[i], "b": seq[i + 1],
                            "lo": float(lo), "hi": float(hi), "x0": float(b_x0[j]), "len": float(b_len[j])})
        xd = x[x["d"] == k]
        first_of = {i: int(np.argmax(b_link == i)) for i in range(nl)}
        n_of = {i: int((b_link == i).sum()) for i in range(nl)}
        pos, li = xd["pos"].to_numpy(), xd["i"].to_numpy()
        j = np.ceil(pos / STEP).astype(int) - 1
        at_stop = j < 0
        j = np.minimum(j, np.array([n_of[i] for i in li]) - 1)
        bi = np.array([first_of[i] for i in li]) + j
        bi[at_stop] = np.array([first_of[i] for i in li[at_stop]]) - 1
        assert (bi >= 0).all()
        gpos = D["cum"][li] + pos
        near = np.abs(gpos[:, None] - D["cum"][None, :]).min(axis=1) <= STOPZONE
        hr = xd["hour"].to_numpy()
        dt = xd["dt"].to_numpy()
        dw = np.array([pd.Timestamp(v).dayofweek for v in xd["day"].to_numpy()])
        obs.append(pd.DataFrame({"dir": D["dir"], "bucket": bi, "day": xd["day"].to_numpy(), "dow": dw,
                                 "hour": hr, "near": near, "dt": dt}))
        # Passages per link / weekday / hour, the three counts.
        for i in range(nl):
            for w in range(5):
                for h in range(24):
                    key = (seq[i], seq[i + 1], w, h)
                    nf, npu = float(npass.get(key, 0)), float(npunct.get(key, 0))
                    pas_rows.append({"dir": D["dir"], "link": i, "a": seq[i], "b": seq[i + 1], "dow": w, "hour": h,
                                     "n_feed": nf, "n_punct": npu, "n_max": max(nf, npu)})
            flen_rows.append({"dir": D["dir"], "link": i, "a": seq[i], "b": seq[i + 1], "glen": float(D["len"][i]),
                              "flen": float(flen.get((k, i), np.nan))})

        # Prototype quantities, for the validation table and the self-check against its JSON.
        def acc(hrs, mask=None):
            m = np.isin(hr, list(hrs)) if mask is None else np.isin(hr, list(hrs)) & mask
            return np.bincount(bi[m], weights=dt[m], minlength=nb) / nd

        def pas_src(hrs, src):
            return np.array([sum(src.get((seq[i], seq[i + 1], w, h), 0) for h in hrs for w in range(5)) / nd for i in range(nl)])

        def pas(hrs):
            return np.array([sum(max(npass.get((seq[i], seq[i + 1], w, h), 0), npunct.get((seq[i], seq[i + 1], w, h), 0))
                                 for h in hrs for w in range(5)) / nd for i in range(nl)])

        wk_check[D["dir"]] = {
            "ts": [[np.round(np.bincount(bi[(hr == h) & (dw == w)], weights=dt[(hr == h) & (dw == w)], minlength=nb)).astype(int).tolist()
                    for h in HOURS] for w in range(5)],
            "p": [[[int(round(max(npass.get((seq[i], seq[i + 1], w, h), 0), npunct.get((seq[i], seq[i + 1], w, h), 0))))
                    for i in range(nl)] for h in HOURS] for w in range(5)]}
        ts, ps = acc(BANDS["jour"]), pas(BANDS["jour"])
        Sd = next(s for s in Sref["dirs"] if s["dir"] == D["dir"])
        pp, pf = pas_src(BANDS["jour"], npunct), pas_src(BANDS["jour"], npass)
        for i in range(nl):
            sp = ts[b_link == i].sum() / ps[i] if ps[i] else np.nan
            ref = Sd["links"][i]["jour"]["s"] if Sd["links"][i]["jour"] else np.nan
            rm = float(refmean.get((seq[i], seq[i + 1]), np.nan))
            val_rows.append({"dir": D["dir"], "link": i, "a": seq[i], "b": seq[i + 1],
                             "from": names[seq[i]].title(), "to": names[seq[i + 1]].title(),
                             "buckets_s": float(sp), "ref_s": ref, "ref_mean_s": rm, "ratio": float(sp / ref),
                             "ratio_mean": float(sp / rm), "pass_day": float(ps[i]), "pass_punct_day": float(pp[i]),
                             "pass_feed_day": float(pf[i])})

        # Headline: counts per direction, stop-zone share, pm excess per passage vs evening.
        n_run = np.bincount(bi[~near], minlength=nb).astype(float)
        cnt = lambda hrs: np.bincount(bi[np.isin(hr, list(hrs)) & ~near], minlength=nb).astype(float)
        P = lambda hrs: pas(hrs) * nd  # passages summed over days
        O_pm, r_ev = cnt(BANDS["pm"]) / P(BANDS["pm"])[b_link], cnt(BANDS["soir"]) / P(BANDS["soir"])[b_link]
        exc = O_pm - r_ev
        top = np.argsort(-np.nan_to_num(exc, nan=-np.inf))[:5]
        headline[str(D["dir"])] = {
            "from": names[seq[0]].title(), "to": names[seq[-1]].title(), "n_buckets": nb, "n_links": nl,
            "length_m": float(D["cum"][-1]),
            "n_obs": int(len(bi)), "n_obs_run": int(n_run.sum()), "share_obs_stopzone": float(near.mean()),
            "share_ts_stopzone": float(dt[near].sum() / dt.sum()),
            "n_obs_hours_5_23": int(np.isin(hr, HOURS).sum()),
            "top5_pm_excess_per_passage_run": [
                {"bucket": int(b), "link": int(b_link[b]), "from": names[seq[b_link[b]]].title(),
                 "to": names[seq[b_link[b] + 1]].title(), "lo_m": round(float(b_x0[b] - D["cum"][b_link[b]]), 1),
                 "len_m": round(float(b_len[b]), 1), "O_pm": round(float(O_pm[b]), 3), "r_soir": round(float(r_ev[b]), 3),
                 "excess_obs_per_passage": round(float(exc[b]), 3)} for b in top]}

    ob = pd.concat(obs, ignore_index=True)
    ob["dt_run"] = np.where(ob["near"], 0.0, ob["dt"])
    ob["run"] = ~ob["near"]

    def agg(keys):
        return (ob.groupby(keys).agg(n_obs=("dt", "size"), n_obs_run=("run", "sum"), ts=("dt", "sum"), ts_run=("dt_run", "sum"))
                .reset_index().astype({"n_obs": "int64", "n_obs_run": "int64"}))
    tabs = {
        "buckets": pd.DataFrame(buckets),
        "obs_dow": agg(["dir", "bucket", "dow", "hour"]),
        "obs_day": agg(["dir", "bucket", "day", "hour"]),
        "passages": pd.DataFrame(pas_rows),
        "flen": pd.DataFrame(flen_rows),
        "validation": pd.DataFrame(val_rows),
        "drops": pd.DataFrame([{"what": "row_drop", "reason": k, "n": int(v)} for k, v in why.items()]
                              + [{"what": "feed", "reason": k, "n": float(v)} for k, v in stats.items()]),
    }
    summary = {"line": T.LINE, "days": days, "n_days": nd,
               "per_dow": [sum(pd.Timestamp(v).dayofweek == w for v in days) for w in range(5)],
               "params": {"step": STEP, "cap": T.CAP, "stopzone": STOPZONE, "layover_m": T.LAYOVER_M,
                          "at_stop_m": T.AT_STOP_M, "ghost": T.GHOST, "bands": {k: [min(v), max(v) + 1] for k, v in BANDS.items()},
                          "hours_shown": [HOURS[0], HOURS[-1] + 1]},
               "feed": stats, "rows": why, "dirs": headline,
               "obs_hours": {str(h): int(n) for h, n in ob.groupby("hour").size().items()}}
    return tabs, summary, wk_check, val_rows


def self_check(wk_check, val_rows, days, proto):
    """Compare with the prototype's own out/time30m_2025.json when it covers the same days."""
    f = Path(proto) / "out/time30m_2025.json"
    if not f.exists():
        return "no prototype json"
    J = json.load(open(f))
    if (J["period"]["first"], J["period"]["last"], J["period"]["weekdays"]) != (days[0], days[-1], len(days)):
        return "skipped (prototype json covers another period)"
    bad = []
    for d in J["dirs"]:
        w = wk_check[d["dir"]]
        if w["ts"] != d["wk"]["ts"]:
            bad.append(f"dir {d['dir']} wk.ts")
        if w["p"] != d["wk"]["p"]:
            bad.append(f"dir {d['dir']} wk.p")
        mine = [r for r in val_rows if r["dir"] == d["dir"]]
        for a, b in zip(mine, d["val"]):
            if abs(round(a["buckets_s"], 1) - b["buckets_s"]) > 0.051 or abs(round(a["pass_day"], 1) - b["pass_day"]) > 0.051:
                bad.append(f"dir {d['dir']} val {b['from']}->{b['to']}")
    return "OK: wk.ts, wk.p and val identical to out/time30m_2025.json" if not bad else "MISMATCH: " + ", ".join(bad)


def main():
    a = args_()
    t0 = time.time()
    proto = Path(a.proto).expanduser().resolve()
    S, T = import_proto(proto)
    days = a.days.split(",") if a.days else None
    if a.write_fixture:
        if not days:
            raise SystemExit("--write-fixture needs --days")
        write_fixture(S, T, days, Path(a.write_fixture).expanduser().resolve())
        return
    ref_path = proto / "out/speed55_2025.json"
    if a.data:
        # The prototype reads fixed relative paths (data2025/...): run it in a scratch dir whose
        # data2025/ points at the fixture.
        root = Path(a.data).expanduser().resolve()
        wd = Path(tempfile.mkdtemp(prefix="golden55_"))
        (wd / "data2025/stib").mkdir(parents=True)
        os.symlink(root / "gtfs", wd / "data2025/gtfs")
        os.symlink(root / "vd55", wd / "data2025/stib/vd55")
        os.symlink(root / "punctuality", wd / "data2025/stib/punctuality")
        os.chdir(wd)
        T.load = S.load = fixture_load(S, root)
        ref_path = root / "speed55_ref.json"
    out = Path(a.out).expanduser().resolve() if a.out else HERE / "golden"
    tabs, summary, wk_check, val_rows = run(S, T, days, ref_path)
    out.mkdir(parents=True, exist_ok=True)
    summary["self_check"] = self_check(wk_check, val_rows, summary["days"], proto)
    summary["source"] = {"proto": str(proto), "data": a.data or str(proto / "data2025"), "runtime_s": round(time.time() - t0, 1)}
    for n, df in tabs.items():
        df.to_parquet(out / f"{n}.parquet", index=False, compression="zstd")
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: summary[k] for k in ("n_days", "rows", "self_check", "source")}, indent=1))
    for n in sorted(out.iterdir()):
        print(f"  {n.name:22} {n.stat().st_size / 1e3:8.1f} kB")


if __name__ == "__main__":
    sys.exit(main())
