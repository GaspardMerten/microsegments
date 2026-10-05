"""Generate the synthetic example data (offline) into examples/data/:

    python examples/make_synth.py
    microsegments run examples/synth.toml -o out/

A simulated tram line "S1" (two directions, 8 links), 3 weeks of STIB-like positions every ~20 s
with outages and a frozen feed, a GTFS change after two thirds of the period (the last two stops are cut), and three
injected hotspots (a signal, a peak-hour queue, an evening holding stop). See microsegments.simulate.
"""
from __future__ import annotations

import sys
from pathlib import Path

from microsegments.simulate import simulate

HERE = Path(__file__).resolve().parent


def main(out: Path = HERE / "data", days: int = 21, seed: int = 0) -> Path:
    sim = simulate(seed=seed, days=days, start="2025-03-03", mode="stib", out_dir=out / "gtfs", change_day=days * 2 // 3)
    sim.write_stib(out / "vd")
    sim.stop_events.write_parquet(out / "events.parquet")
    print(f"{len(sim.observations):,} positions, {len(sim.dates)} days -> {out}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data")
