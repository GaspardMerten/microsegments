"""Input adapters: any source -> ``schema.OBSERVATION`` / ``SNAPSHOT`` / ``STOP_EVENT``, and feed coverage."""
from __future__ import annotations

from .coverage import coverage, coverage_from_config, snapshots_from_observations
from .tabular import read_observations, read_snapshots
from .timeutil import add_local_time

__all__ = ["read_observations", "read_snapshots", "add_local_time", "coverage", "coverage_from_config",
           "snapshots_from_observations"]
