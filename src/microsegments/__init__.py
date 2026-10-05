"""microsegments: count where transit vehicles are observed along a line, per micro-segment.

Quick start::

    import microsegments as ms
    res = ms.run(ms.Config.from_toml("ms.toml"))
    ms.export(res, "report.html")
    ms.plot.matrix(res, direction_id=0)
"""
from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

from .aggregate import count
from .compare import Comparison, compare, run_compare
from .config import Config
from .hotspots import hotspots
from .html import export
from .io import read_observations
from .metrics import Analysis, analyse
from .network import build_network
from .pipeline import RunResult, prepare, run
from .report import to_contract
from .schema import Flag
from .segments import segment
from .simulate import simulate
from . import tune

__all__ = ["Analysis", "Comparison", "Config", "Flag", "RunResult", "__version__", "analyse", "build_network", "compare", "count", "export",
           "hotspots", "plot", "prepare", "read_observations", "run", "run_compare", "segment", "simulate", "to_contract", "tune"]


def __getattr__(name: str):
    # matplotlib is optional: import the plot module on first use
    if name == "plot":
        import importlib
        mod = importlib.import_module(".plot", __name__)
        globals()["plot"] = mod
        return mod
    raise AttributeError(f"module 'microsegments' has no attribute {name!r}")
