"""microsegments: count where transit vehicles are observed along a line, per micro-segment."""
from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

from .config import Config
from .schema import Flag

__all__ = ["Config", "Flag", "__version__"]
