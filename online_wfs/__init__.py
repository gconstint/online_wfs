"""Unattended absolute SGI wavefront reconstruction."""

from .config import load_config
from .pipeline import Pipeline

__all__ = ["Pipeline", "load_config"]
__version__ = "0.4.1"
