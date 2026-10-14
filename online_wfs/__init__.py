"""Array-only absolute SGI wavefront calculations."""

from .config import normalize_config
from .pipeline import Pipeline

__all__ = ["Pipeline", "normalize_config"]
__version__ = "0.4.1"
