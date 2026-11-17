"""Read detector arrays for the examples."""

from pathlib import Path

import numpy as np

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "online_wfs" / "data" / "sample.tif"


def load_image(path: Path, *, key: str = "image") -> np.ndarray:
    """Read a TIFF or a named array from an NPZ bundle."""
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if key not in arrays:
                raise ValueError(f"{path} is missing the {key!r} array")
            return arrays[key]
    import tifffile

    return tifffile.imread(path)
