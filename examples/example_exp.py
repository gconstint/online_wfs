# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor - Experimental Data Example

Analyzes experimental XGI wavefront sensing data.
Results saved to output/exp/.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.utils import load_images
from example_direct import analyze
from params import get_params


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "exp"


def main():
    # Get parameters
    params = get_params()

    # Load images separately (for real-time analysis, replace with EPICS/control system)
    img, dark, flat = load_images(
        params["image_path"],
        params["dark_image_path"],
        params["flat_image_path"],
    )

    # Run analysis with direct image input
    analyze(params, output_dir=OUTPUT_DIR, img=img, dark=dark, flat=flat)


if __name__ == "__main__":
    main()
