# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor - Experimental Data Example

Analyzes experimental XGI wavefront sensing data.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import run_pipeline
from core import calculate_wavelength


def main():
    params = dict()

    # Detector Configuration
    params["pixel_size"] = [0.715e-6, 0.715e-6]  # Pixel size (m)
    params["wavelength"] = calculate_wavelength(7100)  # 7100 eV

    # Optical System Geometry
    params["det2sample"] = 0.35  # Grating-to-detector distance (m)
    params["total_dist"] = 6.5  # Source-to-detector distance (m)
    params["source_dist"] = params["total_dist"] - params["det2sample"]

    # Grating Parameters
    period = 18.38e-6  # Base grating period (m)
    params["grating_period"] = period / 2
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    # Data Source
    img_path = str(Path(__file__).parent.parent / "data" / "sample_exp.tif")
    params.update(
        {
            "image_path": img_path,
            "dark_image_path": None,
            "flat_image_path": None,
        }
    )

    # Run pipeline
    run_pipeline(params, output_prefix="results_exp")


if __name__ == "__main__":
    main()
