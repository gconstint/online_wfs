# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor - Simulation Data Example

Analyzes simulated XGI wavefront sensing data.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import run_pipeline
from core import calculate_wavelength


def main():
    params = dict()

    # Detector Configuration
    params["pixel_size"] = [1.3e-6, 1.3e-6]  # Pixel size (m)
    params["wavelength"] = calculate_wavelength(5000)  # 5000 eV

    # Optical System Geometry
    params["det2sample"] = 2.803  # Grating-to-detector distance (m)
    params["total_dist"] = 3.0  # Source-to-detector distance (m)
    params["source_dist"] = params["total_dist"] - params["det2sample"]

    # Grating Parameters
    period = 4e-6  # Base grating period (m)
    params["grating_period"] = period / np.sqrt(2)  # 45° rotation
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    # Data Source
    img_path = str(Path(__file__).parent.parent / "data" / "sample_sim.tif")
    params.update(
        {
            "image_path": img_path,
            "dark_image_path": None,
            "flat_image_path": None,
        }
    )

    # Run pipeline
    run_pipeline(params, output_prefix="results_sim")


if __name__ == "__main__":
    main()
