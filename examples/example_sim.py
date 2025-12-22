# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor - Simulation Data Example

Analyzes simulated XGI wavefront sensing data.
Results saved to output/sim/.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from example_direct import analyze
from params import get_sim_params


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sim"


def main():
    params = get_sim_params()
    analyze(params, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
