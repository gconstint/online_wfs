# %% md
# X-ray Grating Interferometry (XGI) Wavefront Sensor - Simulation Mode
#
# Purpose:
# Analysis of XGI wavefront-sensing data for simulated beam configurations.
#
# Processing Pipeline:
# 1. Configuration
#    - System geometry parameters
#    - Detector specifications
#    - Grating characteristics
#
# 2. Data Processing
#    - Image preprocessing and normalization
#    - Grating interference analysis
#    - Wavefront reconstruction
#
# 3. Analysis Outputs
#    - Phase map reconstruction
#    - Wavefront error analysis
#    - Beam characteristics
#    - Focus parameters
#
# Input Data:
# - Primary: Raw image data (TIFF format)
# - Optional: Dark field and flat field corrections
#
# Output Products:
# - Reconstructed phase maps
# - Wavefront fitting results
# - Error analysis plots
# - Beam position and size metrics
# - Focus characteristics

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import threading
from queue import Queue

from pipeline import task
from core import plot_phase_error_profiles, calculate_wavelength


def _output1_worker(queue: Queue) -> None:
    """
    Process and display focus adjustment parameters from simulation.

    Args:
        queue (Queue): Thread-safe queue containing focus adjustment values
    """
    while True:
        value = queue.get()
        if value is None:
            # Sentinel reached: worker can exit
            queue.task_done()
            break
        print("focus_adjust:", *value)
        queue.task_done()


def _output2_worker(queue: Queue) -> None:
    """
    Generate and display phase error profile visualizations from simulation.

    Args:
        queue (Queue): Thread-safe queue containing phase error data
    """
    while True:
        value = queue.get()
        if value is None:
            # Sentinel reached: worker can exit
            queue.task_done()
            break
        plot_phase_error_profiles(*value)
        queue.task_done()


def _output3_worker(queue: Queue) -> None:
    """
    Process and display simulated focus position and size measurements.

    Args:
        queue (Queue): Thread-safe queue containing focus analysis results
    """
    while True:
        value = queue.get()
        if value is None:
            # Sentinel reached: worker can exit
            queue.task_done()
            break
        focus_position, focus_size = value
        print(f"focus_position:{focus_position}; focus_size:{focus_size}")
        queue.task_done()


def main():
    # System Configuration Parameters
    params = dict()

    # Detector Specifications
    params["pixel_size"] = [1.3e-6, 1.3e-6]  # Detector pixel size (m)
    params["wavelength"] = calculate_wavelength(5000)  # X-ray wavelength (eV to m)

    # Optical System Geometry
    params["det2sample"] = 2.803  # Grating-to-detector distance (m)
    params["total_dist"] = 3.0  # Source-to-detector distance (m)
    params["source_dist"] = (
        params["total_dist"] - params["det2sample"]
    )  # Source-to-grating distance (m)

    # Grating Configuration
    period = 4e-6  # Base grating period (m)
    params["grating_period"] = period / np.sqrt(
        2
    )  # Effective grating period considering rotation
    # Calculate expected self-imaging period based on geometry
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    # Simulation Data Source
    img_path = str(Path(__file__).parent.parent / "data" / "sample_sim.tif")
    params.update(
        {
            "image_path": img_path,
            "dark_image_path": None,  # Optional dark field correction
            "flat_image_path": None,  # Optional flat field correction
        }
    )

    # Initialize Multi-threaded Output Processing
    queues = {
        "output1": Queue(),  # Focus adjustment parameters
        "output2": Queue(),  # Phase error profiles
        "output3": Queue(),  # Focus analysis results
    }

    # Configure Worker Threads
    workers = {
        "output1": threading.Thread(
            target=_output1_worker, args=(queues["output1"],), name="output1"
        ),
        "output2": threading.Thread(
            target=_output2_worker, args=(queues["output2"],), name="output2"
        ),
        "output3": threading.Thread(
            target=_output3_worker, args=(queues["output3"],), name="output3"
        ),
    }

    # Launch Processing Threads
    for thread in workers.values():
        # Initialize parallel processing for each output type
        thread.start()

    try:
        # Execute Simulation Pipeline
        for output, value in task(params):
            queue = queues.get(output)
            if queue is None:
                print(f"Unhandled output type: {output}")
                continue
            # Distribute results to appropriate processing threads
            queue.put(value)
    finally:
        # Graceful Shutdown Sequence
        for queue in queues.values():
            # Signal thread termination
            queue.put(None)
        for queue in queues.values():
            # Wait for queues to process remaining items
            queue.join()
        for thread in workers.values():
            # Ensure clean thread termination
            thread.join()


if __name__ == "__main__":
    main()
