# %% md
# X-ray Grating Interferometry (XGI) Wavefront Sensor Analysis
#
# Overview:
# This module implements a complete pipeline for XGI wavefront sensing data analysis.
#
# Workflow:
# 1. Parameter Configuration:
#    - Optical system geometry
#    - Detector specifications
#    - Grating characteristics
#
# 2. Data Processing Pipeline:
#    - Image preprocessing and normalization
#    - Grating interference pattern analysis
#    - Wavefront reconstruction
#
# 3. Output Analysis:
#    - Phase map reconstruction
#    - Wavefront error analysis
#    - Beam characteristics (position/size)
#    - Focus parameters
#
# Input Requirements:
# - Raw image data (TIFF format)
# - Optional: Dark field and flat field corrections
#
# Output Products:
# - Reconstructed phase map
# - Wavefront fitting and error analysis
# - Beam position and size measurements
# - Focus characteristics

import threading
from queue import Queue

from pipline import task
from core.phase_fit import plot_phase_error_profiles
from core.utils import calculate_wavelength


def _output1_worker(queue: Queue) -> None:
    """
    Process and display focus adjustment parameters.

    Args:
        queue (Queue): Thread-safe queue containing focus adjustment values
    """
    while True:
        value = queue.get()
        if value is None:
            # Sentinel value received: terminate worker
            queue.task_done()
            break
        print("focus_adjust:", *value)
        queue.task_done()


def _output2_worker(queue: Queue) -> None:
    """
    Generate and display phase error profile visualizations.

    Args:
        queue (Queue): Thread-safe queue containing phase error data
    """
    while True:
        value = queue.get()
        if value is None:
            # Sentinel value received: terminate worker
            queue.task_done()
            break
        plot_phase_error_profiles(*value)
        queue.task_done()


def _output3_worker(queue: Queue) -> None:
    """
    Process and display focus position and size measurements.

    Args:
        queue (Queue): Thread-safe queue containing focus analysis results
    """
    while True:
        value = queue.get()
        if value is None:
            # Sentinel value received: terminate worker
            queue.task_done()
            break
        focus_position, focus_size = value
        print(f"focus_position:{focus_position}; focus_size:{focus_size}")
        queue.task_done()


def main():
    # Initialize System Parameters
    params = dict()

    # Detector Configuration
    params['pixel_size'] = [0.715e-6, 0.715e-6]  # Pixel size in meters (x, y)
    params['wavelength'] = calculate_wavelength(7100)  # X-ray wavelength (eV to m)

    # Optical System Geometry
    params['det2sample'] = 0.35  # Grating-to-detector distance (m)
    params['total_dist'] = 6.5  # Source-to-detector distance (m)
    params['source_dist'] = params['total_dist'] - params['det2sample']  # Source-to-grating distance (m)

    # Grating Parameters
    period = 18.38e-6  # Base grating period (m)
    params['grating_period'] = period / 2  # Effective grating period
    # Calculate expected self-imaging period based on geometry
    params['pattern_period'] = params['grating_period'] * params['total_dist'] / params['source_dist']

    # Data Source Configuration
    img_path = "sample_exp.tif"
    params.update({
        'image_path': img_path,
        "dark_image_path": None,  # Optional dark field correction
        "flat_image_path": None,  # Optional flat field correction
    })

    # Initialize Multi-threaded Output Processing
    queues = {
        "output1": Queue(),  # Focus adjustment parameters
        "output2": Queue(),  # Phase error profiles
        "output3": Queue(),  # Focus analysis results
    }

    # Configure Worker Threads
    workers = {
        "output1": threading.Thread(target=_output1_worker, args=(queues["output1"],), name="output1"),
        "output2": threading.Thread(target=_output2_worker, args=(queues["output2"],), name="output2"),
        "output3": threading.Thread(target=_output3_worker, args=(queues["output3"],), name="output3"),
    }

    # Launch Processing Threads
    for thread in workers.values():
        # Initialize parallel processing for each output type
        thread.start()

    try:
        # Execute Main Processing Pipeline
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
