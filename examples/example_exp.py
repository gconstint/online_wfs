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

# Set matplotlib backend before importing pyplot (required for macOS threading)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for thread safety

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
from queue import Queue

from pipeline import task
from core import plot_phase_error_profiles, calculate_wavelength,plot_phase_fit_results

# Output directory for saved plots
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _wavefront_worker(queue: Queue) -> None:
    """
    Process wavefront fitting results (checkpoint_wavefront).
    Saves phase error profiles and phase fit results to output directory.

    Args:
        queue (Queue): Thread-safe queue containing wavefront data
    """
    while True:
        data = queue.get()
        if data is None:
            queue.task_done()
            break
        # Extract wavefront data
        phase = data.get("phase")
        fitted_phase = data.get("fitted_phase")
        phase_error = data.get("phase_error")
        fit_params = data.get("fit_params")
        virtual_pixel_size = data.get("virtual_pixel_size")
        wavelength = data.get("wavelength")

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = OUTPUT_DIR / f"wavefront_exp_{timestamp}"
        save_dir.mkdir(exist_ok=True)

        # Save phase fit results
        if phase is not None and fitted_phase is not None and fit_params is not None:
            plot_phase_fit_results(
                phase,
                fitted_phase,
                fit_params,
                pixel_size=virtual_pixel_size,
                save_path=str(save_dir),
            )

        # Save phase error profiles
        if phase_error is not None:
            plot_phase_error_profiles(
                phase_error, virtual_pixel_size, wavelength, save_path=str(save_dir)
            )

        print(f"[Wavefront] Results saved to: {save_dir}")
        queue.task_done()


def _aberration_worker(queue: Queue) -> None:
    """
    Process aberration analysis results (checkpoint_aberration).

    Args:
        queue (Queue): Thread-safe queue containing aberration data
    """
    while True:
        data = queue.get()
        if data is None:
            queue.task_done()
            break
        # Extract aberration data
        calibration_result = data.get("calibration_result")
        zernike_results = data.get("zernike_results")
        # Process: e.g., print focus adjustment
        if calibration_result is not None:
            R_avg = calibration_result.get("R_avg", calibration_result.get("R"))
            Delta_avg = calibration_result.get(
                "Delta_avg", calibration_result.get("Delta_z")
            )
            if R_avg is not None:
                print(f"[Aberration] Focus distance: {R_avg * 1e3:.2f} mm")
            if Delta_avg is not None:
                print(f"[Aberration] Focus offset: {Delta_avg * 1e3:.2f} mm")
        if zernike_results is not None:
            # Print RMS residual error
            rms_error = zernike_results.get("rms_error")
            if rms_error is not None:
                print(f"[Aberration] Zernike RMS residual: {rms_error:.4f} rad")

            # Print top aberrations from aberration_analysis
            aberration_analysis = zernike_results.get("aberration_analysis")
            if aberration_analysis:
                # Sort by RMS magnitude and print top 5
                sorted_aberr = sorted(
                    aberration_analysis.items(),
                    key=lambda x: abs(x[1]["rms_nm"]),
                    reverse=True,
                )
                print("[Aberration] Top 5 Zernike terms:")
                for key, aberr in sorted_aberr[:5]:
                    name = aberr["name"]
                    rms_nm = aberr["rms_nm"]
                    print(f"  {key}: {name} = {rms_nm:.3f} nm RMS")
        queue.task_done()


def _focus_worker(queue: Queue) -> None:
    """
    Process focus analysis results (checkpoint_focus).

    Args:
        queue (Queue): Thread-safe queue containing focus analysis results
    """
    while True:
        data = queue.get()
        if data is None:
            queue.task_done()
            break
        # Extract focus data
        focus_position = data.get("focus_position")
        focus_size = data.get("focus_size")
        beam_position = data.get("beam_position")
        beam_size = data.get("beam_size")
        # Process: e.g., print focus info
        if focus_position is not None:
            print(
                f"[Focus] Position: ({focus_position[0] * 1e9:.1f}, {focus_position[1] * 1e9:.1f}) nm"
            )
        if focus_size is not None:
            print(
                f"[Focus] Size (FWHM): {focus_size['fwhm_x'] * 1e9:.1f} x {focus_size['fwhm_y'] * 1e9:.1f} nm"
            )
        if beam_position is not None:
            print(
                f"[Imager] Beam position: ({beam_position[0] * 1e6:.3f}, {beam_position[1] * 1e6:.3f}) um"
            )
        if beam_size is not None:
            print(
                f"[Imager] Beam size (FWHM): {beam_size['fwhm_x'] * 1e6:.3f} x {beam_size['fwhm_y'] * 1e6:.3f} um"
            )
        queue.task_done()


def main():
    # Initialize System Parameters
    params = dict()

    # Detector Configuration
    params["pixel_size"] = [0.715e-6, 0.715e-6]  # Pixel size in meters (x, y)
    params["wavelength"] = calculate_wavelength(7100)  # X-ray wavelength (eV to m)

    # Optical System Geometry
    params["det2sample"] = 0.35  # Grating-to-detector distance (m)
    params["total_dist"] = 6.5  # Source-to-detector distance (m)
    params["source_dist"] = (
        params["total_dist"] - params["det2sample"]
    )  # Source-to-grating distance (m)

    # Grating Parameters
    period = 18.38e-6  # Base grating period (m)
    params["grating_period"] = period / 2  # Effective grating period
    # Calculate expected self-imaging period based on geometry
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    # Data Source Configuration
    img_path = str(Path(__file__).parent.parent / "data" / "sample_exp.tif")
    params.update(
        {
            "image_path": img_path,
            "dark_image_path": None,  # Optional dark field correction
            "flat_image_path": None,  # Optional flat field correction
        }
    )

    # Initialize Multi-threaded Output Processing
    # Map checkpoint names to their queues
    queues = {
        "checkpoint_wavefront": Queue(),  # Wavefront fitting results
        "checkpoint_aberration": Queue(),  # Aberration analysis results
        "checkpoint_focus": Queue(),  # Focus analysis results
    }

    # Configure Worker Threads
    workers = {
        "checkpoint_wavefront": threading.Thread(
            target=_wavefront_worker,
            args=(queues["checkpoint_wavefront"],),
            name="wavefront_worker",
            daemon=True,  # Daemon thread for clean exit
        ),
        "checkpoint_aberration": threading.Thread(
            target=_aberration_worker,
            args=(queues["checkpoint_aberration"],),
            name="aberration_worker",
            daemon=True,
        ),
        "checkpoint_focus": threading.Thread(
            target=_focus_worker,
            args=(queues["checkpoint_focus"],),
            name="focus_worker",
            daemon=True,
        ),
    }

    # Launch Processing Threads
    for thread in workers.values():
        thread.start()

    try:
        # Execute Main Processing Pipeline
        for checkpoint_name, data in task(params, verbose=False, show_plots=False):
            queue = queues.get(checkpoint_name)
            if queue is None:
                # Unknown checkpoint, skip
                continue
            # Distribute results to appropriate processing threads (non-blocking)
            queue.put(data)
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
            thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
