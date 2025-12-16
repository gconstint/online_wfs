# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor Pipeline Runner

This module provides the common infrastructure for running XGI wavefront sensor
analysis with multi-threaded output processing. Example scripts use this module
with different parameter configurations.

Usage:
    from runner import run_pipeline
    params = {...}  # Configure parameters
    run_pipeline(params, output_prefix="exp")
"""

# Set matplotlib backend before importing pyplot (required for macOS threading)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for thread safety

import threading
from pathlib import Path
from datetime import datetime
from queue import Queue

import numpy as np

from pipeline import task
from core import (
    plot_phase_error_profiles,
    plot_phase_fit_results,
    visualize_zernike_analysis,
    plot_beam_visualization,
)

# Output directory for saved plots
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Shared save directory for current run (set by wavefront_worker, used by other workers)
_current_save_dir = None
_output_prefix = "results"


def _wavefront_worker(queue: Queue) -> None:
    """
    Process wavefront fitting results (checkpoint_wavefront).
    Saves phase error profiles and phase fit results to output directory.
    """
    global _current_save_dir
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

        # Create output directory with timestamp (shared with other workers)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = OUTPUT_DIR / f"{_output_prefix}_{timestamp}"
        save_dir.mkdir(exist_ok=True)
        _current_save_dir = save_dir  # Set for other workers to use

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
    Saves Zernike analysis plots to shared output directory.
    """
    while True:
        data = queue.get()
        if data is None:
            queue.task_done()
            break

        # Extract aberration data
        calibration_result = data.get("calibration_result")
        zernike_results = data.get("zernike_results")
        roi_result = data.get("roi_result")
        phase_error = data.get("phase_error")
        wavelength = data.get("wavelength")
        virtual_pixel_size = data.get("virtual_pixel_size")

        # Print focus calibration results
        if calibration_result is not None:
            R_avg = calibration_result.get("R_avg", calibration_result.get("R"))
            Delta_avg = calibration_result.get(
                "Delta_avg", calibration_result.get("Delta_z")
            )
            if R_avg is not None:
                print(f"[Aberration] Focus distance: {R_avg * 1e3:.2f} mm")
            if Delta_avg is not None:
                print(f"[Aberration] Focus offset: {Delta_avg * 1e3:.2f} mm")

        # Print ROI info if available
        if roi_result is not None:
            aperture_radius = roi_result.get("aperture_radius")
            if aperture_radius is not None:
                print(f"[Aberration] ROI radius: {aperture_radius * 1e6:.2f} um")

        if zernike_results is not None:
            # Print RMS residual error
            rms_error = zernike_results.get("rms_error")
            if rms_error is not None:
                print(f"[Aberration] Zernike RMS residual: {rms_error:.4f} rad")

            # Print top aberrations
            aberration_analysis = zernike_results.get("aberration_analysis")
            if aberration_analysis:
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

            # Save Zernike visualization
            fitted_phase = zernike_results.get("fitted_phase")
            residual = zernike_results.get("residual")
            phase_for_viz = (
                roi_result.get("phase_error_cropped")
                if roi_result is not None
                else phase_error
            )

            if (
                phase_for_viz is not None
                and fitted_phase is not None
                and residual is not None
                and wavelength is not None
                and virtual_pixel_size is not None
            ):
                save_dir = _current_save_dir
                if save_dir is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = OUTPUT_DIR / f"{_output_prefix}_{timestamp}"
                    save_dir.mkdir(exist_ok=True)

                visualize_zernike_analysis(
                    phase=phase_for_viz,
                    fitted_phase=fitted_phase,
                    residual=residual,
                    aberration_analysis=aberration_analysis,
                    wavelength=wavelength,
                    pixel_size=virtual_pixel_size,
                    title="Zernike Aberration Analysis",
                    save_path=str(save_dir),
                    verbose=False,
                )
                print(f"[Aberration] Zernike plots saved to: {save_dir}")

        queue.task_done()


def _focus_worker(queue: Queue) -> None:
    """
    Process focus analysis results (checkpoint_focus).
    Saves focus beam visualization to shared output directory.
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
        focus_field = data.get("focus_field")
        dx_focus = data.get("dx_focus")
        dy_focus = data.get("dy_focus")

        # Print focus info
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

        # Save focus beam visualization
        if (
            focus_field is not None
            and focus_size is not None
            and focus_position is not None
            and dx_focus is not None
            and dy_focus is not None
        ):
            save_dir = _current_save_dir
            if save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = OUTPUT_DIR / f"{_output_prefix}_{timestamp}"
                save_dir.mkdir(exist_ok=True)

            # Convert position to microns for visualization
            beam_x_um = focus_position[0] * 1e6
            beam_y_um = focus_position[1] * 1e6
            fwhm_x_um = focus_size["fwhm_x"] * 1e6
            fwhm_y_um = focus_size["fwhm_y"] * 1e6

            plot_beam_visualization(
                intensity=np.abs(focus_field),
                virtual_pixel_size=(dx_focus, dy_focus),
                beam_x_um=beam_x_um,
                beam_y_um=beam_y_um,
                fwhm_x=fwhm_x_um,
                fwhm_y=fwhm_y_um,
                fit_params_x=focus_size.get("fit_params_x"),
                fit_params_y=focus_size.get("fit_params_y"),
                title="Focus Beam Analysis",
                save_path=str(save_dir),
            )
            print(f"[Focus] Beam plot saved to: {save_dir}")

        queue.task_done()


def run_pipeline(params: dict, output_prefix: str = "results") -> None:
    """
    Execute the XGI wavefront sensor pipeline with multi-threaded output processing.

    Args:
        params: Dictionary of system parameters including:
            - pixel_size: Detector pixel size [m]
            - wavelength: X-ray wavelength [m]
            - det2sample: Grating-to-detector distance [m]
            - total_dist: Source-to-detector distance [m]
            - source_dist: Source-to-grating distance [m]
            - grating_period: Effective grating period [m]
            - pattern_period: Expected self-imaging period [m]
            - image_path: Path to input image
            - dark_image_path: Optional dark field image path
            - flat_image_path: Optional flat field image path
        output_prefix: Prefix for output directory name (default: "results")
    """
    global _output_prefix, _current_save_dir
    _output_prefix = output_prefix
    _current_save_dir = None

    # Initialize queues for multi-threaded processing
    queues = {
        "checkpoint_wavefront": Queue(),
        "checkpoint_aberration": Queue(),
        "checkpoint_focus": Queue(),
    }

    # Configure worker threads
    workers = {
        "checkpoint_wavefront": threading.Thread(
            target=_wavefront_worker,
            args=(queues["checkpoint_wavefront"],),
            name="wavefront_worker",
            daemon=True,
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

    # Launch processing threads
    for thread in workers.values():
        thread.start()

    try:
        # Execute main processing pipeline
        for checkpoint_name, data in task(params, verbose=False, show_plots=False):
            queue = queues.get(checkpoint_name)
            if queue is not None:
                queue.put(data)
    finally:
        # Graceful shutdown
        for queue in queues.values():
            queue.put(None)
        for queue in queues.values():
            queue.join()
        for thread in workers.values():
            thread.join(timeout=1.0)
