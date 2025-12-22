# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor - Direct Pipeline Usage Module

This module provides reusable functions for:
1. Running the XGI pipeline and collecting results
2. Printing result summaries
3. Creating and saving plots

Can be run directly for demonstration, or imported by other examples.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pipeline import task


def run_pipeline(params: dict, verbose: bool = True) -> tuple:
    """
    Run the XGI pipeline and collect results from all checkpoints.

    Parameters
    ----------
    params : dict
        Configuration parameters
    verbose : bool
        Whether to print pipeline output

    Returns
    -------
    tuple
        (wavefront_data, aberration_data, focus_data)
    """
    wavefront_data = None
    aberration_data = None
    focus_data = None

    for checkpoint_name, data in task(params, verbose=verbose, show_plots=False):
        if checkpoint_name == "checkpoint_wavefront":
            wavefront_data = data
        elif checkpoint_name == "checkpoint_aberration":
            aberration_data = data
        elif checkpoint_name == "checkpoint_focus":
            focus_data = data

    return wavefront_data, aberration_data, focus_data


def print_results_summary(wavefront_data, aberration_data, focus_data):
    """
    Print a summary of pipeline results.
    """
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    if wavefront_data:
        fit_params = wavefront_data["fit_params"]
        x0, y0, Rx, Ry, A = fit_params
        print(f"\nWavefront Fitting:")
        print(f"  Curvature radius: Rx={Rx * 1e6:.1f} um, Ry={Ry * 1e6:.1f} um")
        print(f"  Center: ({x0 * 1e6:.1f}, {y0 * 1e6:.1f}) um")
        print(f"  Amplitude: {A:.2f} rad")

    if aberration_data:
        cal = aberration_data["calibration_result"]
        if cal:
            R_avg = cal.get("R_avg", cal.get("R", 0))
            print(f"\nFocus Calibration:")
            print(f"  Focus distance: {R_avg * 1e3:.2f} mm")

        zernike = aberration_data["zernike_results"]
        if zernike:
            print(f"\nZernike Analysis:")
            print(f"  RMS residual: {zernike.get('rms_error', 0):.4f} rad")

            aberr_analysis = zernike.get("aberration_analysis", {})
            if aberr_analysis:
                sorted_aberr = sorted(
                    aberr_analysis.items(),
                    key=lambda x: abs(x[1]["rms_nm"]),
                    reverse=True,
                )
                print("  Top 3 Zernike terms:")
                for key, aberr in sorted_aberr[:3]:
                    print(f"    {key}: {aberr['name']} = {aberr['rms_nm']:.3f} nm RMS")

    if focus_data:
        pos = focus_data["focus_position"]
        size = focus_data["focus_size"]
        print(f"\nFocus Analysis:")
        print(f"  Position: ({pos[0] * 1e9:.1f}, {pos[1] * 1e9:.1f}) nm")
        print(
            f"  Size (FWHM): {size['fwhm_x'] * 1e9:.1f} x {size['fwhm_y'] * 1e9:.1f} nm"
        )


def save_phase_maps(wavefront_data, output_dir: Path, filename: str = "phase_maps.png"):
    """
    Save phase maps comparison plot.
    """
    if not wavefront_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    phase = wavefront_data["phase"]
    fitted = wavefront_data["fitted_phase"]
    error = wavefront_data["phase_error"]

    im0 = axes[0].imshow(phase, cmap="RdBu_r")
    axes[0].set_title("Original Phase")
    plt.colorbar(im0, ax=axes[0], label="rad")

    im1 = axes[1].imshow(fitted, cmap="RdBu_r")
    axes[1].set_title("Fitted Parabolic Phase")
    plt.colorbar(im1, ax=axes[1], label="rad")

    im2 = axes[2].imshow(error, cmap="RdBu_r")
    axes[2].set_title("Phase Error (Aberrations)")
    plt.colorbar(im2, ax=axes[2], label="rad")

    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def save_focus_intensity(
    focus_data, output_dir: Path, filename: str = "focus_intensity.png"
):
    """
    Save focus intensity plot with FWHM ellipse.
    """
    if not focus_data:
        return

    focus_field = focus_data["focus_field"]
    focus_intensity = np.abs(focus_field) ** 2
    dx = focus_data["dx_focus"]
    dy = focus_data["dy_focus"]

    focus_pos = focus_data["focus_position"]
    focus_x_nm = focus_pos[0] * 1e9
    focus_y_nm = focus_pos[1] * 1e9
    fwhm_x = focus_data["focus_size"]["fwhm_x"] * 1e9
    fwhm_y = focus_data["focus_size"]["fwhm_y"] * 1e9

    ny, nx = focus_intensity.shape
    extent = (
        -nx / 2 * dx * 1e9,
        nx / 2 * dx * 1e9,
        -ny / 2 * dy * 1e9,
        ny / 2 * dy * 1e9,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(focus_intensity, cmap="hot", extent=extent, origin="lower")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_title("Focus Intensity Distribution")
    plt.colorbar(im, ax=ax, label="Intensity (a.u.)")

    # FWHM ellipse
    ellipse = Ellipse(
        xy=(focus_x_nm, focus_y_nm),
        width=fwhm_x,
        height=fwhm_y,
        fill=False,
        edgecolor="cyan",
        linewidth=2,
        linestyle="--",
        label=f"FWHM: {fwhm_x:.1f} x {fwhm_y:.1f} nm",
    )
    ax.add_patch(ellipse)
    ax.scatter(focus_x_nm, focus_y_nm, c="cyan", s=50, marker="+", linewidths=2)
    ax.legend(loc="upper left", fontsize=10, facecolor="black", labelcolor="white")

    ax.text(
        0.02,
        0.92,
        f"Center: ({focus_x_nm:.1f}, {focus_y_nm:.1f}) nm",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def save_zernike_bar(
    aberration_data, output_dir: Path, filename: str = "zernike_bar.png"
):
    """
    Save Zernike coefficients bar chart.
    """
    if not aberration_data or not aberration_data["zernike_results"]:
        return

    zernike = aberration_data["zernike_results"]
    aberr_analysis = zernike.get("aberration_analysis", {})

    if not aberr_analysis:
        return

    labels, values = [], []
    for key, aberr in sorted(aberr_analysis.items(), key=lambda x: int(x[0][1:])):
        labels.append(f"{key}: {aberr['name'][:15]}")
        values.append(aberr["rms_nm"])

    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = range(len(values))
    ax.barh(y_pos, values, color="steelblue", edgecolor="navy")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("RMS (nm)")
    ax.set_title("Zernike Aberration Coefficients")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # Z0 at top

    plt.tight_layout()
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def save_all_plots(wavefront_data, aberration_data, focus_data, output_dir: Path):
    """
    Save all standard plots to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Saving Plots...")
    print("=" * 60)

    save_phase_maps(wavefront_data, output_dir)
    save_focus_intensity(focus_data, output_dir)
    save_zernike_bar(aberration_data, output_dir)


def analyze(params: dict, output_dir: Path = None, verbose: bool = True):
    """
    Complete analysis workflow: run pipeline, print results, save plots.

    Parameters
    ----------
    params : dict
        Configuration parameters
    output_dir : Path, optional
        Directory for saving plots (default: output/)
    verbose : bool
        Whether to print output
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"

    print("=" * 60)
    print("XGI Wavefront Sensor Analysis")
    print("=" * 60)
    print(f"Image: {params['image_path']}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Run pipeline
    wavefront_data, aberration_data, focus_data = run_pipeline(params, verbose=verbose)

    # Print summary
    print_results_summary(wavefront_data, aberration_data, focus_data)

    # Save plots
    save_all_plots(wavefront_data, aberration_data, focus_data, output_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return wavefront_data, aberration_data, focus_data


# =============================================================================
# Demo: Run with experimental data when executed directly
# =============================================================================
if __name__ == "__main__":
    from params import get_exp_params

    params = get_exp_params()
    output_dir = Path(__file__).parent.parent / "output" / "direct"

    analyze(params, output_dir=output_dir)
