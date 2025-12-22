"""
Detailed Pipeline Benchmark

This script measures and reports the execution time of each stage
in the XGI wavefront reconstruction pipeline, providing insights
into performance bottlenecks and optimization opportunities.

Usage:
    python benchmark_pipeline.py [--runs N] [--parallel] [--serial]
"""

import sys
from pathlib import Path
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import platform
import numpy as np
import scipy
from scipy.fft import fft2, fftshift
from os import cpu_count

from pipeline import (
    load_and_preprocess_image,
    extract_harmonics_and_dpc,
    apply_magnification_correction,
    reconstruct_phase,
    fit_wavefront,
    analyze_aberrations,
    analyze_beam_at_detector,
    analyze_focus_by_propagation,
    task,
)
from core import calculate_wavelength


def get_platform_info() -> dict:
    """Collect platform and system information for benchmark reporting."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }

    # Try to get CPU info (macOS specific)
    try:
        import subprocess

        # Get CPU brand on macOS
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info["cpu_brand"] = result.stdout.strip()

        # Get physical memory
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.strip())
            info["memory_gb"] = mem_bytes / (1024**3)
    except Exception:
        pass

    return info


def print_platform_info():
    """Print platform information in a formatted way."""
    info = get_platform_info()

    print("=" * 70)
    print("PLATFORM INFORMATION")
    print("=" * 70)
    print(f"Python:     {info.get('python_version', 'N/A')}")
    print(f"Platform:   {info.get('platform', 'N/A')}")
    if "cpu_brand" in info:
        print(f"CPU:        {info.get('cpu_brand')}")
    else:
        print(f"Processor:  {info.get('processor', 'N/A')}")
    if "memory_gb" in info:
        print(f"Memory:     {info.get('memory_gb'):.1f} GB")
    print(f"NumPy:      {info.get('numpy_version', 'N/A')}")
    print(f"SciPy:      {info.get('scipy_version', 'N/A')}")
    print("=" * 70)
    print()


def get_default_params() -> dict:
    """Get default pipeline parameters for benchmarking."""
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

    return params


def benchmark_stage_by_stage(params: dict, n_runs: int = 5) -> dict:
    """
    Benchmark each pipeline stage individually.

    Returns a dictionary with timing statistics for each stage.
    """
    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE BENCHMARK")
    print("=" * 70)

    timings = {
        "stage1_load_preprocess": [],
        "stage2_harmonic_dpc": [],
        "stage3_magnification": [],
        "stage4_phase_recon": [],
        "stage5_wavefront_fit": [],
        "stage6_aberration": [],
        "stage7_beam_detector": [],
        "stage8_focus_prop": [],
    }

    descriptions = {
        "stage1_load_preprocess": "Load images, correction, crop, FFT",
        "stage2_harmonic_dpc": "Extract harmonics, compute DPC signals",
        "stage3_magnification": "Apply magnification correction",
        "stage4_phase_recon": "Lowpass filter, DPC integration",
        "stage5_wavefront_fit": "Parabolic wavefront fitting",
        "stage6_aberration": "ROI selection, Zernike analysis",
        "stage7_beam_detector": "Beam position & size at detector",
        "stage8_focus_prop": "Fresnel propagation to focus",
    }

    for run in range(n_runs):
        # Stage 1: Load and preprocess
        start = time.perf_counter()
        img_fft = load_and_preprocess_image(params, verbose=False, do_rotation=False)
        timings["stage1_load_preprocess"].append(time.perf_counter() - start)

        # Stage 2: Harmonic extraction and DPC
        start = time.perf_counter()
        harmonic_result = extract_harmonics_and_dpc(
            img_fft, params.copy(), verbose=False
        )
        timings["stage2_harmonic_dpc"].append(time.perf_counter() - start)

        int00 = harmonic_result["int00"]
        dpc_x = harmonic_result["dpc_x"]
        dpc_y = harmonic_result["dpc_y"]
        virtual_pixel_size = harmonic_result["virtual_pixel_size"]
        run_params = harmonic_result["params"]

        # Stage 3: Magnification correction
        start = time.perf_counter()
        dpc_x_corrected, dpc_y_corrected, _ = apply_magnification_correction(
            dpc_x, dpc_y, run_params, verbose=False
        )
        timings["stage3_magnification"].append(time.perf_counter() - start)

        # Stage 4: Phase reconstruction
        start = time.perf_counter()
        phase = reconstruct_phase(
            dpc_x_corrected, dpc_y_corrected, virtual_pixel_size, verbose=False
        )
        timings["stage4_phase_recon"].append(time.perf_counter() - start)

        # Stage 5: Wavefront fitting
        start = time.perf_counter()
        fitted_phase, phase_error, fit_params = fit_wavefront(
            phase,
            virtual_pixel_size,
            run_params["wavelength"],
            run_params,
            verbose=False,
            show_plots=False,
        )
        timings["stage5_wavefront_fit"].append(time.perf_counter() - start)

        # Stage 6: Aberration analysis
        start = time.perf_counter()
        aberration_result = analyze_aberrations(
            fitted_phase=fitted_phase,
            phase_error=phase_error,
            fit_params=fit_params,
            params=run_params,
            virtual_pixel_size=virtual_pixel_size,
            interactive=False,
            verbose=False,
            show_plots=False,
        )
        timings["stage6_aberration"].append(time.perf_counter() - start)

        # Stage 7: Beam at detector
        start = time.perf_counter()
        beam_position, beam_size = analyze_beam_at_detector(
            int00, virtual_pixel_size, verbose=False, show_plots=False
        )
        timings["stage7_beam_detector"].append(time.perf_counter() - start)

        # Stage 8: Focus propagation
        start = time.perf_counter()
        focus_result = analyze_focus_by_propagation(
            int00=int00,
            phase=phase,
            virtual_pixel_size=virtual_pixel_size,
            wavelength=run_params["wavelength"],
            propagation_distance=run_params["total_dist"],
            beam_size=beam_size,
            verbose=False,
            show_plots=False,
        )
        timings["stage8_focus_prop"].append(time.perf_counter() - start)

        print(f"  Run {run + 1}/{n_runs} completed")

    # Calculate statistics and print results
    print("\n" + "-" * 70)
    print(f"{'Stage':<30} {'Description':<30} {'Mean (ms)':>10} {'Std (ms)':>10}")
    print("-" * 70)

    total_mean = 0
    results = {}

    for stage_key, times in timings.items():
        times_ms = np.array(times) * 1000
        mean_ms = times_ms.mean()
        std_ms = times_ms.std()
        total_mean += mean_ms

        stage_name = stage_key.replace("_", " ").title()
        desc = descriptions.get(stage_key, "")
        print(f"{stage_name:<30} {desc:<30} {mean_ms:>10.2f} {std_ms:>10.2f}")

        results[stage_key] = {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "times_ms": times_ms.tolist(),
            "description": desc,
        }

    print("-" * 70)
    print(f"{'TOTAL':<30} {'':<30} {total_mean:>10.2f}")
    print(f"{'Frequency':<30} {'':<30} {1000 / total_mean:>10.1f} Hz")
    print("-" * 70)

    return results


def benchmark_parallel_vs_serial(params: dict, n_runs: int = 5) -> dict:
    """
    Compare parallel vs serial execution modes.
    """
    print("\n" + "=" * 70)
    print("PARALLEL VS SERIAL COMPARISON")
    print("=" * 70)

    # Warmup
    for _ in task(params, verbose=False, show_plots=False, parallel=True):
        pass

    # Serial mode
    serial_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        for _ in task(params, verbose=False, show_plots=False, parallel=False):
            pass
        serial_times.append(time.perf_counter() - start)

    serial_mean = np.mean(serial_times) * 1000
    serial_std = np.std(serial_times) * 1000

    # Parallel mode
    parallel_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        for _ in task(params, verbose=False, show_plots=False, parallel=True):
            pass
        parallel_times.append(time.perf_counter() - start)

    parallel_mean = np.mean(parallel_times) * 1000
    parallel_std = np.std(parallel_times) * 1000

    print(f"\n{'Mode':<20} {'Mean (ms)':>12} {'Std (ms)':>12} {'Frequency':>12}")
    print("-" * 56)
    print(
        f"{'Serial':<20} {serial_mean:>12.2f} {serial_std:>12.2f} {1000 / serial_mean:>11.1f} Hz"
    )
    print(
        f"{'Parallel':<20} {parallel_mean:>12.2f} {parallel_std:>12.2f} {1000 / parallel_mean:>11.1f} Hz"
    )
    print("-" * 56)

    speedup = serial_mean / parallel_mean
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1:
        print(f"Time saved: {serial_mean - parallel_mean:.1f}ms per frame")
    else:
        print(
            f"Overhead: {parallel_mean - serial_mean:.1f}ms per frame (parallel slower)"
        )

    return {
        "serial": {"mean_ms": serial_mean, "std_ms": serial_std},
        "parallel": {"mean_ms": parallel_mean, "std_ms": parallel_std},
        "speedup": speedup,
    }


def benchmark_full_pipeline(params: dict, n_runs: int = 100) -> dict:
    """
    Benchmark the complete pipeline with many runs for statistical accuracy.
    """
    print("\n" + "=" * 70)
    print(f"FULL PIPELINE BENCHMARK ({n_runs} runs)")
    print("=" * 70)

    # Warmup
    for _ in task(params, verbose=False, show_plots=False, parallel=True):
        pass

    # Benchmark
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        for _ in task(params, verbose=False, show_plots=False, parallel=True):
            pass
        times.append(time.perf_counter() - start)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_runs}")

    times_ms = np.array(times) * 1000

    print("\n" + "-" * 50)
    print("Statistics:")
    print("-" * 50)
    print(f"  Mean:      {times_ms.mean():.2f} ms")
    print(f"  Std:       {times_ms.std():.2f} ms")
    print(f"  Min:       {times_ms.min():.2f} ms")
    print(f"  Max:       {times_ms.max():.2f} ms")
    print(f"  Median:    {np.median(times_ms):.2f} ms")
    print(f"  P95:       {np.percentile(times_ms, 95):.2f} ms")
    print(f"  P99:       {np.percentile(times_ms, 99):.2f} ms")
    print("-" * 50)
    print(f"  Frequency: {1000 / times_ms.mean():.1f} Hz")
    print("-" * 50)

    return {
        "mean_ms": times_ms.mean(),
        "std_ms": times_ms.std(),
        "min_ms": times_ms.min(),
        "max_ms": times_ms.max(),
        "median_ms": np.median(times_ms),
        "p95_ms": np.percentile(times_ms, 95),
        "p99_ms": np.percentile(times_ms, 99),
        "frequency_hz": 1000 / times_ms.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="XGI Pipeline Benchmark")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs for stage benchmark"
    )
    parser.add_argument(
        "--full-runs",
        type=int,
        default=100,
        help="Number of runs for full pipeline benchmark",
    )
    parser.add_argument(
        "--skip-full", action="store_true", help="Skip full pipeline benchmark"
    )
    parser.add_argument(
        "--skip-stages", action="store_true", help="Skip stage-by-stage benchmark"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip parallel vs serial comparison",
    )
    args = parser.parse_args()

    # Print platform info
    print_platform_info()

    # Get parameters
    params = get_default_params()

    results = {}

    # Stage-by-stage benchmark
    if not args.skip_stages:
        results["stages"] = benchmark_stage_by_stage(params, n_runs=args.runs)

    # Parallel vs serial comparison
    if not args.skip_comparison:
        results["comparison"] = benchmark_parallel_vs_serial(params, n_runs=args.runs)

    # Full pipeline benchmark
    if not args.skip_full:
        results["full"] = benchmark_full_pipeline(params, n_runs=args.full_runs)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
