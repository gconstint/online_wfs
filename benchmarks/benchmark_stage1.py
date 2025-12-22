"""
Benchmark for load_and_preprocess_image function.

This script measures and reports the execution time of each step
in the load_and_preprocess_image function, helping identify optimization
opportunities.

Usage:
    python benchmark_stage1.py [--runs N]
"""

import sys
from pathlib import Path
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.fft import fft2, fftshift
from os import cpu_count

from core import (
    load_images,
    image_correction,
    center_crop,
    calculate_harmonic_periods,
    accurate_harmonic_periods,
    calculate_rotation_angle_from_peaks,
    calculate_wavelength,
)


DEFAULT_CROP_SIZE = 2048


def get_default_params() -> dict:
    """Get default pipeline parameters."""
    params = dict()

    params["pixel_size"] = [0.715e-6, 0.715e-6]
    params["wavelength"] = calculate_wavelength(7100)
    params["det2sample"] = 0.35
    params["total_dist"] = 6.5
    params["source_dist"] = params["total_dist"] - params["det2sample"]

    period = 18.38e-6
    params["grating_period"] = period / 2
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    img_path = str(Path(__file__).parent.parent / "data" / "sample_exp.tif")
    params.update(
        {
            "image_path": img_path,
            "dark_image_path": None,
            "flat_image_path": None,
        }
    )

    return params


def benchmark_load_and_preprocess(params: dict, n_runs: int = 10) -> dict:
    """
    Benchmark each step in load_and_preprocess_image.
    """
    print("\n" + "=" * 70)
    print("LOAD_AND_PREPROCESS_IMAGE BENCHMARK")
    print("=" * 70)

    # Define timing categories
    steps = [
        "load_images",
        "image_correction",
        "center_crop",
        "harmonic_periods",
        "asarray_float32",
        "fft2",
        "fftshift",
        "total_fft",
    ]

    timings = {step: [] for step in steps}

    descriptions = {
        "load_images": "Load TIFF files from disk",
        "image_correction": "Dark/flat field correction",
        "center_crop": "Center crop to 2048x2048",
        "harmonic_periods": "Calculate harmonic periods",
        "asarray_float32": "Convert to float32 C-contiguous",
        "fft2": "FFT2 computation",
        "fftshift": "FFT shift operation",
        "total_fft": "asarray + fft2 + fftshift combined",
    }

    print(f"\nRunning {n_runs} iterations...")

    for run in range(n_runs):
        # Step 1: Load images
        start = time.perf_counter()
        img, dark, flat = load_images(
            params["image_path"], params["dark_image_path"], params["flat_image_path"]
        )
        timings["load_images"].append(time.perf_counter() - start)

        # Step 2: Image correction
        start = time.perf_counter()
        img = image_correction(img, flat=flat, dark=dark, epsilon=1e-8, normalize=False)
        timings["image_correction"].append(time.perf_counter() - start)

        # Step 3: Center crop
        start = time.perf_counter()
        img_cropped = center_crop(img, target_size=DEFAULT_CROP_SIZE)
        timings["center_crop"].append(time.perf_counter() - start)

        # Step 4: Calculate harmonic periods
        start = time.perf_counter()
        harmonic_periods = calculate_harmonic_periods(
            (img_cropped.shape[0], img_cropped.shape[1]),
            params["pixel_size"],
            params["pattern_period"],
        )
        timings["harmonic_periods"].append(time.perf_counter() - start)

        # Step 5: Convert to float32
        start = time.perf_counter()
        img32 = np.asarray(img_cropped, dtype=np.float32, order="C")
        timings["asarray_float32"].append(time.perf_counter() - start)

        # Step 6: FFT2
        start = time.perf_counter()
        img_fft_raw = fft2(img32, norm="ortho", workers=cpu_count())
        timings["fft2"].append(time.perf_counter() - start)

        # Step 7: FFT shift
        start = time.perf_counter()
        img_fft = fftshift(img_fft_raw)
        timings["fftshift"].append(time.perf_counter() - start)

        # Total FFT time
        timings["total_fft"].append(
            timings["asarray_float32"][-1]
            + timings["fft2"][-1]
            + timings["fftshift"][-1]
        )

        if (run + 1) % 5 == 0 or run == 0:
            print(f"  Run {run + 1}/{n_runs} completed")

    # Calculate statistics and print results
    print("\n" + "-" * 80)
    print(
        f"{'Step':<20} {'Description':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'%':>8}"
    )
    print("-" * 80)

    # Calculate total time (excluding total_fft which is a sum)
    total_mean = (
        sum(np.mean(timings[s]) for s in steps if s != "total_fft") * 1000
        - np.mean(timings["total_fft"]) * 1000
        + np.mean(timings["total_fft"]) * 1000
    )

    # Actually, total is simply sum of non-redundant steps
    non_redundant = [
        "load_images",
        "image_correction",
        "center_crop",
        "harmonic_periods",
        "total_fft",
    ]
    total_mean = sum(np.mean(timings[s]) for s in non_redundant) * 1000

    results = {}
    for step in steps:
        times_ms = np.array(timings[step]) * 1000
        mean_ms = times_ms.mean()
        std_ms = times_ms.std()

        desc = descriptions.get(step, "")

        # Calculate percentage only for non-redundant steps
        if step in non_redundant:
            pct = mean_ms / total_mean * 100
            print(
                f"{step:<20} {desc:<35} {mean_ms:>10.2f} {std_ms:>10.2f} {pct:>7.1f}%"
            )
        else:
            print(f"  └─ {step:<16} {desc:<35} {mean_ms:>10.2f} {std_ms:>10.2f}")

        results[step] = {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "times_ms": times_ms.tolist(),
            "description": desc,
        }

    print("-" * 80)
    print(f"{'TOTAL':<20} {'':<35} {total_mean:>10.2f}")
    print(f"{'Frequency':<20} {'':<35} {1000 / total_mean:>10.1f} Hz")
    print("-" * 80)

    return results


def benchmark_fft_optimizations(n_runs: int = 10) -> dict:
    """
    Benchmark different FFT optimization strategies.
    """
    print("\n" + "=" * 70)
    print("FFT OPTIMIZATION COMPARISON")
    print("=" * 70)

    # Create test data
    data = np.random.randn(2048, 2048).astype(np.float32)

    strategies = {}

    # Strategy 1: Current approach (fft2 + fftshift with ortho normalization)
    times = []
    for i in range(n_runs + 2):
        data_c = np.asarray(data, dtype=np.float32, order="C")
        start = time.perf_counter()
        result = fftshift(fft2(data_c, norm="ortho", workers=cpu_count()))
        times.append(time.perf_counter() - start)
    strategies["current (ortho)"] = np.mean(times[2:]) * 1000

    # Strategy 2: No normalization
    times = []
    for i in range(n_runs + 2):
        data_c = np.asarray(data, dtype=np.float32, order="C")
        start = time.perf_counter()
        result = fftshift(fft2(data_c, workers=cpu_count()))
        times.append(time.perf_counter() - start)
    strategies["no normalization"] = np.mean(times[2:]) * 1000

    # Strategy 3: Different thread counts
    for n_threads in [1, 2, 4, 8]:
        times = []
        for i in range(n_runs + 2):
            data_c = np.asarray(data, dtype=np.float32, order="C")
            start = time.perf_counter()
            result = fftshift(fft2(data_c, norm="ortho", workers=n_threads))
            times.append(time.perf_counter() - start)
        strategies[f"threads={n_threads}"] = np.mean(times[2:]) * 1000

    # Strategy 4: Pre-allocated complex input (avoid internal conversion)
    times = []
    for i in range(n_runs + 2):
        data_c = np.asarray(data, dtype=np.complex64, order="C")
        start = time.perf_counter()
        result = fftshift(fft2(data_c, norm="ortho", workers=cpu_count()))
        times.append(time.perf_counter() - start)
    strategies["complex64 input"] = np.mean(times[2:]) * 1000

    # Print results
    print(f"\n{'Strategy':<25} {'Time (ms)':>12} {'vs Current':>12}")
    print("-" * 50)

    current_time = strategies["current (ortho)"]
    for name, time_ms in strategies.items():
        speedup = current_time / time_ms
        speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "-"
        print(f"{name:<25} {time_ms:>12.2f} {speedup_str:>12}")

    print("-" * 50)

    return strategies


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Benchmark")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs (default: 10)"
    )
    parser.add_argument(
        "--skip-fft-opt",
        action="store_true",
        help="Skip FFT optimization comparison",
    )
    args = parser.parse_args()

    # Get parameters
    params = get_default_params()

    # Benchmark load_and_preprocess_image
    results = benchmark_load_and_preprocess(params, n_runs=args.runs)

    # Benchmark FFT optimizations
    if not args.skip_fft_opt:
        fft_results = benchmark_fft_optimizations(n_runs=args.runs)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
