import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable image display for accurate time measurement
# Must be called before importing pyplot
# import matplotlib

# matplotlib.use("Agg")

import timeit
import platform
import numpy as np
import scipy
from pipeline import task
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

    print("=" * 50)
    print("PLATFORM INFORMATION")
    print("=" * 50)
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
    print("=" * 50)
    print()


def run_pipeline_task():
    """Wrapper function to run the pipeline task for timeit."""

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

    # The task is a generator, so we need to iterate through it to execute it
    # Use verbose=False and show_plots=False for accurate benchmarking
    for _ in task(params, verbose=False, show_plots=False):
        pass


if __name__ == "__main__":
    # Print platform information
    print_platform_info()

    # Number of times to run the test
    number_of_runs = 100

    # Collect individual run times
    times = []
    for i in range(number_of_runs):
        execution_time = timeit.timeit(run_pipeline_task, number=1)
        times.append(execution_time)
        print(f"Run {i + 1}/{number_of_runs}: {execution_time:.4f} seconds")

    # Calculate statistics
    times = np.array(times)
    print("\n" + "=" * 40)
    print("Benchmark Results (SGI Reconstruction)")
    print("=" * 40)
    print(f"Number of runs: {number_of_runs}")
    print(f"Mean time:      {times.mean():.4f} seconds")
    print(f"Std deviation:  {times.std():.4f} seconds")
    print(f"Min time:       {times.min():.4f} seconds")
    print(f"Max time:       {times.max():.4f} seconds")
