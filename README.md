# X-ray Grating Interferometry (XGI) Wavefront Sensor

A comprehensive data analysis pipeline for X-ray beam characterization using grating interferometry techniques.

## Overview

This project provides a complete and robust data analysis pipeline for X-ray Grating Interferometry (XGI) wavefront sensing. It is designed for high-performance, real-time analysis of X-ray beam wavefronts, supporting both single-frame analysis and continuous streaming modes. The pipeline is implemented in Python with optimized numerical algorithms and multi-threading capabilities.

## Key Features

*   **Real-time Capabilities:**
    *   **Streaming Runner (`runner.py`):** A dedicated runner for continuous data acquisition and processing.
    *   **Three-Stage Architecture:** Decoupled Data Reading, Processing, and Saving for maximum throughput.
    *   **Async Saving:** Non-blocking I/O operations for saving complex result sets.
*   **Advanced Analysis:**
    *   **Wavefront Reconstruction:** DPC-based phase retrieval with Frankot-Chellappa integration.
    *   **Zernike Decomposition:** Full Zernike polynomial fitting (up to 36 terms) for aberration analysis.
    *   **Focus Characterization:** Numerical propagation to focus, spot size (FWHM) measurement, and intensity profiling.
    *   **Beam Profiling:** 2D Gaussian fitting and statistical beam analysis.
*   **Dual Mode:** Seamlessly switch between Experimental data and Simulation modes.

## Quick Start (Streaming Mode)

The core feature of this project is the streaming runner, which simulates a continuous data acquisition pipeline.

**Run the streaming pipeline:**

```bash
# Default mode (10Hz, infinite loop)
python runner.py

# High-speed mode (20Hz)
python runner.py --fps 20

# Run for a specific duration (e.g., 60 seconds)
python runner.py --duration 60
```

**Output Structure:**
Results are saved in `output/stream/frame_XXXXXX/`:

*   `phase.npy`: Reconstructed phase map
*   `focus_field.npy`: Complex field at the focal plane
*   `zernike_results.txt`: Zernike coefficients and RMS error
*   `params.txt`: Comprehensive parameter file (Wavefront, Calibration, ROI, Focus metadata)

## Project Structure

```
online_wfs/
├── runner.py                       # Main Entry Point: Streaming Pipeline Runner
├── pipeline.py                     # Core XGI Analysis Pipeline (8-stage workflow)
├── params.py                       # Configuration Management
├── core/                           # Analysis Kernels
│   ├── phase_analysis.py           # Phase reconstruction & DPC
│   ├── zernike_analysis.py         # Zernike fitting
│   ├── propagation.py              # Wavefront propagation
│   ├── grating_analysis.py         # Interferogram analysis
│   └── ...
├── data/                           # Sample Data
│   └── sample_exp.tif              # Default experimental sample
└── output/                         # Analysis Results directory
```

## Architecture

The system utilizes a balanced generic pipeline architecture:

1.  **Stage 1: Data Reader (Thread)**
    *   Simulates/Acquires frames at target FPS.
    *   Implements frame dropping policy to maintain real-time constraints.

2.  **Stage 2: Processor (Main)**
    *   Executes the 8-stage XGI pipeline.
    *   Performs FFTs, phase unwrapping, and fitting.

3.  **Stage 3: Saver (Thread)**
    *   Asynchronously writes results to disk.
    *   Handles data formatting (NPY/TXT).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/online_wfs.git
    cd online_wfs
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

-   **numpy, scipy**: Core numerical computing
-   **matplotlib**: Visualization (optional for headless runner)
-   **Pillow, opencv-python**: Image I/O and processing
-   **scikit-image**: Advanced image algorithms

## License

This project is open source. See the LICENSE file for details.