# X-ray Grating Interferometry (XGI) Wavefront Sensor

A comprehensive data analysis pipeline for X-ray beam characterization using grating interferometry techniques.

## Overview

This project provides a complete and robust data analysis pipeline for X-ray Grating Interferometry (XGI) wavefront sensing. It is designed for both experimental and simulated data, offering detailed beam characterization, phase reconstruction, and focus analysis capabilities. The pipeline is implemented in Python and leverages multi-threading for efficient real-time analysis.

## How it Works

The pipeline follows a standard XGI analysis workflow:

1.  **Image Preprocessing:** Raw images are corrected for dark-field and flat-field variations and then cropped to a standardized size.
2.  **Frequency Domain Analysis:** The preprocessed image is transformed into the frequency domain using FFT. The harmonic peaks of the grating pattern are identified to determine the precise grating period and orientation.
3.  **Phase Reconstruction:** The differential phase contrast (DPC) in both the horizontal and vertical directions is calculated from the harmonic components. The phase is then reconstructed from the DPC signals using a Frankot-Chellappa algorithm.
4.  **Wavefront Analysis:** A model (e.g., a parabolic wavefront) is fitted to the reconstructed phase to quantify aberrations and other wavefront characteristics.
5.  **Focus Analysis:** The pipeline analyzes the beam's focus by propagating the reconstructed wavefront to the focal plane and characterizing the spot size and position.

## Features

*   **End-to-End Pipeline:** A complete solution for XGI wavefront sensing, from raw data to final analysis.
*   **Dual Mode:** Supports both experimental and simulated data.
*   **High Performance:** Utilizes multi-threading and optimized numerical algorithms for real-time processing.
*   **Comprehensive Analysis:** Provides a wide range of analysis capabilities, including:
    *   Image preprocessing and normalization
    *   Grating interference pattern analysis
    *   Wavefront reconstruction and phase map generation
    *   Wavefront error analysis and Zernike fitting
    *   Beam position and size measurements
    *   Focus calibration and characterization
    *   Mirror surface error analysis

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/online_wfs.git
    cd online_wfs
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The pipeline can be run in either experimental or simulation mode using the provided example scripts.

### Experimental Mode

To process experimental data, run the `example_exp.py` script:

```bash
python examples/example_exp.py
```

### Simulation Mode

To process simulated data, run the `example_sim.py` script:

```bash
python examples/example_sim.py
```

### Benchmarking

To run performance benchmarks:

```bash
python benchmarks/benchmark_pipeline.py
```

## Configuration

The pipeline is configured through a Python dictionary of parameters. The key parameters are:

```python
params = {
    'pixel_size': [0.715e-6, 0.715e-6],  # Detector pixel size (m)
    'wavelength': wavelength,             # X-ray wavelength (m)
    'det2sample': 0.35,                  # Grating-to-detector distance (m)
    'total_dist': 6.5,                   # Source-to-detector distance (m)
    'grating_period': period,            # Grating period (m)
}
```

These parameters can be adjusted in the example scripts located in the `examples/` directory.

## Project Structure

```
online_wfs/
├── core/                           # Core analysis modules
│   ├── __init__.py                 # Package initialization with public API
│   ├── beam_analysis.py            # Beam characterization and profiling
│   ├── dpc_preprocess.py           # DPC signal preprocessing
│   ├── focus_calibration.py        # Focus position calibration
│   ├── grating_analysis.py         # Grating pattern analysis
│   ├── mirror_surface_analysis.py  # Mirror surface error analysis
│   ├── phase_analysis.py           # Phase reconstruction algorithms
│   ├── phase_fit.py                # Wavefront fitting functions
│   ├── propagation.py              # Beam propagation calculations
│   ├── roi_utils.py                # ROI selection utilities
│   ├── source_distance.py          # Source distance estimation
│   ├── utils.py                    # General utility functions
│   └── zernike_analysis.py         # Zernike polynomial analysis
├── data/                           # Sample data files
│   ├── sample_exp.tif              # Sample experimental data
│   └── sample_sim.tif              # Sample simulation data
├── examples/                       # Example scripts
│   ├── example_exp.py              # Experimental mode example
│   └── example_sim.py              # Simulation mode example
├── benchmarks/                     # Performance benchmarking
│   └── benchmark_pipeline.py       # Pipeline benchmark script
├── docs/                           # Documentation
├── pipeline.py                     # Main processing pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing and optimization
- **matplotlib**: Visualization
- **Pillow**: Image I/O
- **opencv-python**: Image processing
- **scikit-image**: Phase unwrapping and image analysis
- **tqdm**: Progress bars

## License

This project is open source. See the LICENSE file for details.