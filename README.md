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
    *   Wavefront error analysis
    *   Beam position and size measurements
    *   Focus characterization

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
python example_exp.py
```

### Simulation Mode

To process simulated data, run the `example_sim.py` script:

```bash
python example_sim.py
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

These parameters can be adjusted in the `example_exp.py` and `example_sim.py` scripts.

## Project Structure

*   `core/`: Core analysis modules
    *   `__init__.py`: Makes Python treat the directory as a package
    *   `beam_analysis.py`: Functions for beam characterization
    *   `dpc_preprocess.py`: Functions for DPC preprocessing
    *   `grating_analysis.py`: Functions for grating pattern analysis
    *   `optical_physics.py`: Functions for physical calculations
    *   `phase_analysis.py`: Functions for phase reconstruction
    *   `phase_fit.py`: Functions for wavefront fitting
    *   `propagation.py`: Functions for beam propagation
    *   `utils.py`: Utility functions
    *   `wfs_pipeline.py`: Functions for WFS pipeline
*   `example_exp.py`: Example script for experimental mode
*   `example_sim.py`: Example script for simulation mode
*   `benchmark_pipeline.py`: Script for benchmarking the pipeline
*   `pipline.py`: Main processing pipeline
*   `requirements.txt`: Python dependencies
*   `README.md`: This file