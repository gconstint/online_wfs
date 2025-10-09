# X-ray Grating Interferometry (XGI) Wavefront Sensor

A comprehensive data analysis pipeline for X-ray beam characterization using grating interferometry techniques.

## Overview

This project implements an analysis pipeline for X-ray Grating Interferometry (XGI) wavefront sensing, supporting both experimental and simulated data processing. The system provides detailed beam characterization, phase reconstruction, and focus analysis capabilities.

## Features

- Complete wavefront sensing pipeline
- Support for both experimental and simulation modes
- Multi-threaded data processing
- Real-time analysis capabilities
- Comprehensive beam characterization
- Advanced phase reconstruction
- Focus quality assessment

## Key Capabilities

- Image preprocessing and normalization
- Grating interference pattern analysis
- Wavefront reconstruction
- Phase map generation
- Error profile analysis
- Beam position and size measurements
- Focus characterization

## Technical Details

### System Requirements

- Python 3.7+
- NumPy
- SciPy
- Threading support
- TIFF image processing capabilities

### Core Components

1. **Data Processing Pipeline**
   - Raw image preprocessing
   - Dark field and flat field corrections
   - Center-crop standardization

2. **Frequency Domain Analysis**
   - FFT-based processing
   - Harmonic period calculation
   - Pattern alignment

3. **Phase Analysis**
   - DPC (Differential Phase Contrast) calculation
   - Wavefront reconstruction
   - Error profile generation

4. **Focus Analysis**
   - Beam position tracking
   - Size measurements
   - Focus quality assessment

## Usage

### Experimental Mode

```python
from example_exp import main as exp_main

# Run experimental analysis
exp_main()
```

### Simulation Mode

```python
from example_sim import main as sim_main

# Run simulation analysis
sim_main()
```

### Configuration Parameters

Key system parameters that can be configured:

```python
params = {
    'pixel_size': [0.715e-6, 0.715e-6],  # Detector pixel size (m)
    'wavelength': wavelength,             # X-ray wavelength
    'det2sample': 0.35,                  # Grating-to-detector distance (m)
    'total_dist': 6.5,                   # Source-to-detector distance (m)
    'grating_period': period,            # Grating period (m)
}
```

## Output Products

1. **Phase Maps**
   - Reconstructed phase distribution
   - Wavefront error analysis

2. **Beam Characteristics**
   - Position measurements
   - Size calculations

3. **Analysis Visualizations**
   - Error profiles
   - Focus analysis plots

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/online_wfs.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
online_wfs/
├── core/                     # Core analysis modules
│   ├── beam_analysis.py     # Beam characterization
│   ├── dpc_preprocess.py    # DPC preprocessing
│   ├── grating_analysis.py  # Grating pattern analysis
│   ├── optical_physics.py   # Physical calculations
│   ├── phase_analysis.py    # Phase reconstruction
│   ├── phase_fit.py        # Wavefront fitting
│   ├── propagation.py      # Beam propagation
│   └── utils.py            # Utility functions
├── example_exp.py          # Experimental mode
├── example_sim.py          # Simulation mode
└── pipeline.py             # Main processing pipeline
```
