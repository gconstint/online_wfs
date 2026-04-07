# online_wfs

X-ray Grating Interferometry (XGI) wavefront sensing pipeline.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick Start

All examples load parameters from a JSON config file:

```bash
# Run the full analysis pipeline
python examples/example_pipeline.py

# Calculate Talbot distances
python examples/example_talbot_distance.py

# Calculate grating contrast
python examples/example_contrast.py
```

## Configuration

Edit `examples/params.json` before running:

```json
{
    "p_energy": 9000,
    "pixel_size": 0.48e-6,
    "det2sample": 0.65,
    "source_dist": 100.0,
    "g_period": 18.38e-6,
    "g_angle": 45,
    "image_path": "data/sample.tif",
    "dark_image_path": null,
    "flat_image_path": null,
    "rotation_angle": null,
    "lowpass_cutoff": 0.35,
    "parallel": true,
    "verbose": false,
    "show_plots": false
}
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `p_energy` | Photon energy (eV) |
| `pixel_size` | Detector pixel size (m) |
| `det2sample` | Detector-to-sample distance (m) |
| `source_dist` | Source-to-grating distance (m) |
| `g_period` | Grating period (m) |
| `g_angle` | Grating rotation angle (degrees) |
| `image_path` | Path to input image |

## Project Structure

```
online_wfs/
├── pipeline.py        # Main analysis pipeline
├── config.py          # Configuration loader
├── core/              # Analysis modules
│   ├── phase_analysis.py
│   ├── grating_analysis.py
│   ├── zernike_analysis.py
│   ├── propagation.py
│   └── ...
└── func/              # Utility functions
    ├── calculate_contrast.py
    └── calculate_talbot_distance.py
examples/
├── params.json            # Config file
├── example_pipeline.py
├── example_contrast.py
└── example_talbot_distance.py
```

## Dependencies

- numpy, scipy
- matplotlib
- Pillow, opencv-python
- scikit-image
