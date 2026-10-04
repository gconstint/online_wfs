# online_wfs

Unattended absolute SGI wavefront analysis, aligned with `wfs_sgi` commit
`8b3d6e0`. Python 3.10 or newer is required.

## Run and view results

From the repository root:

```bash
python main.py
```

The default input is the measured TIFF and its physical configuration in
`online_wfs/data/`. The command prints the horizontal/vertical focus distances
and saves three files:

- `output/results.png`: detector image, horizontal/vertical DPC, phase error,
  residual phase and focus distances.
- `output/arrays.npz`: the four complete numerical maps.
- `output/result.json`: only the four array-file references/units and two focus
  distances. Each JSON field uses the corresponding array key in `arrays.npz`.

To also open a local figure window, run `python main.py --show`. Other input and
output paths can be supplied without interactive prompts:

```bash
python main.py --output-dir output/sample
python main.py --config /path/to/config.json --image /path/to/frame.tif --output-dir output/frame
```

Plotting uses the optional `plots` extra (Matplotlib). It is outside the online
processing package. Full data is saved unchanged; figure panels show the finite
measurement region with physical coordinates in micrometers.

## Minimal online API

Only five physical inputs are required. Photon energy is eV; all lengths are
meters. Grating angle defaults to 45 degrees.

```python
from online_wfs import Pipeline

processor = Pipeline({
    "p_energy": 12000,
    "pixel_size": 0.48e-6,  # Scalar or (py, px)
    "g_period": 30e-6,
    "source_dist": 0.5,    # Source to grating
    "det2sample": 0.65,    # Grating to detector
})
result = processor.process(image)
```

The returned dictionary contains exactly six fields:

| Field | Meaning | Unit |
|---|---|---|
| `dpc_x` | Horizontal carrier-demodulated wavefront gradient | rad/m |
| `dpc_y` | Vertical carrier-demodulated wavefront gradient | rad/m |
| `phase_error` | Phase integrated after subtracting the affine DPC background | rad |
| `residual_phase` | Phase error minus its Zernike fit | rad |
| `focus_distance_x_m` | Horizontal distance from grating to Gaussian waist | m |
| `focus_distance_y_m` | Vertical distance from grating to Gaussian waist | m |

Maps use the nominal grating-plane coordinates and `(row, column)` indexing.
Unsupported pixels are NaN. The theoretical carrier already includes the
spherical curvature implied by `source_dist`, so the demodulated DPC measures
deviations from that carrier. Before inferring focus, the processor restores
the full detector-plane curvature and combines it with the detector intensity
width through `1/q = 1/R - i λ/(π w²)`. It then propagates back to the grating:
`q_grating = q_detector - det2sample`. The two distances are `abs(Re(q_grating))`.
They follow the measured image rather than being fixed to `source_dist`.
Focus here is the waist position in the Gaussian-q model. For a non-Gaussian
beam, these distances are equivalent Gaussian estimates.

Reuse one processor for serial frames. Optional dark/flat arrays are copied at
setup: `Pipeline(params, dark=dark_image, flat=flat_image)`. For TIFF input use
`image_path`, `dark_image_path` and `flat_image_path` in JSON, then call
`Pipeline("config.json").process()`. JSON paths are relative to that file.
A fixed `rotation_angle` in degrees defaults to zero.

Thresholds, aperture selection and fitting choices are internal defaults.
The package uses absolute mode, without interactive operations, plotting,
checkpoint generators or timing fields. Detailed numerical reconstruction
helpers remain internal to the reconstruction and analysis modules rather than
the root API.

## Install and build

For a working Python/HTTPS environment:

```bash
python -m pip install '.[plots]'  # Online runtime plus main.py plotting
python -m pip install build
python -m build
```

The current workspace has all required build/runtime libraries installed but
its Conda SSL libraries mix x86_64 Python with arm64 OpenSSL. Reuse existing
local dependencies without downloading:

```bash
python -m pip install --no-index --no-build-isolation --no-deps .
python -m build --no-isolation
python -m pip install --no-index --no-deps dist/online_wfs-0.4.1-py3-none-any.whl
```

For editable installation, add `-e` before `.`. Matplotlib must already be
available when using `main.py`; the online runtime requires only NumPy, SciPy
and tifffile. Optional `rotation` installs OpenCV for a supplied fixed angle.

Wheel and source distributions include the measured TIFF/configuration.
The source archive also includes `main.py` and `tools/plot_results.py`.
Packaged data can be loaded from any working directory:

```python
from online_wfs.data import load_sample
image, params = load_sample()
```

## Examples

The four examples run from the repository root:

```bash
python -m examples.example_pipeline
python -m examples.example_slit_size
python -m examples.example_contrast
python -m examples.example_talbot_distance
```

The pipeline and contrast examples load the packaged measured frame and its
configuration. The Talbot example reads only the packaged configuration; the
slit example defines its inputs directly. All examples print their results.

## Repository layout and benchmark

```text
online_wfs/            # Installable online package
  core/                # FFT geometry, DPC, Poisson and aperture calculations
  analysis/            # Internal beam, focus and Zernike calculations
  physics/             # Contrast, slit and Talbot calculators
  data/                # Measured TIFF and matching physical configuration
main.py                # Save results and call the plot helper
tools/                 # Plotting source code, outside the online package
benchmarks/            # Timing code, using the packaged measured frame
examples/              # Pipeline, slit size, contrast and Talbot distance examples
docs/                  # Numerical alignment notes and historical flowcharts
output/                # Generated arrays, figures and JSON reports (Git-ignored)
  archive/             # Previously generated sample artifacts and reports
dist/                  # Built wheel and source distributions (Git-ignored)
```

```bash
python -m benchmarks.benchmark_online --output output/benchmark.json
```

Source code stays in the package, `tools/`, `benchmarks/` and `examples/`.
Runtime files default to `output/`; use `--output-dir` to select another result
directory. The plot helper is called by `main.py`, which also reproduces the
packaged-sample figure. Generated files are excluded from the source archive.

The benchmark reads the packaged TIFF once and repeats complete calls, comparing
reused/fresh processors. Its default is two warmups and five measured repetitions
per path; TIFF loading is outside the timed region. The online package contains
no clocks. Numerical defaults/provenance are recorded in `docs/alignment.md`.
There is no `tests/` folder or pytest dependency/configuration.
