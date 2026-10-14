# online_wfs

Array-only absolute SGI wavefront analysis. Python 3.10 or newer is required.

`online_wfs` performs numerical calculations from parameter mappings and arrays.
File loading, saving, plotting and command-line output belong to local `tools/`
helpers and `main.py`. Sample TIFF/JSON resources remain in `online_wfs/data/`
and ship with the package. `tools/` and `reconstructions/` are Git-ignored local
directories and are excluded from distributions.
Local `tests/` and `docs/` are also Git-ignored and excluded from distributions.

## Run the saved detector data

Both local datasets use the same `Pipeline`. From the repository root:

```bash
python -m tools.reconstruct reconstructions/ssrf_bl13u_figure6_20261013 --output-dir output_current
python -m tools.reconstruct reconstructions/xray_wfs_recon_sgi_single_shot_20261013 --output-dir output_current
```

Each command writes `arrays.npz`, `result.json` and `results.png` to that
dataset's `output_current/`. Add `--check` to verify these saved results.
Historical result directories and validation reports are archived outside the
repository; each case retains only its current output. The calculation now
uses detector-resolved exact-carrier demodulation, an eighth-order circular
super-Gaussian filter, the original circular support, open-boundary weighted CG
integration and blockwise 36-term Zernike fitting. Phase unwrapping uses
`skimage.restoration.unwrap_phase` in 2-D on measured support with a fixed seed.
The integer 2*pi branch is anchored at the center pixel, or the nearest
supported pixel when the center is excluded. Returned carrier phases are not
centered by their mean.

The local loader applies `crop_indexes: [y0, y1, x0, x1]` before calling the
calculation package and computes the carrier phase at the crop center relative
to the original detector center. It passes this numerical correction as
`carrier_phase_offset_rad: [offset_x, offset_y]`. Complex harmonics are multiplied
by `exp(-1j * harmonic_peak_sign * offset)` before taking their angles. Dark/flat
arrays may match the full frame or selected crop.
An optional `flat_valid_mask_path` verifies that every selected calibration
pixel is valid. Crop metadata and file paths stay outside `online_wfs`.

Generic numerical options in each JSON describe phase cleanup and beam fitting;
the calculation package contains no branches keyed to dataset names or figures.

## Reconstruction order

| Stage | Operation |
|---|---|
| Harmonic extraction | Compute the nominal detector period `P = g_period * fringe_factor * M`; multiply by exact fractional complex carriers and filter each sideband with `exp(-0.5 * (r / radius)**8)`. |
| Output sampling | Retain every detector pixel in the selected crop, with no harmonic-grid reduction or interpolation. Detector sampling is `(py, px)`; the phase/DPC coordinate grid uses `(py, px) / M`. |
| Phase unwrapping | Correct the crop's constant carrier phases, take the complex harmonic angles, intersect the circular aperture with the harmonic-amplitude support, and unwrap each direction with `skimage` in 2-D. Anchor the integer `2*pi` branch at the center or its nearest supported pixel. |
| DPC background | Convert to rad/m. Within 80% of the aperture radius, fit X rows to `Bx = sx*x + tx` and Y columns to `By = sy*y + ty`, and take the median slopes and intercepts. Subtract both slope and intercept. |
| Base phase error | Integrate `(dpc_x - Bx, dpc_y - By)` once with open-boundary, amplitude-weighted gradient least squares. Integration fixes the piston by setting the measured-support mean to zero. |
| Phase cleanup | Jointly fit `1, x, y, x**2, x*y, y**2` on the full measured circular support; subtract only `x, y, x*y`. Retain the fitted piston and pure `x**2/y**2` contributions. |
| Zernike residual | Fit 36 OSA/ANSI terms (`j = 0..35`) to the cleaned phase with radial Tukey weights (`alpha = 0.3`), then compute `residual_phase = phase_error - zernike_fit_phase`. |

Amplitude reliability is the minimum of the two harmonics, each normalized by
its maximum within the circle. Pixels below 1% reliability are excluded. Edge
weights are the minimum reliability of their two supported endpoints. The
background-fit mask also excludes a 16-pixel image-edge margin.

`phase_cleanup="none"` returns the base integrated phase in `phase_error`;
the default `"tilt_xy"` returns the cleaned phase. Zernike fitting always follows
the selected phase cleanup. The public API returns four maps and two focus
distances. Internal reconstruction retains only the three reconstructed maps,
DC intensity, two DPC slopes, aperture geometry and grating-plane sampling needed
by the remaining calculations. Harmonic amplitudes and wrapped phases are
released after unwrapping. DPC backgrounds use one-dimensional affine profiles;
Zernike fitting writes the residual in blocks without a full fitted-phase map.

When comparing phase stages, use unweighted RMS about the mean (`np.std`) on
the same measured support. RMS values depend on the frame and processing order.

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
| `phase_error` | Phase after affine DPC background subtraction and configured phase cleanup | rad |
| `residual_phase` | Phase error minus its Zernike fit | rad |
| `focus_distance_x_m` | Horizontal distance from grating to Gaussian waist | m |
| `focus_distance_y_m` | Vertical distance from grating to Gaussian waist | m |

Maps retain the input image dimensions and use nominal grating-plane coordinates
with `(row, column)` indexing. Pixel centers are
`(index - (size-1)/2) * pixel_size / M`.
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
setup: `Pipeline(params, dark=dark_image, flat=flat_image)`. A fixed
`rotation_angle` in degrees defaults to zero. `Pipeline` accepts a parameter mapping and requires an image
array for every `process(image)` call; it never opens a configuration or image
file. `normalize_config(params)` validates parameters and derives optical values
in memory.

Two optional algorithm choices are supported:

- `phase_cleanup="tilt_xy"` (default): jointly fit the six quadratic phase terms
  and remove only x, y and xy. Pure x²/y² and piston are retained. Set `"none"`
  to keep phase immediately after affine DPC-background subtraction and integration.
- `beam_width_method="gaussian_profile"` (default): fit the horizontal/vertical
  DC profiles with the half-maximum fallback. Set `"gaussian_envelope"` for the
  additive-background marginal-envelope fit, which reports unresolved widths
  as errors.

Both choices describe numerical operations and apply to arbitrary images.

For an already-cropped array, provide `carrier_phase_offset_rad` if its carrier
origin should remain tied to a larger detector frame. With zero rotation,
`dy = (y0+y1-1)/2 - (raw_height-1)/2` and
`dx = (x0+x1-1)/2 - (raw_width-1)/2`; the offsets are
`[2*pi*dx/period_x_px, 2*pi*dy/period_y_px]`. Add any independently calibrated
carrier offsets to these values. A full frame defaults to `[0, 0]`. The local
`load_inputs` helper computes the crop contribution automatically from the
original frame shape, including a supplied alignment rotation. Do not apply
that same crop contribution again after loading.

Local file workflows use `tools.io.load_inputs("config.json")` to load the
detector and calibration arrays. Its `image_path`, `dark_image_path`,
`flat_image_path` and `flat_valid_mask_path` fields resolve relative to the JSON
file and are removed before parameters reach the calculation package.
The loader rejects unsupported configuration fields.

Thresholds, circular-aperture geometry and Zernike settings are internal defaults.
The package uses absolute mode, without interactive operations, plotting,
checkpoint generators or timing fields. Detailed numerical reconstruction
helpers remain internal to the reconstruction and analysis modules rather than
the root API.

Plotting and saving can be composed independently:

```python
from online_wfs import Pipeline
from tools.io import load_inputs
from tools.plot_results import plot_results
from tools.save_results import save_result

image, params, calibration = load_inputs("config.json")
processor = Pipeline(params, **calibration)
result = processor.process(image)
save_result(result, "output/custom")
plot_results(image, result, processor.params, "output/custom/results.png")
```

## Install and build

For a working Python/HTTPS environment:

```bash
python -m pip install .  # Calculation package: NumPy, SciPy and scikit-image
python -m pip install '.[io,plots]'  # Local TIFF loading and plotting helpers
python -m pip install build
python -m build
```

To reuse already installed dependencies without downloading:

```bash
python -m pip install --no-index --no-build-isolation --no-deps .
python -m build --no-isolation
python -m pip install --no-index --no-deps dist/online_wfs-0.4.1-py3-none-any.whl
```

For editable installation, add `-e` before `.`. Matplotlib must already be
available when using `main.py`; the calculation runtime requires NumPy, SciPy
and scikit-image. The `io` extra installs tifffile for local TIFF loading. Optional
`rotation` installs OpenCV for a supplied fixed angle.

Wheel and source distributions include the measured TIFF/configuration.
The source archive excludes local execution helpers and reconstruction outputs.
With the local helpers available, packaged data can be loaded from any working
directory:

```python
from tools.io import load_sample
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
Contrast plotting is separate: pass the dictionary returned by
`online_wfs.physics.contrast.analyze_contrast(...)` to
`tools.plot_contrast.plot_contrast(analysis, "output/contrast.png")`.

## Repository layout

```text
online_wfs/            # Installable calculation package
  config.py            # In-memory parameter validation and derived optics
  preprocessing.py     # Array-only detector corrections
  pipeline.py          # Array input to six computed result fields
  core/                # Carrier demodulation, 2-D unwrapping and weighted integration
  analysis/            # Internal beam, focus and Zernike calculations
  physics/             # Contrast, slit and Talbot calculators
  data/                # Static measured TIFF and matching JSON, packaged unchanged
reconstructions/       # Local measured cases and outputs (Git-ignored)
  */data.npz           # One lossless frame/calibration bundle per case
  */params.json        # Saved input parameters per case
main.py                # Save results and call the plot helper
tools/                 # Local loading, plotting, saving and execution (Git-ignored)
examples/              # Pipeline, slit size, contrast and Talbot distance examples
tests/                 # Local numerical checks (Git-ignored)
docs/                  # Local method records and historical flowcharts (Git-ignored)
output/                # Generated arrays, figures and JSON reports (Git-ignored)
  archive/             # Previously generated sample artifacts and reports
dist/                  # Built wheel and source distributions (Git-ignored)
```

Runtime files default to `output/`; use `--output-dir` to select another result
directory. `main.py` composes the local input loader, calculation pipeline,
saving helper and plot helper. The calculation package contains no file loading,
file writing, plots, console output or timing code. Generated files and local
helpers are excluded from the source archive. Existing local `tools/` and
`reconstructions/` files remain on disk when excluded from Git tracking.
