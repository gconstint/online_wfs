"""Fourier fringe visibility for an aligned 45-degree checkerboard grating.

For nominal grating period P, the effective period is d=P/sqrt(2). Divergent
illumination gives detector pitch p=d*(R+z)/R, where R is focus-to-grating
distance and z is grating-to-detector distance. All input lengths are meters;
pixel sampling is scalar or (vertical, horizontal).

The directional estimates are Vx=2*abs(F01)/abs(F00) and Vy=2*abs(F10)/abs(F00).
They equal Michelson visibility for an isolated sinusoid with uniform mean
illumination. For general patterns these are first-harmonic/DC ratios, can
exceed one, and are not clipped. Off-bin carriers, envelopes and phase
variations can reduce the global FFT peak estimate; no window or background
correction is applied implicitly.

Use measure_contrast() with the nominal period.
Fresnel pitch scaling follows Seaberg et al. (2019), equation (3).
https://doi.org/10.1107/S1600577519005721
"""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real

import numpy as np
from scipy.fft import fft2, fftshift

from ..core.arrays import normalize_pixel_size, validate_image
from ..core.harmonic_peaks import calculate_harmonic_periods, calculate_peak_index
from ..defaults import cpu_workers
from .talbot import calculate_effective_grating_period


@dataclass(frozen=True)
class ContrastResult:
    """Directional visibility and geometry, retaining the FFT for optional plots.

    Peaks use shifted-FFT [row, column] indices. FFT spacings are frequency-bin
    separations; fringe periods are real-image pixel counts. The spectrum is
    omitted from the representation and report, and the input is not modified.
    """

    grating_period: float
    effective_grating_period: float
    pixel_size: tuple[float, float]
    source_grating_dist: float
    grating_detector_dist: float
    contrast_horizontal: float
    contrast_vertical: float
    peaks: dict[str, list[int]]
    spectrum: np.ndarray = field(repr=False, compare=False)

    @property
    def contrast(self) -> float:
        """Return the arithmetic mean of the two directional visibilities."""
        return (self.contrast_horizontal + self.contrast_vertical) / 2.0

    @property
    def magnification(self) -> float:
        """Return the geometric detector-to-grating magnification."""
        return 1.0 + self.grating_detector_dist / self.source_grating_dist

    @property
    def pattern_period(self) -> float:
        """Return the expected detector-plane fringe pitch [m]."""
        return self.effective_grating_period * self.magnification

    @property
    def horizontal_fft_spacing_px(self) -> int:
        """Return the horizontal DC-to-first-order spacing in FFT bins."""
        return abs(self.peaks["01"][1] - self.peaks["00"][1])

    @property
    def vertical_fft_spacing_px(self) -> int:
        """Return the vertical DC-to-first-order spacing in FFT bins."""
        return abs(self.peaks["10"][0] - self.peaks["00"][0])

    @property
    def horizontal_fringe_period_px(self) -> float:
        """Return the measured horizontal fringe period in image pixels."""
        return self.spectrum.shape[1] / self.horizontal_fft_spacing_px

    @property
    def vertical_fringe_period_px(self) -> float:
        """Return the measured vertical fringe period in image pixels."""
        return self.spectrum.shape[0] / self.vertical_fft_spacing_px

    def as_analysis(self) -> dict:
        """Return the legacy analysis dictionary accepted by tools.plot_contrast.

        Historical '*_period_pixels' keys mean FFT spacings, not image periods.
        The arrays and peak dictionary are shared with this result.
        """
        return {
            "contrast": self.contrast,
            "contrast_horizontal": self.contrast_horizontal,
            "contrast_vertical": self.contrast_vertical,
            "spectrum": self.spectrum,
            "peaks": self.peaks,
            "horizontal_period_pixels": self.horizontal_fft_spacing_px,
            "vertical_period_pixels": self.vertical_fft_spacing_px,
        }

    def format_report(self) -> str:
        """Return scalar inputs and measurements without printing or file I/O."""
        py, px = self.pixel_size
        rows = (
            ("Image shape (rows, columns)", str(self.spectrum.shape)),
            ("Pixel size (vertical, horizontal)", f"{py * 1e6:.6g}, {px * 1e6:.6g} um"),
            ("Nominal grating period", f"{self.grating_period * 1e6:.6f} um"),
            (
                "Effective grating period",
                f"{self.effective_grating_period * 1e6:.6f} um",
            ),
            ("Source-to-grating distance", f"{self.source_grating_dist:.6g} m"),
            ("Grating-to-detector distance", f"{self.grating_detector_dist:.6g} m"),
            ("Magnification", f"{self.magnification:.6f}"),
            ("Expected detector pitch", f"{self.pattern_period * 1e6:.6f} um"),
            (
                "FFT spacing (horizontal, vertical)",
                f"{self.horizontal_fft_spacing_px}, {self.vertical_fft_spacing_px} bins",
            ),
            (
                "Fringe period (horizontal, vertical)",
                f"{self.horizontal_fringe_period_px:.6f}, {self.vertical_fringe_period_px:.6f} image pixels",
            ),
            ("Horizontal fringe contrast", f"{self.contrast_horizontal:.6%}"),
            ("Vertical fringe contrast", f"{self.contrast_vertical:.6%}"),
            ("Mean fringe contrast", f"{self.contrast:.6%}"),
        )
        return "\n".join(f"{name:<35}: {value}" for name, value in rows)


def _require_positive_finite(value: float, name: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


def _search_radii(search_region) -> tuple[int, int]:
    """Accept a nonnegative integer or a (vertical, horizontal) radius pair."""
    if isinstance(search_region, Integral):
        values = (search_region, search_region)
    else:
        try:
            values = tuple(search_region)
        except TypeError as exc:
            raise ValueError(
                "search_region must be an integer or (vertical, horizontal) pair."
            ) from exc
    if len(values) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0
        for v in values
    ):
        raise ValueError("search_region must contain nonnegative integer radii.")
    return int(values[0]), int(values[1])


def find_peak_in_region(img_fft, theoretical_idx, search_region) -> list[int]:
    """Find a local FFT maximum using inclusive, clipped search radii.

    Only the small search crop is converted to magnitude; no full-image mask
    is allocated. A zero-valued crop retains the predicted bin. An out-of-frame
    center or a non-finite crop raises ValueError rather than wrapping indices.
    """
    spectrum = np.asarray(img_fft)
    if spectrum.ndim != 2 or min(spectrum.shape) == 0:
        raise ValueError("img_fft must be a nonempty 2D array.")
    predicted = np.asarray(theoretical_idx, dtype=float)
    if predicted.shape != (2,) or not np.all(np.isfinite(predicted)):
        raise ValueError("theoretical_idx must contain finite [row, column] indices.")
    dc = np.asarray(spectrum.shape) // 2
    # Round the carrier offset, so half-bin ties are independent of ROI parity.
    center = dc + np.rint(predicted - dc).astype(int)
    if np.any(center < 0) or np.any(center >= spectrum.shape):
        raise ValueError("theoretical peak must be inside the FFT spectrum.")
    ry, rx = _search_radii(search_region)
    y0, x0 = max(0, center[0] - ry), max(0, center[1] - rx)
    y1, x1 = (
        min(spectrum.shape[0], center[0] + ry + 1),
        min(spectrum.shape[1], center[1] + rx + 1),
    )
    magnitude = np.abs(spectrum[y0:y1, x0:x1])
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("peak search region contains non-finite values.")
    if not np.any(magnitude > 0):
        return center.tolist()
    row, column = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    return [int(y0 + row), int(x0 + column)]


def measure_contrast(
    img: np.ndarray,
    *,
    pixel_size: float | tuple[float, float],
    grating_period: float,
    source_grating_dist: float,
    grating_detector_dist: float,
    search_region: int | tuple[int, int] | None = None,
) -> ContrastResult:
    """Measure both first-harmonic visibilities from physical parameters.

    Pass a finite real 2D detector image, pixel size [m], nominal checkerboard
    period [m] BEFORE division by sqrt(2), focus-to-grating distance [m] and
    grating-to-detector distance [m]. Images must have their carrier axes
    aligned with the detector axes. Dark/flat corrections belong upstream.

    search_region is an inclusive FFT-bin radius or (vertical, horizontal)
    pair. Zero selects the predicted bin and None uses one quarter of each
    predicted FFT spacing. Radii must remain below half those spacings to
    isolate the harmonics. DC is fixed at the exact shifted FFT origin.

    Read result.contrast for the mean or use result.format_report() for a
    report. Invalid inputs and unresolved or aliased geometry raise ValueError.
    """
    effective_period = calculate_effective_grating_period(grating_period)
    image = validate_image(img)
    if np.asarray(pixel_size).dtype.kind not in "iuf":
        raise ValueError(
            "pixel_size must be a positive scalar or (vertical, horizontal) pair."
        )
    pixels = normalize_pixel_size(pixel_size)
    _require_positive_finite(source_grating_dist, "source_grating_dist")
    _require_positive_finite(grating_detector_dist, "grating_detector_dist")
    _require_positive_finite(effective_period, "effective_grating_period")
    magnification = 1.0 + grating_detector_dist / source_grating_dist
    pattern_period = effective_period * magnification
    _require_positive_finite(pattern_period, "pattern_period")
    if np.any(pattern_period <= 2 * np.asarray(pixels)):
        raise ValueError(
            "predicted fringe pitch must exceed two detector pixels (Nyquist limit)."
        )

    offsets = np.asarray(
        calculate_harmonic_periods(image.shape, pixels, pattern_period)
    )
    dc = np.asarray(image.shape) // 2
    if np.any(offsets < 1) or np.any(dc + np.rint(offsets) >= image.shape):
        raise ValueError(
            "first-order peaks must be resolved between DC and the FFT boundary; check image size and geometry."
        )
    radii = (
        tuple(np.floor(offsets / 4).astype(int))
        if search_region is None
        else _search_radii(search_region)
    )
    if np.any(2 * np.asarray(radii) >= offsets):
        raise ValueError(
            "search_region radii must be less than half the corresponding FFT spacing to separate DC and first orders."
        )

    spectrum = fftshift(fft2(image, norm="ortho", workers=cpu_workers()))
    dc_amplitude = float(abs(spectrum[tuple(dc)]))
    if not np.isfinite(dc_amplitude) or spectrum[tuple(dc)].real <= 0:
        raise ValueError(
            "image must have a finite positive mean intensity for contrast normalization."
        )
    peaks = {"00": dc.tolist()}
    for name, order in (("01", (0, 1)), ("10", (1, 0))):
        predicted = calculate_peak_index(*order, *image.shape, *offsets)
        peaks[name] = find_peak_in_region(spectrum, predicted, radii)
    horizontal = 2.0 * (float(abs(spectrum[tuple(peaks["01"])])) / dc_amplitude)
    vertical = 2.0 * (float(abs(spectrum[tuple(peaks["10"])])) / dc_amplitude)
    if not np.isfinite(horizontal + vertical):
        raise ValueError(
            "contrast is non-finite; check the image intensity scale and background."
        )
    return ContrastResult(
        grating_period=float(grating_period),
        effective_grating_period=float(effective_period),
        pixel_size=pixels,
        source_grating_dist=float(source_grating_dist),
        grating_detector_dist=float(grating_detector_dist),
        contrast_horizontal=horizontal,
        contrast_vertical=vertical,
        peaks=peaks,
        spectrum=spectrum,
    )
