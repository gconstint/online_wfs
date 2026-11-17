"""FFT peak localization and crop geometry, independent of reconstruction/plots."""

from typing import Dict, List, Tuple

import numpy as np


# Used by reconstruction, contrast and pixel calibration.


def calculate_harmonic_periods(
    img_shape: Tuple[int, int], pixel_size: Tuple[float, float], pattern_period: float
) -> List[float]:
    """Return [vertical, horizontal] FFT carrier spacings in bins.

    Detector sampling ``pixel_size`` and real-space ``pattern_period`` use
    meters. The historical function name is retained for compatibility.
    """
    vert_harm = (pixel_size[0] * img_shape[0]) / pattern_period
    hor_harm = (pixel_size[1] * img_shape[1]) / pattern_period
    return [vert_harm, hor_harm]


def calculate_peak_index(
    har_v: int,
    har_h: int,
    n_rows: int,
    n_columns: int,
    period_vert: float,
    period_hor: float,
) -> List[float]:
    """Return a theoretical harmonic peak as [row, column] in shifted FFT bins.

    ``period_vert`` and ``period_hor`` are DC-to-first-order FFT spacings,
    rather than real-space fringe periods.
    """
    return [n_rows // 2 + har_v * period_vert, n_columns // 2 + har_h * period_hor]


def harmonic_window_bounds(shape, peak, harmonic_period):
    """Return [y0, y1, x0, x1] for a complete, even-sized FFT window."""
    periods = _validate_harmonic_periods(harmonic_period)
    center = _nearest_fft_bin(shape, peak)
    half = np.floor(periods / 2).astype(int)
    bounds = [
        center[0] - half[0],
        center[0] + half[0],
        center[1] - half[1],
        center[1] + half[1],
    ]
    if bounds[0] < 0 or bounds[1] > shape[0] or bounds[2] < 0 or bounds[3] > shape[1]:
        raise ValueError("harmonic extraction window extends outside the FFT spectrum")
    return list(map(int, bounds))


def refine_harmonic_peak(img_fft, peak):
    """Estimate [row, column] with a 3-point log-parabola on each axis.

    The estimate stays within half a bin of the local integer maximum. Flat,
    isolated, boundary or non-concave peaks retain their integer coordinates.
    This local model is best suited to smooth, isolated spectral lobes.
    """
    row, col = map(int, peak)
    rows, columns = img_fft.shape
    if not (0 <= row < rows and 0 <= col < columns):
        raise ValueError("peak must be inside the FFT spectrum")

    def offset(values):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)) or values[1] <= 0:
            return 0.0
        values = values / values[1]
        if max(values[0], values[2]) <= 1e-6:
            return 0.0
        logarithm = np.log(np.maximum(values, np.finfo(float).tiny))
        denominator = logarithm[0] - 2 * logarithm[1] + logarithm[2]
        if denominator >= -1e-14:
            return 0.0
        vertex = 0.5 * (logarithm[0] - logarithm[2]) / denominator
        return float(np.clip(vertex, -0.5, 0.5))

    dy = offset(np.abs(img_fft[row - 1 : row + 2, col])) if 0 < row < rows - 1 else 0.0
    dx = (
        offset(np.abs(img_fft[row, col - 1 : col + 2]))
        if 0 < col < columns - 1
        else 0.0
    )
    return [row + dy, col + dx]


def _validate_harmonic_periods(periods):
    values = np.asarray(periods, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values) & (values >= 2)):
        raise ValueError("harmonic periods must contain two finite values >= 2 pixels")
    return values


def _nearest_fft_bin(shape, peak):
    """Round offsets from DC, so half-bin ties do not depend on ROI parity."""
    dc = np.asarray(shape, dtype=int) // 2
    return (dc + np.rint(np.asarray(peak, dtype=float) - dc).astype(int)).tolist()


# Currently unused by project workflows; retained for reference.
###########
HARMONIC_NAMES = ("00", "01", "10")
HARMONIC_PEAK_MODES = ("theoretical", "integer", "subpixel")
DEFAULT_HARMONIC_PEAK_MODE = "theoretical"


def validate_harmonic_peak_mode(mode):
    """Validate an explicit crop mode without changing the physical carrier."""
    if mode not in HARMONIC_PEAK_MODES:
        raise ValueError(
            "harmonic_peak_mode must be 'theoretical', 'integer' or 'subpixel'"
        )
    return mode


def _peak_search_radius(periods):
    # Keep the search away from DC and adjacent orders, including short periods.
    return max(1, int(min(periods) // 4))


def _find_integer_peak(spectrum, theoretical_peak, search_radius):
    """Search a bounded neighborhood of the nearest theoretical FFT bin."""
    rows, columns = spectrum.shape
    cy, cx = _nearest_fft_bin(spectrum.shape, theoretical_peak)
    y0, y1 = max(0, cy - search_radius), min(rows, cy + search_radius + 1)
    x0, x1 = max(0, cx - search_radius), min(columns, cx + search_radius + 1)
    magnitude = np.abs(spectrum[y0:y1, x0:x1])
    if magnitude.size == 0:
        raise ValueError("harmonic peak search window is empty")
    if not np.all(np.isfinite(magnitude)) or not np.any(magnitude > 0):
        raise ValueError(
            "harmonic peak search window must contain finite, nonzero data"
        )
    row, column = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    return [int(y0 + row), int(x0 + column)]


def get_harmonic_geometry(
    img_fft,
    harmonic_period,
    harmonic_ij="00",
    *,
    peak_mode=DEFAULT_HARMONIC_PEAK_MODE,
    search_region=None,
):
    """Plan a harmonic crop and carrier correction without extracting arrays.

    The default theoretical center retains the fixed absolute-phase baseline.
    Integer/subpixel modes move the window while retaining that same carrier.
    """
    spectrum = np.asarray(img_fft)
    if spectrum.ndim != 2:
        raise ValueError("img_fft must be a 2D array")
    if harmonic_ij not in {"00", "01", "10"}:
        raise ValueError("harmonic_ij must be '00', '01' or '10'")
    mode = validate_harmonic_peak_mode(peak_mode)
    periods = _validate_harmonic_periods(harmonic_period)
    if search_region is None:
        search_region = _peak_search_radius(periods)
    if (
        isinstance(search_region, (bool, np.bool_))
        or int(search_region) != search_region
        or search_region < 1
    ):
        raise ValueError("search_region must be a positive integer")
    # Validate periods and geometry before attempting the local search.
    theoretical = calculate_peak_index(
        int(harmonic_ij[0]), int(harmonic_ij[1]), *spectrum.shape, *periods
    )
    harmonic_window_bounds(spectrum.shape, theoretical, periods)
    radius = min(int(search_region), _peak_search_radius(periods))
    integer_peak = _find_integer_peak(spectrum, theoretical, radius)
    measured = refine_harmonic_peak(spectrum, integer_peak)
    selected = {
        "theoretical": theoretical,
        "integer": integer_peak,
        "subpixel": measured,
    }[mode]
    # DC is known exactly and must not acquire a spurious carrier.
    if harmonic_ij == "00":
        selected = theoretical
    bounds = harmonic_window_bounds(spectrum.shape, selected, periods)
    y0, y1, x0, x1 = bounds
    center = [(y0 + y1) // 2, (x0 + x1) // 2]
    delta = np.asarray(measured) - theoretical
    geometry = {
        "peak_mode": mode,
        "theoretical_peak": list(map(float, theoretical)),
        "integer_peak": integer_peak,
        "measured_peak": measured,
        "selected_peak": list(map(float, selected)),
        "crop_center": center,
        "window_bounds": bounds,
        "measured_minus_theoretical": delta.tolist(),
        "carrier_offset": (np.asarray(theoretical) - center).tolist(),
        "search_radius": radius,
        "peak_at_search_boundary": bool(
            np.any(
                np.abs(
                    np.asarray(integer_peak)
                    - _nearest_fft_bin(spectrum.shape, theoretical)
                )
                >= radius
            )
        ),
    }
    return geometry


def accurate_harmonic_periods(
    img_fft: np.ndarray,
    init_periods: List[float],
    *,
    subpixel: bool = True,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Measure harmonic spacing and [row, column] peaks for calibration.

    ``subpixel=True`` refines each local maximum with a bounded log-parabola.
    Measured spacing is a diagnostic/calibration result; it does not silently
    replace the physical carrier or reconstruction sampling.
    """
    periods = _validate_harmonic_periods(init_periods)
    spectrum = np.asarray(img_fft)
    if spectrum.ndim != 2:
        raise ValueError("img_fft must be a 2D array")
    radius = _peak_search_radius(periods)
    peaks = {}
    for name in HARMONIC_NAMES:
        theoretical = calculate_peak_index(
            int(name[0]), int(name[1]), *spectrum.shape, *periods
        )
        peak = _find_integer_peak(spectrum, theoretical, radius)
        peaks[name] = (
            refine_harmonic_peak(spectrum, peak) if subpixel else list(map(float, peak))
        )
    return [
        abs(peaks["10"][0] - peaks["00"][0]),
        abs(peaks["01"][1] - peaks["00"][1]),
    ], peaks


def calculate_rotation_angle_from_peaks(
    peak_positions: Dict[str, List[float]],
) -> float:
    """Calculate the grating rotation angle from the 00, 01, and 10 peaks."""
    peak_00 = peak_positions["00"]
    peak_01 = peak_positions["01"]
    peak_10 = peak_positions["10"]

    # Horizontal angle
    delta_y_h = peak_01[0] - peak_00[0]
    delta_x_h = peak_01[1] - peak_00[1]
    angle_h = np.arctan2(delta_y_h, delta_x_h)

    # Vertical angle
    delta_y_v = peak_10[0] - peak_00[0]
    delta_x_v = peak_10[1] - peak_00[1]
    angle_v = np.arctan2(delta_y_v, delta_x_v) - np.pi / 2.0

    # Circular mean avoids a branch discontinuity near +/-180 degrees.
    angle = np.arctan2(
        np.sin(angle_h) + np.sin(angle_v),
        np.cos(angle_h) + np.cos(angle_v),
    )

    return float(np.degrees(angle))
