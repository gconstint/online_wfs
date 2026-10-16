"""M2 detector pixel calibration from a slit spherical-wave reference.

Sampling uses (py, px) in meters. Positions/vectors use shifted FFT bins in
(row, column) order; fringe periods use detector pixels. Calibrate first, then
pass the calibrated pixel sizes to the separate reconstruction workflow.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.optimize import minimize_scalar

from ..config import normalize_config
from ..core.arrays import validate_image
from ..core.harmonic_peaks import (
    calculate_harmonic_periods,
    calculate_peak_index,
    refine_harmonic_peak,
)
from ..preprocessing import preprocess_image


@dataclass(frozen=True)
class PixelSizeCalibrationResult:
    """Independent X/Y pixel sizes, carrier diagnostics and optional arrays.

    Set include_arrays=True when calling calibrate_pixel_size to retain
    spectrum and corrected_image for external plotting. The spectrum is the
    shifted FFT of a demeaned, Hann-windowed image, not the reconstruction FFT.
    as_dict() always excludes arrays; format_report() returns text without I/O.
    """

    image_shape: tuple[int, int]
    parameters: dict
    optics: dict
    carrier_geometry: dict
    comparison: dict
    preprocessing: dict
    spectrum: np.ndarray | None = field(default=None, repr=False, compare=False)
    corrected_image: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def scale_factor_yx(self) -> tuple[float, float]:
        """Return measured/theoretical axis projections as (Y, X), without averaging."""
        axes = self.comparison["axis_comparison"]
        return tuple(axes[axis]["measured_over_theoretical"] for axis in ("y", "x"))

    @property
    def pixel_size_yx(self) -> tuple[float, float]:
        """Calibrated (py, px) in meters, with no averaging between axes."""
        return tuple(
            float(initial * scale)
            for initial, scale in zip(
                self.parameters["pixel_size"], self.scale_factor_yx
            )
        )

    @property
    def pixel_size_y_m(self) -> float:
        """Calibrated vertical pixel size in meters."""
        return self.pixel_size_yx[0]

    @property
    def pixel_size_x_m(self) -> float:
        """Calibrated horizontal pixel size in meters."""
        return self.pixel_size_yx[1]

    def calibrated_params(self) -> dict:
        """Return physical inputs with calibrated pixels for later reconstruction.

        Other geometry describes this slit exposure. Set the reconstruction's
        source_dist for its own reference curvature when changing experiments.
        No input parameter or image is modified and reconstruction is not run.
        """
        return {**deepcopy(self.parameters), "pixel_size": list(self.pixel_size_yx)}

    def as_dict(self) -> dict:
        """Return an independent JSON-compatible numerical report without arrays."""
        geometry = self.carrier_geometry
        report = {
            "method": "slit_spherical_wave_harmonic_pixel_calibration",
            "parameters": {
                **self.parameters,
                "pixel_size": list(self.parameters["pixel_size"]),
            },
            "optics": self.optics,
            "preprocessing": self.preprocessing,
            "image_shape": list(self.image_shape),
            "carrier_geometry": geometry,
            "spacing_y": geometry["axis_spacing_y_px"],
            "spacing_x": geometry["axis_spacing_x_px"],
            **self.comparison,
            "peak_displacement_yx_bins": {
                order: self.comparison["peak_comparison"][order][
                    "measured_minus_theoretical_yx_bins"
                ]
                for order in ("00", "01", "10")
            },
            "spacing_ratio_yx_measured_over_expected": list(self.scale_factor_yx),
            "calibration": {
                "spacing_method": "single_peak_axis_projection",
                "initial_pixel_size_m": list(self.parameters["pixel_size"]),
                "calibrated_pixel_size_m": list(self.pixel_size_yx),
                "pixel_size_y_m": self.pixel_size_y_m,
                "pixel_size_x_m": self.pixel_size_x_m,
                "scale_factor_yx": list(self.scale_factor_yx),
                "formula": "p_i = p_initial_i * q_measured_i / q_expected_i",
            },
            "calibrated_params": self.calibrated_params(),
            "ratio_interpretation": (
                "pixel scale factors use single-peak detector-axis projections "
                "with valid R, z, P_g and a spherical reference; no rotation "
                "deprojection is applied"
            ),
        }
        return deepcopy(report)

    def format_report(self) -> str:
        """Format the axis comparison and initial/calibrated X/Y pixel sizes."""
        return _format_report(self.as_dict())


def _format_report(report):
    lines = [
        "Axis   Theory [bins]  Projection [bins]  Measured/theory   Difference",
    ]
    for axis, values in report["axis_comparison"].items():
        lines.append(
            f" {axis.upper()}     {values['theoretical_spacing_bins']:11.6f}  "
            f"{values['measured_spacing_bins']:15.6f}  "
            f"{values['measured_over_theoretical']:15.9f}  "
            f"{values['relative_difference_fraction']:+.6%}"
        )
    lines.append("")
    for index, axis in enumerate(("y", "x")):
        initial = report["parameters"]["pixel_size"][index]
        calibrated = report["calibration"][f"pixel_size_{axis}_m"]
        lines.append(
            f"Pixel size {axis.upper()} [um]: {initial * 1e6:.9f} -> {calibrated * 1e6:.9f}"
        )
    return "\n".join(lines)


def calibrate_pixel_size(
    image, params, *, dark=None, flat=None, include_arrays=False
) -> PixelSizeCalibrationResult:
    """Calibrate X/Y pixels from known slit geometry without modifying inputs.

    image must be a real finite 2-D detector array. params accepts the same
    physical inputs as Pipeline: p_energy [eV], pixel_size [m, scalar or py/px],
    g_period [m, nominal], g_angle [degrees, default 45], source_dist [m] and
    det2sample [m]. Apply any detector crop before this call and crop dark/flat
    arrays identically. No circle mask is applied to this FFT measurement.

    Dark/flat corrections match preprocess_image. Peaks use a separable Hann
    window, bounded integer search and local continuous DFT refinement, with
    the DC coordinate fixed at shape//2. For P_d=(1+z/R)*P_g, use the single-peak
    detector-axis projections: py=P_d*abs(dy10)/Ny and px=P_d*abs(dx01)/Nx.
    No rotation deprojection or Euclidean carrier length enters calibration.
    For ideal rotated equal-period carriers, these estimates retain cos(theta).
    X and Y are returned independently.
    R must be the known slit-to-grating distance and both axes must receive
    the assumed spherical reference. This is a global pixel-scale calibration.
    Optional arrays are retained only when include_arrays=True. This function
    performs no file access, console output, plotting or reconstruction.
    """
    normalized = normalize_config(params)
    corrected = preprocess_image(image, dark=dark, flat=flat)
    geometry, spectrum = measure_carriers(corrected, normalized)
    measurement = {
        "image_shape": list(corrected.shape),
        "carrier_geometry": geometry,
        "spacing_y": geometry["axis_spacing_y_px"],
        "spacing_x": geometry["axis_spacing_x_px"],
    }
    physical = {
        key: normalized[key]
        for key in (
            "p_energy",
            "pixel_size",
            "g_period",
            "g_angle",
            "source_dist",
            "det2sample",
        )
    }
    return PixelSizeCalibrationResult(
        image_shape=corrected.shape,
        parameters=physical,
        optics={
            "wavelength_m": normalized["wavelength"],
            "magnification": normalized["total_dist"] / normalized["source_dist"],
            "projected_grating_period_m": normalized["grating_period"],
            "detector_pattern_period_m": normalized["pattern_period"],
        },
        carrier_geometry=geometry,
        comparison=compare_carriers(measurement, normalized),
        preprocessing={
            "dark_corrected": dark is not None,
            "flat_corrected": flat is not None,
            "image_rotation_applied": False,
            "fft_window": geometry["window"],
            "dc_origin": geometry["dc_mode"],
            "peak_refinement": geometry["refinement"],
        },
        spectrum=spectrum if include_arrays else None,
        corrected_image=corrected if include_arrays else None,
    )


def _refine_frequency(image, initial, integer):
    """Maximize a local continuous DFT without zero-padding the detector FFT.

    Alternate one-dimensional searches using projected rows/columns. This
    avoids the systematic fractional-bin bias of three-point interpolation
    while retaining a bounded neighborhood of the detected integer maximum.
    """
    coordinates = [(np.arange(size) - (size - 1) / 2) / size for size in image.shape]
    frequency = np.array(initial, dtype=float)
    for _ in range(6):
        previous = frequency.copy()
        for axis in (0, 1):
            other = 1 - axis
            phase = np.exp(-2j * np.pi * frequency[other] * coordinates[other])
            # einsum avoids copying the full real detector into a complex array.
            line = (
                np.einsum("ij,j->i", image, phase)
                if axis == 0
                else np.einsum("i,ij->j", phase, image)
            )
            scale = max(float(np.sum(np.abs(line))), np.finfo(float).tiny)
            line = line / scale

            def objective(value):
                amplitude = np.exp(-2j * np.pi * value * coordinates[axis]) @ line
                return -float(abs(amplitude) ** 2)

            solution = minimize_scalar(
                objective,
                bounds=(integer[axis] - 0.75, integer[axis] + 0.75),
                method="bounded",
                options={"xatol": 1e-9},
            )
            if not solution.success:
                raise ValueError("local DFT carrier refinement did not converge")
            frequency[axis] = solution.x
        if np.max(np.abs(frequency - previous)) < 1e-6:
            return frequency
    raise ValueError("local DFT carrier refinement did not converge in six sweeps")


def _locate_carrier(spectrum, expected, order):
    """Search each FFT axis using its own carrier spacing.

    A narrow detector crop has very different row/column FFT scales. Using
    the shorter spacing for both search axes can exclude a rotated carrier
    even when it is well resolved along the long axis.
    """
    dc = np.asarray(spectrum.shape) // 2
    theoretical = calculate_peak_index(
        int(order[0]), int(order[1]), *spectrum.shape, *expected
    )
    center = dc + np.rint(np.asarray(theoretical) - dc).astype(int)
    # Round outward so a short-axis fractional peak still has neighboring bins.
    radius = np.maximum(1, np.ceil(expected / 4)).astype(int)
    lower = np.maximum(0, center - radius)
    upper = np.minimum(spectrum.shape, center + radius + 1)
    y0, x0 = lower
    y1, x1 = upper
    patch = np.abs(spectrum[y0:y1, x0:x1])
    if not patch.size:
        raise ValueError(
            f"{order} carrier search is outside the detector FFT; check initial pixels and geometry"
        )
    if not np.all(np.isfinite(patch)) or not np.any(patch > 0):
        raise ValueError(f"{order} carrier search must contain finite, nonzero data")
    index = np.asarray(np.unravel_index(np.argmax(patch), patch.shape)) + lower
    return {
        "theoretical_peak": theoretical,
        "integer_peak": index.tolist(),
        "measured_peak": refine_harmonic_peak(spectrum, index),
        "search_radius": radius,
        "peak_at_search_boundary": bool(np.any(np.abs(index - center) >= radius)),
    }


def measure_carriers(image, params):
    """Measure 01/10 in native detector coordinates, without image rotation.

    A separable Hann window suppresses crop-edge leakage. DC is fixed at its
    known FFT coordinate, never fitted to a low-frequency illumination lobe.
    Short crops with >=2 expected cycles per axis are accepted. Search radii
    follow each axis independently. Peaks still need a local peak/background
    amplitude ratio >=5 and must remain away from the search boundary.
    """
    image = validate_image(image, "calibration image")
    if min(image.shape) < 8:
        raise ValueError("calibration image must be at least 8x8 pixels")
    expected = np.array(
        calculate_harmonic_periods(
            image.shape, params["pixel_size"], params["pattern_period"]
        )
    )
    if np.any(expected < 2):
        minimum_shape = np.ceil(
            2 * params["pattern_period"] / np.asarray(params["pixel_size"])
        ).astype(int)
        details = "; ".join(
            f"{axis}: {image.shape[index]} pixels, {expected[index]:.3f} cycles "
            f"(need >= {minimum_shape[index]} pixels)"
            for index, axis in enumerate(("Y", "X"))
            if expected[index] < 2
        )
        raise ValueError(
            "calibration requires at least 2 carrier cycles per ROI axis; "
            f"{details}. Increase the ROI extent along the indicated axes."
        )
    window = np.hanning(image.shape[0])[:, None] * np.hanning(image.shape[1])[None, :]
    mean = np.sum(image * window) / np.sum(window)
    weighted = (image - mean) * window
    if not np.any(weighted):
        raise ValueError("detector image contains no carrier modulation")
    spectrum = fftshift(fft2(weighted))
    dc = np.asarray(image.shape) // 2
    peaks = {"00": dc.astype(float).tolist()}
    theoretical = {"00": peaks["00"]}
    diagnostics = {}
    spectral_floor = float(np.max(np.abs(spectrum))) * 1e-12
    for order in ("01", "10"):
        item = _locate_carrier(spectrum, expected, order)
        if item["peak_at_search_boundary"]:
            raise ValueError(
                f"{order} peak is at the search boundary; check initial pixel size, "
                "geometry, grating orientation and ROI"
            )
        integer = np.asarray(item["integer_peak"])
        center = dc + np.rint(np.asarray(item["theoretical_peak"]) - dc).astype(int)
        radius = item["search_radius"]
        y0, x0 = np.maximum(0, center - radius)
        y1, x1 = np.minimum(spectrum.shape, center + radius + 1)
        patch = np.abs(spectrum[y0:y1, x0:x1])
        yy, xx = np.indices(patch.shape)
        outside_peak = (np.abs(yy + y0 - integer[0]) > 1) | (
            np.abs(xx + x0 - integer[1]) > 1
        )
        # A 3x3 search can contain only the peak lobe; avoid an empty median.
        background = float(
            np.median(patch[outside_peak])
            if np.any(outside_peak)
            else np.median(np.abs(spectrum))
        )
        amplitude = float(abs(spectrum[tuple(integer)]))
        noise_floor = max(background, spectral_floor)
        peak_ratio = amplitude / noise_floor
        if peak_ratio < 5:
            raise ValueError(
                f"{order} carrier peak/background is {peak_ratio:.2f}; requires >=5"
            )
        frequency = _refine_frequency(
            weighted, np.asarray(item["measured_peak"]) - dc, integer - dc
        )
        if np.any(np.abs(frequency) / image.shape >= 0.5):
            raise ValueError(f"{order} carrier reaches the detector Nyquist limit")
        peaks[order] = (frequency + dc).tolist()
        theoretical[order] = item["theoretical_peak"]
        diagnostics[order] = {
            "integer_peak_px": integer.tolist(),
            "search_radius_yx_bins": radius.tolist(),
            "peak_to_background_amplitude": peak_ratio,
        }
    vector01 = np.asarray(peaks["01"]) - dc
    vector10 = np.asarray(peaks["10"]) - dc
    vectors = np.array([vector01, vector10])
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0) or abs(np.linalg.det(vectors)) < 1e-6 * np.prod(norms):
        raise ValueError("measured carriers must be distinct and non-collinear")
    geometry = {
        "peaks_px": peaks,
        "theoretical_peaks_px": theoretical,
        "expected_spacing_yx_bins": expected.tolist(),
        "vector_01_px": vector01.tolist(),
        "vector_10_px": vector10.tolist(),
        "axis_spacing_y_px": float(abs(vector10[0])),
        "axis_spacing_x_px": float(abs(vector01[1])),
        "spacing_method": "single_peak_axis_projection",
        "diagnostics": diagnostics,
        "window": "separable_hann",
        "refinement": "local_continuous_dft",
        "dc_mode": "exact_fft_origin",
    }
    return geometry, spectrum


def compare_carriers(measurement, params):
    """Describe peak shifts and scale ratios without changing physical inputs.

    Rotation and physical-angle diagnostics use the initial detector sampling.
    Calibration uses abs(dy10) for Y and abs(dx01) for X. Carrier rotation and
    physical vector lengths remain diagnostics, without rotation deprojection.
    """
    geometry = measurement["carrier_geometry"]
    expected = np.asarray(geometry["expected_spacing_yx_bins"])
    measured = np.abs([geometry["vector_10_px"][0], geometry["vector_01_px"][1]])
    shape = np.asarray(measurement["image_shape"])
    dc = shape // 2
    peaks = {}
    for order in ("00", "01", "10"):
        theoretical = np.asarray(geometry["theoretical_peaks_px"][order])
        actual = np.asarray(geometry["peaks_px"][order])
        integer = (
            dc
            if order == "00"
            else np.asarray(geometry["diagnostics"][order]["integer_peak_px"])
        )
        peaks[order] = {
            "theoretical_peak_yx_bins": theoretical.tolist(),
            "integer_peak_yx_bins": integer.tolist(),
            "measured_peak_yx_bins": actual.tolist(),
            "measured_minus_theoretical_yx_bins": (actual - theoretical).tolist(),
            "refinement_from_integer_yx_bins": (actual - integer).tolist(),
            "measured_carrier_yx_bins": (actual - dc).tolist(),
            "measured_frequency_yx_cycles_per_pixel": ((actual - dc) / shape).tolist(),
        }
    vectors = np.array([geometry["vector_01_px"], geometry["vector_10_px"]])
    physical = vectors / (shape * params["pixel_size"])
    rotations = np.array(
        [
            np.arctan2(physical[0, 0], physical[0, 1]),
            np.arctan2(-physical[1, 1], physical[1, 0]),
        ]
    )
    mean_angle = np.arctan2(np.sin(rotations).sum(), np.cos(rotations).sum())
    lengths = np.linalg.norm(physical, axis=1)
    cosine = np.clip(np.dot(*physical) / np.prod(lengths), -1, 1)
    return {
        "peak_comparison": peaks,
        "axis_comparison": {
            axis: {
                "theoretical_spacing_bins": float(expected[index]),
                "measured_spacing_bins": float(measured[index]),
                "measured_over_theoretical": float(measured[index] / expected[index]),
                "relative_difference_fraction": float(
                    measured[index] / expected[index] - 1
                ),
                "theoretical_fringe_period_detector_pixels": float(
                    shape[index] / expected[index]
                ),
                "measured_fringe_period_detector_pixels": float(
                    shape[index] / measured[index]
                ),
            }
            for index, axis in enumerate(("y", "x"))
        },
        "geometry_under_initial_pixel_sizes": {
            "carrier_rotation_01_deg": float(np.degrees(rotations[0])),
            "carrier_rotation_10_deg": float(np.degrees(rotations[1])),
            "mean_carrier_rotation_deg": float(np.degrees(mean_angle)),
            "angle_between_carriers_deg": float(np.degrees(np.arccos(cosine))),
            "carrier_periods_01_10_m": (1 / lengths).tolist(),
            "interpretation": "uses assumed (py, px); carrier orientation is distinct from nominal g_angle",
        },
    }
