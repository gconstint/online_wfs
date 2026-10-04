"""Absolute SGI unwrapping, affine DPC fitting and phase integration."""

import numpy as np

from .config import normalize_config
from .core.arrays import centered_fft, circular_domain, validate_image
from .core.dpc_conversion import carrier_phase_to_wavefront_gradient
from .core.dpc_fit import fit_dpc_background
from .core.harmonics import extract_harmonics
from .core.poisson import SupportedPoisson
from .core.roi import select_online_circle


def reconstruct_from_fft(detector_fft, params, *, roi=None):
    """Return numerical results from a centered orthonormal detector spectrum.

    roi is internal stream state: a recorded physical aperture and its source.
    DPC is rad/m and phase_error is rad, both on the nominal grating-plane grid
    obtained from detector coordinates via the configured magnification.
    phase_error integrates DPC after removing its fitted affine background;
    it is the aberration phase rather than the full spherical wavefront phase.
    """
    params = normalize_config(params)
    spectrum = np.asarray(detector_fft)
    if (
        spectrum.ndim != 2
        or min(spectrum.shape) < 2
        or not np.all(np.isfinite(spectrum))
    ):
        raise ValueError("detector_fft must be a finite nonempty 2D spectrum")
    harmonics = extract_harmonics(spectrum, params)
    del spectrum, detector_fft
    geometry, roi_source = (
        select_online_circle(harmonics, params) if roi is None else roi
    )
    # Grating-plane sampling is detector reconstruction sampling divided by nominal M.
    sampling = harmonics["wavefront_pixel_size"]
    params.update(
        circle_center_m=geometry["center_m"], circle_radius_m=geometry["radius_m"]
    )
    mask, geometry = circular_domain(
        harmonics["int01"].shape,
        sampling,
        center_m=geometry["center_m"],
        radius_m=geometry["radius_m"],
    )
    graph = SupportedPoisson(
        harmonics["int01"],
        harmonics["int10"],
        params["support_threshold"],
        support_mask=mask,
    )
    dpc = [
        carrier_phase_to_wavefront_gradient(
            graph.unwrap(harmonics[f"wrapped_phase_{axis}_rad"]),
            params["pattern_period"],
            params["det2sample"],
            params["wavelength"],
            params["harmonic_peak_sign"],
        )
        for axis in ("x", "y")
    ]
    fit_mask, _ = circular_domain(
        mask.shape,
        sampling,
        center_m=geometry["center_m"],
        radius_m=geometry["radius_m"] * params["dpc_fit_valid_fraction"],
    )
    fit_mask &= graph.support
    fit = fit_dpc_background(*dpc, sampling, params["wavelength"], valid_mask=fit_mask)
    fitted = [
        np.where(graph.support, fit["dpc_fit"][f"dpc_{axis}_fit"], np.nan)
        for axis in ("x", "y")
    ]
    residual = [gradient - background for gradient, background in zip(dpc, fitted)]
    # Integrate with grating-plane sampling to obtain grating-plane phase error in rad.
    # Fitted tilt and X/Y curvature terms are removed; full spherical phase is excluded.
    phase = graph.integrate(*residual, sampling)
    if not np.all(np.isfinite(phase[graph.support])):
        raise ValueError("non-finite reconstructed phase inside measured support")
    return {
        "dpc_x": dpc[0],
        "dpc_y": dpc[1],
        "dpc_x_fit": fitted[0],
        "dpc_y_fit": fitted[1],
        "dpc_x_residual": residual[0],
        "dpc_y_residual": residual[1],
        "phase_error": phase,
        "fit_parameters": fit["fit_params"],
        "fit_mask": fit_mask,
        "reconstruction_support": graph.support,
        "reconstruction_roi_mask": graph.roi_mask,
        "reconstruction_roi": geometry,
        "roi_selection_source": roi_source,
        "detector_pixel_size_m": tuple(harmonics["detector_pixel_size"]),
        "wavefront_pixel_size_m": tuple(sampling),
        "phase_plane": "grating",
        "dpc_plane": "grating",
        "params": params,
        "harmonic_peak_diagnostics": harmonics["harmonic_peak_diagnostics"],
        **{
            key: harmonics[key]
            for key in (
                "int00",
                "int01",
                "int10",
                "wrapped_phase_x_rad",
                "wrapped_phase_y_rad",
            )
        },
    }


def reconstruct_wavefront(detector_image, params):
    """Numerical reconstruction of an already corrected/aligned detector array.

    For raw frames and complete Zernike/focus outputs, use Pipeline.process().
    Unsupported phase and DPC pixels are NaN.
    """
    return reconstruct_from_fft(centered_fft(validate_image(detector_image)), params)
