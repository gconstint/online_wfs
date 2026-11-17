"""Absolute SGI unwrapping, affine DPC fitting and phase integration."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .analysis.phase import remove_phase_tilt_xy
from .config import normalize_config
from .defaults import cpu_workers
from .core.arrays import circular_domain, validate_image
from .core.dpc_conversion import carrier_phase_to_wavefront_gradient
from .core.dpc_fit import fit_dpc_background
from .core.harmonics import extract_harmonics_from_image
from .core.poisson import SupportedPoisson
from .core.roi import select_reconstruction_circle


def _reconstruct_harmonics(harmonics, params):
    """Fit and integrate gradients on the small measured harmonic grid."""
    # Grating-plane sampling is detector reconstruction sampling divided by nominal M.
    sampling = harmonics["wavefront_pixel_size"]
    mask, geometry = select_reconstruction_circle(harmonics, params)
    graph = SupportedPoisson(
        harmonics["int01"],
        harmonics["int10"],
        params["support_threshold"],
        support_mask=mask,
        method=params["integration_method"],
    )
    phases = [harmonics[f"wrapped_phase_{axis}_rad"] for axis in ("x", "y")]
    workers = min(2, cpu_workers())
    if workers > 1 and mask.size >= 65536:
        # skimage releases the GIL in its 2-D unwrap; the two carriers are independent.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            unwrapped = list(executor.map(graph.unwrap, phases))
    else:
        unwrapped = [graph.unwrap(phase) for phase in phases]
    dpc = [
        carrier_phase_to_wavefront_gradient(
            phase,
            params["pattern_period"],
            params["det2sample"],
            params["wavelength"],
            params["harmonic_peak_sign"],
        )
        for phase in unwrapped
    ]
    del phases, unwrapped
    border_y, border_x = harmonics["fit_edge_margin_yx"]
    # Amplitudes and wrapped phases have been consumed by support and unwrapping.
    del harmonics
    fit_mask, _ = circular_domain(
        mask.shape,
        sampling,
        center_m=geometry["center_m"],
        radius_m=geometry["radius_m"] * params["dpc_fit_valid_fraction"],
    )
    fit_mask &= graph.support
    fit_mask[:border_y] = False
    fit_mask[-border_y:] = False
    fit_mask[:, :border_x] = False
    fit_mask[:, -border_x:] = False
    if fit_mask.sum() < 16:
        raise ValueError("too few measured pixels remain for DPC fitting")
    fit = fit_dpc_background(*dpc, sampling, valid_mask=fit_mask)
    rows, columns = mask.shape
    y = (np.arange(rows) - (rows - 1) / 2) * sampling[0]
    x = (np.arange(columns) - (columns - 1) / 2) * sampling[1]
    # Broadcast the one-dimensional backgrounds; raw DPC remains an output.
    residual_x = dpc[0] - (fit["slope_x"] * x + fit["intercept_x"])
    residual_y = dpc[1] - (fit["slope_y"] * y + fit["intercept_y"])[:, None]
    del fit_mask, mask
    # Integrate with grating-plane sampling to obtain grating-plane phase error in rad.
    # Fitted tilt and X/Y curvature terms are removed; full spherical phase is excluded.
    phase = graph.integrate(residual_x, residual_y, sampling)
    support = graph.support
    # Phase cleanup needs support, but no longer needs the integrator or gradients.
    del graph, residual_x, residual_y
    if params["phase_cleanup"] == "tilt_xy":
        phase = remove_phase_tilt_xy(phase, sampling, support, geometry)
    if not np.all(np.isfinite(phase[support])):
        raise ValueError("non-finite reconstructed phase inside measured support")
    return {
        "dpc_x": dpc[0],
        "dpc_y": dpc[1],
        "phase_error": phase,
        "dpc_slopes": (fit["slope_x"], fit["slope_y"]),
        "reconstruction_roi": geometry,
        "wavefront_pixel_size_m": tuple(sampling),
    }


def reconstruct_wavefront(detector_image, params):
    """Numerical reconstruction of an already corrected/aligned detector array.

    For raw frames and complete Zernike/focus outputs, use Pipeline.process(image).
    Unsupported phase and DPC pixels are NaN.
    """
    params = normalize_config(params)
    return _reconstruct_harmonics(
        extract_harmonics_from_image(
            validate_image(detector_image),
            params,
            include_dc=False,
        ),
        params,
    )
