"""One ordinary call per camera frame; all requested outputs return together."""

from copy import deepcopy

import numpy as np

from .analysis.beam import fit_profile_fwhm
from .analysis.envelope import fit_detector_fwhm
from .analysis.focus import focus_distance_from_dpc
from .analysis.zernike import fit_zernike_phase
from .config import normalize_config
from .core.arrays import validate_image
from .preprocessing import preprocess_image
from .reconstruction import reconstruct_wavefront


def _snapshot(image, name):
    if image is None:
        return None
    snapshot = validate_image(image, name).copy()
    snapshot.setflags(write=False)
    return snapshot


class Pipeline:
    """Absolute wavefront processor for serial calls from one camera stream.

    Dark/flat calibration is snapshotted at setup. The physical aperture is
    reused; Poisson weights follow each frame's amplitudes.
    Zernike fitting uses bounded-memory blocks on the full reconstruction grid.
    Inputs are physical parameters and arrays. The processor performs no file
    access, plotting, console interaction or elapsed-time recording.
    """

    def __init__(self, params, *, dark=None, flat=None):
        self._params = normalize_config(params)
        self._dark, self._flat = (
            _snapshot(dark, "dark image"),
            _snapshot(flat, "flat image"),
        )
        self._roi = None
        self._roi_shape = None

    @property
    def params(self):
        """Return a copy of the physical inputs and effective numerical defaults."""
        return deepcopy(self._params)

    def process(self, image):
        """Return DPC, phase error, Zernike residual and focus estimates for a frame.

        DPC units are rad/m; phase units are rad; lengths are meters. Focus
        distances are nonnegative Gaussian-q estimates from the grating plane,
        with the nominal spherical carrier curvature restored before inference.
        All four maps use the nominal grating-plane grid, x_g = x_d / M.
        phase_error excludes the fitted affine DPC background and applies the
        configured phase cleanup (x/y/xy removal by default); residual_phase
        further subtracts its Zernike fit on that same plane. abs(Re(q_grating))
        gives a distance magnitude and does not encode upstream/downstream.
        Only the four requested arrays and two focus distances are returned.
        """
        corrected = preprocess_image(
            image,
            dark=self._dark,
            flat=self._flat,
            rotation_angle=self._params["rotation_angle"],
        )
        image_shape = corrected.shape
        roi = self._roi if self._roi_shape == image_shape else None
        result = reconstruct_wavefront(corrected, self._params, roi=roi)
        del corrected
        if self._params["beam_width_method"] == "gaussian_envelope":
            widths = fit_detector_fwhm(result["int00"], self._params["pixel_size"])
        else:
            widths = fit_profile_fwhm(result["int00"], self._params["pixel_size"])
        del result["int00"]
        # Fit Zernike terms to the grating-plane phase error; the residual uses the same plane.
        geometry = result["reconstruction_roi"]
        residual_phase = fit_zernike_phase(
            result["phase_error"],
            result["wavefront_pixel_size_m"],
            np.isfinite(result["phase_error"]),
            geometry["radius_m"],
            center_m=geometry["center_m"],
        )
        focus = focus_distance_from_dpc(
            *result["dpc_slopes"],
            *widths,
            self._params["wavelength"],
            self._params["source_dist"],
            self._params["det2sample"],
        )
        # Only successful frames update the stream aperture; failed frames can be retried.
        self._roi = deepcopy(geometry)
        self._roi_shape = image_shape
        return {
            # Carrier-demodulated DPC on the grating-plane grid, in rad/m.
            "dpc_x": result["dpc_x"],
            "dpc_y": result["dpc_y"],
            # Grating-plane phase error in rad, with the fitted DPC background removed.
            "phase_error": result["phase_error"],
            # Phase error minus its Zernike fit on the same grating plane, in rad.
            "residual_phase": residual_phase,
            # Grating-to-waist distance magnitudes in m, without upstream/downstream signs.
            "focus_distance_x_m": abs(focus[0]),
            "focus_distance_y_m": abs(focus[1]),
        }
