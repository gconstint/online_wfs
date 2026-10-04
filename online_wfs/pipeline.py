"""One ordinary call per camera frame; all requested outputs return together."""

from copy import deepcopy

from .analysis.beam import calc_beam_size
from .analysis.focus import focus_distance_from_dpc
from .analysis.zernike import ZernikeFitter
from .config import load_config
from .core.arrays import centered_fft, validate_image
from .io import load_image, preprocess_image
from .reconstruction import reconstruct_from_fft


def _snapshot(image, name):
    if image is None:
        return None
    snapshot = validate_image(image, name).copy()
    snapshot.setflags(write=False)
    return snapshot


class Pipeline:
    """Absolute wavefront processor for serial calls from one camera stream.

    Dark/flat calibration is snapshotted at setup. Aperture geometry and the
    Zernike projector are reused; Poisson weights follow each frame's amplitudes.
    The processor neither interacts with the user nor records elapsed time.
    """

    def __init__(self, params, *, dark=None, flat=None):
        self._params = load_config(params)
        if dark is None and self._params["dark_image_path"]:
            dark = load_image(self._params["dark_image_path"])
        if flat is None and self._params["flat_image_path"]:
            flat = load_image(self._params["flat_image_path"])
        self._dark, self._flat = (
            _snapshot(dark, "dark image"),
            _snapshot(flat, "flat image"),
        )
        self._zernike = ZernikeFitter()
        self._roi = None
        self._roi_shape = None

    @property
    def params(self):
        """Return a copy of the physical inputs and effective numerical defaults."""
        return deepcopy(self._params)

    def process(self, image=None):
        """Return DPC, phase error, Zernike residual and focus estimates for a frame.

        DPC units are rad/m; phase units are rad; lengths are meters. Focus
        distances are nonnegative Gaussian-q estimates from the grating plane,
        with the nominal spherical carrier curvature restored before inference.
        All four maps use the nominal grating-plane grid, x_g = x_d / M.
        phase_error excludes the fitted affine DPC background; residual_phase
        further subtracts its Zernike fit on that same plane. abs(Re(q_grating))
        gives a distance magnitude and does not encode upstream/downstream.
        Only the four requested arrays and two focus distances are returned.
        """
        if image is None:
            if not self._params["image_path"]:
                raise ValueError("provide an image array or image_path")
            image = load_image(self._params["image_path"])
        corrected = preprocess_image(
            image,
            dark=self._dark,
            flat=self._flat,
            rotation_angle=self._params["rotation_angle"],
        )
        image_shape = corrected.shape
        roi = self._roi if self._roi_shape == image_shape else None
        result = reconstruct_from_fft(centered_fft(corrected), self._params, roi=roi)
        # Fit Zernike terms to the grating-plane phase error; the residual uses the same plane.
        zernike = self._zernike.fit(
            result["phase_error"],
            result["wavefront_pixel_size_m"],
            self._params["wavelength"],
            result["reconstruction_roi"],
        )
        _, size = calc_beam_size(result["int00"], result["detector_pixel_size_m"])
        fit = result["fit_parameters"]
        waist = focus_distance_from_dpc(
            fit["slope_x"],
            fit["slope_y"],
            size["fwhm_x"],
            size["fwhm_y"],
            self._params["wavelength"],
            self._params["source_dist"],
            self._params["det2sample"],
        )
        # Only successful frames update the stream aperture; failed frames can be retried.
        self._roi = (
            deepcopy(result["reconstruction_roi"]),
            result["roi_selection_source"],
        )
        self._roi_shape = image_shape
        return {
            # Carrier-demodulated DPC on the grating-plane grid, in rad/m.
            "dpc_x": result["dpc_x"],
            "dpc_y": result["dpc_y"],
            # Grating-plane phase error in rad, with the fitted DPC background removed.
            "phase_error": result["phase_error"],
            # Phase error minus its Zernike fit on the same grating plane, in rad.
            "residual_phase": zernike["residual"],
            # Grating-to-waist distance magnitudes in m, without upstream/downstream signs.
            "focus_distance_x_m": abs(waist["distance_x"]),
            "focus_distance_y_m": abs(waist["distance_y"]),
        }
