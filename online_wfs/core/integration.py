"""Select weighted FFT-preconditioned CG or direct Frankot-Chellappa integration."""

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2
from scipy.ndimage import label

from ..defaults import cpu_workers
from ._integration_cg import WeightedCGIntegrator
from .arrays import normalize_pixel_size


class DirectFCIntegrator:
    """Project gradients onto an integrable field with one Fourier FC solve.

    Physical gradients are scaled to phase per pixel. Unsupported values are
    zero extended, then reflected with the derivative parity used by WavePy:
    gx is odd in x/even in y; gy is even in x/odd in y. Two forward FFTs and
    one inverse FFT give the phase without an iterative solver or weights.

    This rectangular reflection model changes the former weighted open-boundary
    target. The returned phase has the original shape, NaN outside support and
    zero mean on support. Fourier coefficients are built for each instance.
    """

    def __init__(self, support):
        self.support = np.asarray(support, bool).copy()
        if self.support.ndim != 2 or min(self.support.shape) < 2:
            raise ValueError("integration requires a two-dimensional field")
        if self.support.sum() < 4 or label(self.support)[1] != 1:
            raise ValueError("integration support must be one connected region")
        self.transform_shape = tuple(2 * size for size in self.support.shape)
        self.fft_workers = cpu_workers()
        height, width = self.transform_shape
        ky = 2 * np.pi * fftfreq(height)[:, None]
        kx = 2 * np.pi * fftfreq(width)[None, :]
        denominator = kx**2 + ky**2
        denominator[0, 0] = 1
        # Continuous derivative symbols on the pixel grid: d/dx <-> i*kx.
        self.coefficients_x, self.coefficients_y = (
            -1j * kx / denominator,
            -1j * ky / denominator,
        )

    @staticmethod
    def _reflect(gradient, odd_axis):
        """Reflect a gradient to twice each axis, changing its normal sign."""
        height, width = gradient.shape
        padded = np.empty((2 * height, 2 * width), dtype=float)
        padded[:height, :width] = gradient
        padded[height:, :width] = gradient[::-1] * (-1 if odd_axis == 0 else 1)
        padded[:height, width:] = gradient[:, ::-1] * (-1 if odd_axis == 1 else 1)
        padded[height:, width:] = -gradient[::-1, ::-1]
        return padded

    def integrate(self, gx, gy, pixel_size):
        """Integrate physical gradients (rad/m) to a real phase map (rad)."""
        if np.iscomplexobj(gx) or np.iscomplexobj(gy):
            raise ValueError("gradient fields must be real-valued")
        gx, gy = np.asarray(gx, float), np.asarray(gy, float)
        if gx.shape != self.support.shape or gy.shape != self.support.shape:
            raise ValueError("gradient fields must match the integration grid")
        if not np.all(np.isfinite(gx[self.support])) or not np.all(
            np.isfinite(gy[self.support])
        ):
            raise ValueError("gradients must be finite inside measured support")
        py, px = normalize_pixel_size(pixel_size)
        gx = np.where(self.support, gx, 0) * px
        gy = np.where(self.support, gy, 0) * py
        spectrum = fft2(self._reflect(gx, odd_axis=1), workers=self.fft_workers)
        spectrum *= self.coefficients_x
        spectrum += (
            fft2(self._reflect(gy, odd_axis=0), workers=self.fft_workers)
            * self.coefficients_y
        )
        spectrum[0, 0] = 0
        height, width = self.support.shape
        phase = ifft2(spectrum, workers=self.fft_workers).real[:height, :width].copy()
        phase -= phase[self.support].mean()
        phase[~self.support] = np.nan
        return phase


class GradientIntegrator:
    """Integrate gradients with an explicit method, defaulting to ``fc_cg``.

    ``fc_cg`` solves the original weighted open-boundary system with Fourier
    preconditioning and CG. ``fc`` applies the unweighted reflected FC
    projection. Reliability supplies weights only to ``fc_cg``. Both methods
    return the original grid with zero supported mean and NaN outside support.
    """

    def __init__(self, support, reliability=None, *, method="fc_cg"):
        if method == "fc_cg":
            self._solver = WeightedCGIntegrator(support, reliability)
        elif method == "fc":
            self._solver = DirectFCIntegrator(support)
        else:
            raise ValueError("integration method must be 'fc_cg' or 'fc'")
        self.method = method
        self.support = self._solver.support
        self.transform_shape = self._solver.transform_shape

    @property
    def iterations(self):
        """CG iteration count; the direct FC method reports zero."""
        return self._solver.iterations if self.method == "fc_cg" else 0

    @property
    def relative_residual(self):
        """True CG system residual; direct FC has no iterative residual."""
        return self._solver.relative_residual if self.method == "fc_cg" else None

    def integrate(self, gx, gy, pixel_size):
        """Integrate physical gradients to phase with the selected method."""
        return self._solver.integrate(gx, gy, pixel_size)
