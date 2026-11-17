"""Measured-edge weighted integration with an FFT/FC preconditioner."""

import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.ndimage import label
from scipy.sparse.linalg import LinearOperator, cg

from ..defaults import cpu_workers
from .arrays import normalize_pixel_size

_RELATIVE_TOLERANCE = 1e-9
_MAX_ITERATIONS = 2000


class WeightedCGIntegrator:
    """Solve weighted gradient least squares on the measured support.

    This follows Jetson's CircleFC model: trapezoidal measured edge differences,
    minimum endpoint reliability, and an edge-compatible Fourier inverse used
    to precondition CG. The final phase solves the masked open-boundary system;
    exterior gradients are never treated as measured zeros.

    CPU arithmetic remains float64. Convergence is checked with a fresh b-A*phi
    residual, and the supported phase mean fixes the piston. Each instance
    builds its own Fourier coefficients, support and weights.
    """

    def __init__(self, support, reliability=None):
        self.support = np.asarray(support, bool).copy()
        if self.support.ndim != 2 or min(self.support.shape) < 2:
            raise ValueError("integration requires a two-dimensional field")
        if self.support.sum() < 4 or label(self.support)[1] != 1:
            raise ValueError("integration support must be one connected region")
        self.horizontal = self.support[:, :-1] & self.support[:, 1:]
        self.vertical = self.support[:-1] & self.support[1:]
        weights = (
            np.ones(self.support.shape)
            if reliability is None
            else np.asarray(reliability, float)
        )
        if weights.shape != self.support.shape or not np.all(
            np.isfinite(weights[self.support]) & (weights[self.support] > 0)
        ):
            raise ValueError("integration weights must be positive on support")
        self.wx = np.where(
            self.horizontal, np.minimum(weights[:, :-1], weights[:, 1:]), 0
        )
        self.wy = np.where(self.vertical, np.minimum(weights[:-1], weights[1:]), 0)
        # Jetson uses power-of-two FFT workspaces without resampling measured data.
        self.transform_shape = tuple(
            1 << (n - 1).bit_length() for n in self.support.shape
        )
        self.fft_workers = cpu_workers()
        h, w = self.transform_shape
        sy = np.sin(np.pi * np.arange(h)[:, None] / h)
        sx = np.sin(np.pi * np.arange(w // 2 + 1)[None, :] / w)
        # Forward edge symbol Q=exp(ik)-1, rather than the continuous ik.
        eigenvalues = 4 * (sx**2 + sy**2)
        eigenvalues[0, 0] = 1
        self.inverse_eigenvalues = 1 / eigenvalues
        self.inverse_eigenvalues[0, 0] = 0
        self.iterations = 0
        self.relative_residual = 0.0

    def _multiply(self, values):
        field = np.zeros(self.support.shape)
        field[self.support] = values
        hx = np.diff(field, axis=1) * self.wx
        hy = np.diff(field, axis=0) * self.wy
        output = np.zeros(self.support.shape)
        output[:, :-1] -= hx
        output[:, 1:] += hx
        output[:-1] -= hy
        output[1:] += hy
        return output[self.support] + values.mean()

    def _precondition(self, values):
        field = np.zeros(self.transform_shape)
        h, w = self.support.shape
        field[:h, :w][self.support] = values - values.mean()
        spectrum = rfft2(field, workers=self.fft_workers)
        spectrum *= self.inverse_eigenvalues
        result = irfft2(spectrum, s=self.transform_shape, workers=self.fft_workers)[
            :h, :w
        ][self.support]
        return result - result.mean() + values.mean()

    def solve(self, horizontal_difference, vertical_difference):
        """Solve from edge phase increments in rad, excluding unsupported edges."""
        dx = np.asarray(horizontal_difference, float)
        dy = np.asarray(vertical_difference, float)
        if dx.shape != self.horizontal.shape or dy.shape != self.vertical.shape:
            raise ValueError("edge differences must match the integration grid")
        if not np.all(np.isfinite(dx[self.horizontal])) or not np.all(
            np.isfinite(dy[self.vertical])
        ):
            raise ValueError("edge differences must be finite inside measured support")
        hx = np.where(self.horizontal, dx, 0) * self.wx
        hy = np.where(self.vertical, dy, 0) * self.wy
        rhs = np.zeros(self.support.shape)
        rhs[:, :-1] -= hx
        rhs[:, 1:] += hx
        rhs[:-1] -= hy
        rhs[1:] += hy
        values = rhs[self.support]
        values -= values.mean()
        norm = np.linalg.norm(values)
        self.iterations = 0
        self.relative_residual = 0.0
        if norm == 0:
            return np.where(self.support, 0.0, np.nan)
        shape = (len(values), len(values))
        operator = LinearOperator(shape, matvec=self._multiply, dtype=float)
        preconditioner = LinearOperator(shape, matvec=self._precondition, dtype=float)

        def count_iteration(_):
            self.iterations += 1

        solution = None
        # A fresh residual certifies convergence; one restart repairs accumulated
        # recursive-residual roundoff, as in CircleFC's true-residual check.
        for _ in range(2):
            solution, info = cg(
                operator,
                values,
                x0=solution,
                M=preconditioner,
                rtol=_RELATIVE_TOLERANCE,
                atol=0,
                maxiter=_MAX_ITERATIONS,
                callback=count_iteration,
            )
            solution -= solution.mean()
            residual = values - self._multiply(solution)
            self.relative_residual = float(np.linalg.norm(residual) / norm)
            if np.isfinite(self.relative_residual) and (
                self.relative_residual <= _RELATIVE_TOLERANCE
            ):
                output = np.full(self.support.shape, np.nan)
                output[self.support] = solution
                return output
            if info != 0 or not np.isfinite(self.relative_residual):
                break
        raise RuntimeError(
            "weighted FC phase integration did not converge "
            f"(cg={info}, true relative residual={self.relative_residual:.3g})"
        )

    def integrate(self, gx, gy, pixel_size):
        """Integrate physical gradients to phase in rad; mask exterior with NaN."""
        gx, gy = np.asarray(gx, float), np.asarray(gy, float)
        if gx.shape != self.support.shape or gy.shape != self.support.shape:
            raise ValueError("gradient fields must match the integration grid")
        if not np.all(np.isfinite(gx[self.support])) or not np.all(
            np.isfinite(gy[self.support])
        ):
            raise ValueError("gradients must be finite inside measured support")
        py, px = normalize_pixel_size(pixel_size)
        return self.solve(
            0.5 * (gx[:, :-1] + gx[:, 1:]) * px,
            0.5 * (gy[:-1] + gy[1:]) * py,
        )
