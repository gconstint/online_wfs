"""Open-boundary weighted integration with a DCT-preconditioned CG solve."""

from os import cpu_count

import numpy as np
from scipy.fft import dctn, idctn, next_fast_len
from scipy.ndimage import label
from scipy.sparse.linalg import LinearOperator, cg

from .arrays import normalize_pixel_size


class GradientIntegrator:
    """Weighted least squares for collocated gradients on one connected mask.

    A DCT inverse is exact for a full uniform rectangle and preconditions the
    masked solve. Memory grows linearly with image size, without sparse LU fill.
    """

    def __init__(self, support, reliability=None):
        self.support = np.asarray(support, bool).copy()
        if self.support.ndim != 2 or min(self.support.shape) < 2:
            raise ValueError("integration requires a two-dimensional field")
        if self.support.sum() < 4 or label(self.support)[1] != 1:
            raise ValueError("integration support must be one connected region")
        self.horizontal = self.support[:, :-1] & self.support[:, 1:]
        self.vertical = self.support[:-1] & self.support[1:]
        if reliability is None:
            weights = np.ones(self.support.shape)
        else:
            weights = np.asarray(reliability, float)
            if weights.shape != self.support.shape or not np.all(
                np.isfinite(weights[self.support]) & (weights[self.support] > 0)
            ):
                raise ValueError("integration weights must be positive on support")
        self.wx = np.where(
            self.horizontal, np.minimum(weights[:, :-1], weights[:, 1:]), 0
        )
        self.wy = np.where(self.vertical, np.minimum(weights[:-1], weights[1:]), 0)
        h, w = self.support.shape
        self.uniform_rectangle = bool(self.support.all() and reliability is None)
        self.transform_shape = (
            (h, w) if self.uniform_rectangle else (next_fast_len(h), next_fast_len(w))
        )
        self.fft_workers = min(4, cpu_count() or 1)
        th, tw = self.transform_shape
        self.eigenvalues = (
            2
            - 2 * np.cos(np.pi * np.arange(th)[:, None] / th)
            + 2
            - 2 * np.cos(np.pi * np.arange(tw)[None, :] / tw)
        )
        self.eigenvalues[0, 0] = 1

    def _dct_inverse(self, rhs):
        # FFT workers accelerate only the preconditioner, preserving the measured solve.
        spectrum = (
            dctn(
                rhs,
                type=2,
                norm="ortho",
                s=self.transform_shape,
                workers=self.fft_workers,
            )
            / self.eigenvalues
        )
        spectrum[0, 0] = 0
        result = idctn(spectrum, type=2, norm="ortho", workers=self.fft_workers)
        return result[: rhs.shape[0], : rhs.shape[1]]

    def solve(self, horizontal_difference, vertical_difference):
        dx, dy = (
            np.asarray(horizontal_difference, float),
            np.asarray(vertical_difference, float),
        )
        del horizontal_difference, vertical_difference
        if dx.shape != self.horizontal.shape or dy.shape != self.vertical.shape:
            raise ValueError("edge differences must match the integration grid")
        if not np.all(np.isfinite(dx[self.horizontal])) or not np.all(
            np.isfinite(dy[self.vertical])
        ):
            raise ValueError("nonfinite edge difference inside measured support")
        rhs = np.zeros(self.support.shape)
        hx = np.where(self.horizontal, dx, 0) * self.wx
        hy = np.where(self.vertical, dy, 0) * self.wy
        rhs[:, :-1] -= hx
        rhs[:, 1:] += hx
        rhs[:-1] -= hy
        rhs[1:] += hy
        # Edge data has been accumulated into rhs; CG only needs the node values.
        del dx, dy, hx, hy
        if self.uniform_rectangle:
            return self._dct_inverse(rhs)

        n = int(self.support.sum())

        def multiply(values):
            field = np.zeros(self.support.shape)
            field[self.support] = values
            hx = np.diff(field, axis=1) * self.wx
            hy = np.diff(field, axis=0) * self.wy
            out = np.zeros_like(field)
            out[:, :-1] -= hx
            out[:, 1:] += hx
            out[:-1] -= hy
            out[1:] += hy
            return out[self.support] + values.mean()

        def precondition(values):
            field = np.zeros(self.support.shape)
            field[self.support] = values - values.mean()
            out = self._dct_inverse(field)[self.support]
            return out - out.mean() + values.mean()

        operator = LinearOperator((n, n), matvec=multiply, dtype=float)
        preconditioner = LinearOperator((n, n), matvec=precondition, dtype=float)
        values = rhs[self.support]
        del rhs
        values -= values.mean()

        solution, info = cg(
            operator,
            values,
            M=preconditioner,
            rtol=1e-9,
            atol=0,
            maxiter=2000,
        )
        if info != 0:
            raise RuntimeError(
                f"open-boundary phase integration did not converge (cg={info})"
            )
        output = np.full(self.support.shape, np.nan)
        output[self.support] = solution - solution.mean()
        return output

    def integrate(self, gx, gy, pixel_size):
        gx, gy = np.asarray(gx, float), np.asarray(gy, float)
        if gx.shape != self.support.shape or gy.shape != self.support.shape:
            raise ValueError("gradient fields must match the integration grid")
        py, px = normalize_pixel_size(pixel_size)
        return self.solve(
            0.5 * (gx[:, :-1] + gx[:, 1:]) * px, 0.5 * (gy[:-1] + gy[1:]) * py
        )
