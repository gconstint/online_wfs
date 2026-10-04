"""Least-squares unwrapping and integration on a physical circular domain.

Both Poisson problems use the same connected circular support; extrapolated dark
pixels never enter either solve. No reference, truth, or fitted aberration basis
is used. Pixels outside support are NaN, not invented wavefront measurements.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

from .arrays import normalize_pixel_size


class SupportedPoisson:
    """Weighted graph least squares: min sum_e w_e (phi_j-phi_i-d_e)^2."""

    def __init__(
        self,
        amplitude_x,
        amplitude_y,
        support_threshold=0.01,
        *,
        support_mask=None,
    ):
        if not 0 < support_threshold < 1:
            raise ValueError("support_threshold must lie in (0, 1)")
        ax, ay = np.asarray(amplitude_x, float), np.asarray(amplitude_y, float)
        if ax.shape != ay.shape or ax.ndim != 2 or min(ax.shape) < 3:
            raise ValueError("Expected matching 2D harmonic amplitudes")
        if (
            not np.all(np.isfinite(ax))
            or not np.all(np.isfinite(ay))
            or np.any(ax < 0)
            or np.any(ay < 0)
            or min(ax.max(), ay.max()) <= 0
        ):
            raise ValueError("Invalid harmonic amplitudes")
        reliability = np.minimum(ax / ax.max(), ay / ay.max())
        if support_mask is None:
            raise ValueError("an explicit circular support_mask is required")
        self.roi_mask = np.asarray(support_mask, dtype=bool).copy()
        if self.roi_mask.shape != ax.shape or not np.any(self.roi_mask):
            raise ValueError(
                "support_mask must be nonempty and match harmonic amplitudes"
            )
        if label(self.roi_mask)[1] != 1:
            raise ValueError("support_mask must be one connected region")
        self.support = self.roi_mask & (reliability >= support_threshold)
        if label(self.support)[1] != 1:
            raise ValueError(
                "frame measurements inside the online circle must form one connected region"
            )
        self.horizontal = self.support[:, :-1] & self.support[:, 1:]
        self.vertical = self.support[:-1] & self.support[1:]
        indices = np.full(ax.shape, -1, dtype=int)
        self.size = int(self.support.sum())
        if self.size < 16:
            raise ValueError("Reconstruction support is too small")
        indices[self.support] = np.arange(self.size)
        self.start = np.r_[
            indices[:, :-1][self.horizontal], indices[:-1][self.vertical]
        ]
        self.end = np.r_[indices[:, 1:][self.horizontal], indices[1:][self.vertical]]
        self.weights = np.r_[
            np.minimum(reliability[:, :-1], reliability[:, 1:])[self.horizontal],
            np.minimum(reliability[:-1], reliability[1:])[self.vertical],
        ]
        i, j, w = self.start, self.end, self.weights
        matrix = coo_matrix(
            (np.r_[w, w, -w, -w], (np.r_[i, j, i, j], np.r_[i, j, j, i])),
            shape=(self.size, self.size),
        ).tocsc()
        # One gauge constraint in one connected component; factor once and reuse.
        self.solve_reduced = factorized(matrix[1:, 1:])

    def solve(self, horizontal_difference, vertical_difference):
        values = np.r_[
            horizontal_difference[self.horizontal], vertical_difference[self.vertical]
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite phase difference inside measured support")
        rhs = np.bincount(
            self.end, weights=self.weights * values, minlength=self.size
        ) - np.bincount(self.start, weights=self.weights * values, minlength=self.size)
        solution = np.r_[0.0, self.solve_reduced(rhs[1:])]
        solution -= solution.mean()
        output = np.full(self.support.shape, np.nan)
        output[self.support] = solution
        return output

    def unwrap(self, phase):
        phase = np.asarray(phase, dtype=float)
        if phase.shape != self.support.shape:
            raise ValueError("wrapped phase must match the reconstruction grid")
        return self.solve(
            np.angle(np.exp(1j * np.diff(phase, axis=1))),
            np.angle(np.exp(1j * np.diff(phase, axis=0))),
        )

    def integrate(self, gx, gy, sampling_yx):
        gx, gy = np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)
        if gx.shape != self.support.shape or gy.shape != self.support.shape:
            raise ValueError("gradients must match the reconstruction grid")
        py, px = normalize_pixel_size(sampling_yx)
        # Trapezoidal quadrature converts collocated gradients to edge differences.
        return self.solve(
            0.5 * (gx[:, :-1] + gx[:, 1:]) * px, 0.5 * (gy[:-1] + gy[1:]) * py
        )
