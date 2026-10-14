"""Masked skimage unwrapping and weighted Poisson phase integration."""

from __future__ import annotations

from inspect import signature

import numpy as np
from scipy.ndimage import label
from skimage.restoration import unwrap_phase

from .arrays import normalize_pixel_size
from .integration import GradientIntegrator


class SupportedPoisson(GradientIntegrator):
    """Measured support for 2-D unwrapping and weighted phase integration."""

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
        ):
            raise ValueError("Invalid harmonic amplitudes")
        if support_mask is None:
            raise ValueError("an explicit circular support_mask is required")
        roi_mask = np.asarray(support_mask, dtype=bool)
        if roi_mask.shape != ax.shape or not np.any(roi_mask):
            raise ValueError(
                "support_mask must be nonempty and match harmonic amplitudes"
            )
        maxima = [amplitude[roi_mask].max() for amplitude in (ax, ay)]
        if min(maxima) <= 0:
            raise ValueError("both carriers must have nonzero measured amplitude")
        reliability = np.minimum(ax / maxima[0], ay / maxima[1])
        support = roi_mask & (reliability >= support_threshold)
        candidates = int(support.sum())
        if candidates < 16:
            raise ValueError("measured support must contain at least 16 pixels")
        components, count = label(support)
        if count > 1:
            sizes = np.bincount(components.ravel())
            largest = int(np.argmax(sizes[1:]) + 1)
            discarded = candidates - int(sizes[largest])
            if discarded > 0.01 * candidates or sizes[largest] < 16:
                raise ValueError(
                    "measured support is substantially fragmented; check the aperture "
                    "and carrier amplitudes"
                )
            support = components == largest
        super().__init__(support, reliability)

    def unwrap(self, phase):
        """Unwrap in 2-D and anchor the integer 2*pi branch at a supported pixel."""
        phase = np.asarray(phase, dtype=float)
        if phase.shape != self.support.shape:
            raise ValueError("wrapped phase must match the reconstruction grid")
        if not np.all(np.isfinite(phase[self.support])):
            raise ValueError("wrapped phase must be finite on measured support")
        field = np.ma.array(np.where(self.support, phase, 0), mask=~self.support)
        parameters = signature(unwrap_phase).parameters
        seed = {key: 0 for key in ("rng", "seed") if key in parameters}
        unwrapped = np.asarray(unwrap_phase(field, **seed).filled(np.nan), dtype=float)
        anchor = (phase.shape[0] // 2, phase.shape[1] // 2)
        if not self.support[anchor]:
            indices = np.argwhere(self.support)
            anchor = tuple(indices[np.argmin(np.sum((indices - anchor) ** 2, axis=1))])
        # Preserve the measured center branch without centering by the mean.
        branch = np.rint((unwrapped[anchor] - phase[anchor]) / (2 * np.pi))
        unwrapped -= branch * 2 * np.pi
        return unwrapped

    def integrate(self, gx, gy, sampling_yx):
        gx, gy = np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)
        if gx.shape != self.support.shape or gy.shape != self.support.shape:
            raise ValueError("gradients must match the reconstruction grid")
        py, px = normalize_pixel_size(sampling_yx)
        # Trapezoidal quadrature converts collocated gradients to edge differences.
        return self.solve(
            0.5 * (gx[:, :-1] + gx[:, 1:]) * px, 0.5 * (gy[:-1] + gy[1:]) * py
        )
