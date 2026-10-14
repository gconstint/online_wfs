"""Selected low-order phase removal from a joint quadratic fit."""

import numpy as np

from ..core.arrays import coordinate_grid


def remove_phase_tilt_xy(phase, pixel_size, support, geometry):
    """Subtract only x, y and xy terms; retain piston and pure x²/y² curvature."""
    phase = np.asarray(phase, float)
    support = np.asarray(support, bool) & np.isfinite(phase)
    if support.shape != phase.shape or support.sum() < 6:
        raise ValueError("phase cleanup requires at least six supported samples")
    xx, yy = coordinate_grid(phase.shape, pixel_size)
    cx, cy = geometry["center_m"]
    radius = geometry["radius_m"]
    u, v = (xx - cx) / radius, (yy - cy) / radius
    del xx, yy
    design = np.column_stack(
        [
            np.ones(support.sum()),
            u[support],
            v[support],
            u[support] ** 2,
            (u * v)[support],
            v[support] ** 2,
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, phase[support], rcond=None)
    del design
    if rank != 6:
        raise ValueError("quadratic phase cleanup is rank deficient")
    removed = coefficients[1] * u + coefficients[2] * v + coefficients[4] * u * v
    return np.where(support, phase - removed, np.nan)
