"""Choose a usable physical circle without an interactive aperture selector."""

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .arrays import circular_domain


def select_online_circle(harmonics, params):
    """Keep the SGI default circle when connected; otherwise inscribe a circle.

    The fallback uses measured first-order harmonic confidence at the unchanged
    1% threshold. A physical distance transform selects the largest supported
    circle. Its geometry is explicit, and unsupported pixels are never filled.
    """
    ax = harmonics["support_amplitude_x"]
    ay = harmonics["support_amplitude_y"]
    if (
        not np.all(np.isfinite(ax))
        or not np.all(np.isfinite(ay))
        or min(ax.max(), ay.max()) <= 0
    ):
        raise ValueError("invalid harmonic amplitudes for automatic online ROI")
    reliability = np.minimum(ax / ax.max(), ay / ay.max())
    valid = reliability >= params["support_threshold"]
    sampling = harmonics["wavefront_pixel_size"]
    mask, geometry = circular_domain(valid.shape, sampling)
    if label(mask & valid)[1] == 1 and (mask & valid).sum() >= 36:
        return geometry, "default_circle"
    # Padding includes the image boundary among the invalid measurements.
    ny, nx = valid.shape
    distance = distance_transform_edt(np.pad(valid, 1), sampling=sampling)[1:-1, 1:-1]
    y = (np.arange(ny) - ny / 2) * sampling[0]
    x = (np.arange(nx) - nx / 2) * sampling[1]
    maximum = np.minimum(
        ny * sampling[0] / 2 - np.abs(y[:, None]),
        nx * sampling[1] / 2 - np.abs(x[None, :]),
    )
    # Keep the nearest invalid pixel outside the circle, including exact ties.
    radius_map = np.minimum(distance * (1 - 32 * np.finfo(float).eps), maximum)
    row, col = np.unravel_index(np.argmax(radius_map), distance.shape)
    center = ((col - nx / 2) * sampling[1], (row - ny / 2) * sampling[0])
    radius = radius_map[row, col]
    if radius <= 0:
        raise ValueError("no measured circle is available for online reconstruction")
    mask, geometry = circular_domain(
        valid.shape, sampling, center_m=center, radius_m=radius
    )
    if mask.sum() < 36 or not np.all(valid[mask]) or label(mask)[1] != 1:
        raise ValueError(
            "no supported circle is large enough for online reconstruction and 36 Zernike terms"
        )
    return geometry, "automatic_harmonic_support"
