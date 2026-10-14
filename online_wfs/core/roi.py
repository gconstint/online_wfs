"""Fixed physical aperture selection on the detector-resolved grating grid."""

from .arrays import circular_domain


def select_reconstruction_circle(harmonics, params):
    """Return the physical aperture mask and geometry without brightness clipping."""
    return circular_domain(
        harmonics["int01"].shape,
        harmonics["wavefront_pixel_size"],
        center_m=params["circle_center_m"],
        radius_m=params["circle_radius_m"],
        radius_fraction=params["circle_radius_fraction"],
    )
