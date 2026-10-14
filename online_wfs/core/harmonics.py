"""Detector-resolved exact-carrier demodulation with a super-Gaussian filter."""

import numpy as np

from .dpc_conversion import detector_to_wavefront_pixel_size


def spatial_carrier_phases(
    shape: tuple[int, int],
    carrier_period_px: tuple[float, float] | np.ndarray,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact fractional carrier phases on the pixel-centered grid."""
    periods = np.asarray(carrier_period_px, dtype=float)
    if periods.shape != (2,) or np.any(periods <= 2.0):
        raise ValueError("carrier periods must be [u,v] values above 2 pixels")
    y = np.arange(shape[0], dtype=float) - (shape[0] - 1) / 2.0
    x = np.arange(shape[1], dtype=float) - (shape[1] - 1) / 2.0
    angle = np.deg2rad(rotation_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    coordinate_u = cosine * x[None, :] + sine * y[:, None]
    coordinate_v = -sine * x[None, :] + cosine * y[:, None]
    phase_u = 2.0 * np.pi * coordinate_u / periods[0]
    phase_v = 2.0 * np.pi * coordinate_v / periods[1]
    return phase_u, phase_v


def _fourier_lowpass(field: np.ndarray, radius_px: float) -> np.ndarray:
    """Apply exp(-0.5 * (r / radius)**8) in unshifted FFT coordinates."""
    if radius_px <= 0:
        raise ValueError("demodulation_radius_px must be positive")
    frequency_y = np.fft.fftfreq(field.shape[0]) * field.shape[0]
    frequency_x = np.fft.fftfreq(field.shape[1]) * field.shape[1]
    radius_squared = frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2
    # An eighth-order super-Gaussian keeps the modulation sideband nearly flat
    # while sharply suppressing the DC, conjugate and orthogonal carrier lobes.
    # A hard circular crop would create stronger ringing near image boundaries.
    mask = np.exp(-0.5 * (radius_squared / radius_px**2) ** 4)
    return np.fft.ifft2(np.fft.fft2(field) * mask)


def demodulation_radius(shape, periods, rotation_deg, requested=None):
    """Keep each low-pass sideband separate from DC and other first harmonics."""
    periods = np.asarray(periods, float)
    # Validate period/rotation using the same carrier geometry as extraction.
    if periods.shape != (2,) or not np.all(np.isfinite(periods) & (periods > 2)):
        raise ValueError(
            "carrier periods must contain two finite values above 2 pixels"
        )
    angle = np.deg2rad(rotation_deg)
    if not np.isfinite(angle):
        raise ValueError("carrier rotation must be finite")
    vectors = (
        np.array([[np.sin(angle), np.cos(angle)], [np.cos(angle), -np.sin(angle)]])
        / periods[:, None]
    )
    if np.any(np.abs(vectors) >= 0.5):
        raise ValueError("carrier exceeds the detector Nyquist frequency")
    lobes = np.vstack((np.zeros(2), vectors, -vectors))
    distances = []
    for carrier in vectors:
        delta = lobes - carrier
        delta -= np.rint(delta)
        spacing = np.linalg.norm(delta * np.asarray(shape), axis=1)
        distances.extend(spacing[spacing > 1e-8])
    nearest = min(distances)
    radius = 0.45 * nearest if requested is None else float(requested)
    if not np.isfinite(radius) or radius <= 0 or radius > 0.8 * nearest:
        raise ValueError(
            "demodulation_radius_px overlaps another carrier; use null/None for automatic bandwidth"
        )
    return radius


def extract_harmonics_from_image(image, params):
    """Demodulate one carrier at a time and retain only integration inputs."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 3 or not np.all(np.isfinite(image)):
        raise ValueError("harmonic extraction requires a finite 2D detector image")
    pixels = np.asarray(params["pixel_size"], float)
    periods = params["pattern_period"] / pixels[::-1]
    radius = demodulation_radius(image.shape, periods, 0.0)
    sign = params["harmonic_peak_sign"]
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("harmonic_peak_sign must be +1 or -1")
    offsets = np.asarray(params.get("carrier_phase_offset_rad", (0.0, 0.0)), float)
    if offsets.shape != (2,) or not np.all(np.isfinite(offsets)):
        raise ValueError(
            "carrier_phase_offset_rad must contain two finite [x,y] radians"
        )
    magnification = params["total_dist"] / params["source_dist"]
    harmonics = {
        # Release the imaginary DC buffer; preserve memory order for profile reductions.
        "int00": _fourier_lowpass(image, radius).real.copy(order="K"),
        "wavefront_pixel_size": detector_to_wavefront_pixel_size(pixels, magnification),
    }
    phases = spatial_carrier_phases(image.shape, periods)
    for axis, amplitude_key, phase, offset in zip(
        ("x", "y"), ("int01", "int10"), phases, offsets
    ):
        carrier = _fourier_lowpass(image * np.exp(-1j * sign * phase), radius)
        # Restore the original detector origin before taking angles or unwrapping.
        if np.any(offsets):
            carrier *= np.exp(-1j * sign * offset)
        harmonics[amplitude_key] = np.abs(carrier)
        harmonics[f"wrapped_phase_{axis}_rad"] = np.angle(carrier)
        del carrier
    return harmonics


def extract_harmonics(detector_fft, params):
    """Accept a centered orthonormal detector spectrum without changing sampling."""
    image = np.fft.ifft2(np.fft.ifftshift(detector_fft), norm="ortho").real
    return extract_harmonics_from_image(image, params)
