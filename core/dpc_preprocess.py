"""
DPC (Differential Phase Contrast) preprocessing module.

Optimized for robustness against experimental noise (NaNs) and memory efficiency.
"""

import numpy as np

# 建议使用 scipy.fft，通常比 numpy.fft 更快
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple


def _create_cosine_edge_taper(
    image_shape: Tuple[int, int], taper_ratio: float = 0.08, dtype=np.float32
) -> np.ndarray:
    """
    Create a cosine edge taper window using broadcasting to save memory.
    """
    height, width = image_shape

    taper_h = max(1, int(height * taper_ratio))
    taper_w = max(1, int(width * taper_ratio))

    # 1D transitions
    # 0.5 * (1 - cos) creates a curve from 0 to 1
    curve_h = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_h, dtype=dtype)))
    curve_w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_w, dtype=dtype)))

    # Construct 1D windows
    win_h = np.ones(height, dtype=dtype)
    win_h[:taper_h] = curve_h
    win_h[-taper_h:] = curve_h[::-1]

    win_w = np.ones(width, dtype=dtype)
    win_w[:taper_w] = curve_w
    win_w[-taper_w:] = curve_w[::-1]

    # Return 2D window using broadcasting (column vector * row vector)
    # shape: (height, 1) * (1, width) -> (height, width)
    return win_h[:, None] * win_w[None, :]


def _create_raised_cosine_lowpass_filter(
    image_shape: Tuple[int, int],
    cutoff_frequency: float = 0.35,
    rolloff_width: float = 0.08,
    dtype=np.float32,
) -> np.ndarray:
    """
    Create filter using open grids to reduce memory usage.
    """
    height, width = image_shape

    f_stop = cutoff_frequency
    f_pass = max(0.0, f_stop - rolloff_width)

    # Generate shifted frequency coordinates directly (-0.5 to 0.5)
    # explicit fftshift logic applied to the axes generation
    fy = np.fft.fftshift(np.fft.fftfreq(height)).astype(dtype)
    fx = np.fft.fftshift(np.fft.fftfreq(width)).astype(dtype)

    # Use broadcasting to calculate radial frequency without full meshgrid
    # fy[:, None] is (H, 1), fx[None, :] is (1, W) -> result is (H, W)
    radial_freq = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Initialize filter
    lowpass_filter = np.ones(image_shape, dtype=dtype)

    # Apply stopband
    lowpass_filter[radial_freq >= f_stop] = 0.0

    # Apply transition
    mask = (radial_freq > f_pass) & (radial_freq < f_stop)
    if np.any(mask):
        # Normalize transition region to [0, 1]
        norm_freq = (radial_freq[mask] - f_pass) / (f_stop - f_pass)
        # Raised cosine decay: 1 -> 0
        lowpass_filter[mask] = 0.5 * (1 + np.cos(np.pi * norm_freq))

    return lowpass_filter


def _apply_reflective_padding(
    image: np.ndarray, padding_ratio: float = 0.125
) -> Tuple[np.ndarray, Tuple[slice, slice]]:
    """
    Same logic, added docstring for return type clarity.
    """
    height, width = image.shape
    pad_h = int(height * padding_ratio)
    pad_w = int(width * padding_ratio)

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    crop_slice = (slice(pad_h, pad_h + height), slice(pad_w, pad_w + width))

    return padded_image, crop_slice


def preprocess_dpc(
    dpc_image: np.ndarray,
    padding_ratio: float = 0.125,
    taper_ratio: float = 0.08,
    lowpass_cutoff: float = 0.35,
    lowpass_rolloff: float = 0.08,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Preprocesses DPC image with NaN handling and memory optimizations.

    Args:
        fill_value: Value to replace NaNs with (after mean subtraction).
    """
    if dpc_image.ndim != 2:
        raise ValueError(f"Input must be 2D, got shape {dpc_image.shape}")

    # 1. Robust DC Removal
    # Use nanmean to ignore dead pixels/mask during mean calculation
    mean_val = np.nanmean(dpc_image)
    dpc_zero_mean = dpc_image - mean_val

    # Critical: Replace NaNs with 0 (or other value) before FFT
    # Otherwise FFT output will be all NaNs
    dpc_zero_mean = np.nan_to_num(dpc_zero_mean, nan=fill_value)

    # 2. Reflective Padding
    padded_image, crop_slice = _apply_reflective_padding(
        dpc_zero_mean, padding_ratio=padding_ratio
    )

    # 3. Tapering
    # Generate window directly in the correct dtype
    taper_window = _create_cosine_edge_taper(
        padded_image.shape, taper_ratio=taper_ratio, dtype=padded_image.dtype
    )
    # In-place multiplication if possible to save memory
    padded_image *= taper_window

    # 4. Filter Generation
    lowpass_filter = _create_raised_cosine_lowpass_filter(
        padded_image.shape,
        cutoff_frequency=lowpass_cutoff,
        rolloff_width=lowpass_rolloff,
        dtype=padded_image.dtype,
    )

    # 5. Frequency Domain Filtering
    # fft2 -> shift is standard.
    # Note: scipy.fft handles shifts efficiently.
    freq_domain = fftshift(fft2(padded_image))

    # Apply filter
    freq_domain *= lowpass_filter

    # Inverse FFT
    # using ifftshift before ifft2 is mathematically correct to align phases
    filtered_image = np.real(ifft2(ifftshift(freq_domain)))

    # 6. Crop and Return
    # Ensure output matches input dtype
    return filtered_image[crop_slice].astype(dpc_image.dtype)
