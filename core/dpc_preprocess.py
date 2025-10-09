import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def _create_cosine_edge_taper(image_shape, taper_ratio=0.08, dtype=np.float32):
    """
    Create a cosine edge taper window for smooth edge transitions.

    The window maintains value 1 in the center and smoothly tapers to 0 at edges
    using a cosine function to minimize FFT artifacts.

    Args:
        image_shape (tuple): Shape of the image (height, width)
        taper_ratio (float): Fraction of image dimensions to use for tapering
        dtype (np.dtype): Data type for the output array

    Returns:
        np.ndarray: 2D taper window with same shape as input
    """
    height, width = image_shape

    # Calculate taper widths (minimum 1 pixel)
    taper_height = max(1, int(height * taper_ratio))
    taper_width = max(1, int(width * taper_ratio))

    # Initialize 1D windows
    vertical_window = np.ones(height, dtype=dtype)
    horizontal_window = np.ones(width, dtype=dtype)

    # Create cosine transitions
    cosine_vertical = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_height, dtype=dtype)))
    cosine_horizontal = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_width, dtype=dtype)))

    # Apply tapering to edges
    vertical_window[:taper_height] = cosine_vertical
    vertical_window[-taper_height:] = cosine_vertical[::-1]
    horizontal_window[:taper_width] = cosine_horizontal
    horizontal_window[-taper_width:] = cosine_horizontal[::-1]

    # Create 2D window by outer product
    return vertical_window[:, None] * horizontal_window[None, :]

def _create_raised_cosine_lowpass_filter(image_shape, cutoff_frequency=0.35,
                                        rolloff_width=0.08, dtype=np.float32):
    """
    Create a raised-cosine lowpass filter for smooth frequency domain filtering.

    The filter has a flat passband, smooth rolloff transition, and complete stopband
    to minimize ringing artifacts in the filtered image.

    Args:
        image_shape (tuple): Shape of the image (height, width)
        cutoff_frequency (float): Cutoff frequency as fraction of Nyquist (0-0.5)
        rolloff_width (float): Width of transition band as fraction of Nyquist
        dtype (np.dtype): Data type for the output array

    Returns:
        np.ndarray: 2D frequency domain filter with same shape as input
    """
    height, width = image_shape
    nyquist_frequency = min(height, width) / 2.0

    # Calculate frequency thresholds
    stopband_frequency = float(cutoff_frequency) * nyquist_frequency
    passband_frequency = max(0.0, stopband_frequency - float(rolloff_width) * nyquist_frequency)

    # Create frequency coordinate grids
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    radial_frequency = np.hypot(x_coords - center_x, y_coords - center_y).astype(dtype)

    # Initialize filter (all pass)
    lowpass_filter = np.ones_like(radial_frequency, dtype=dtype)

    # Apply stopband (complete attenuation)
    lowpass_filter[radial_frequency >= stopband_frequency] = 0.0

    # Apply smooth transition in rolloff region
    transition_mask = (radial_frequency > passband_frequency) & (radial_frequency < stopband_frequency)
    if np.any(transition_mask):
        transition_values = radial_frequency[transition_mask]
        frequency_range = stopband_frequency - passband_frequency
        normalized_transition = (transition_values - passband_frequency) / frequency_range
        lowpass_filter[transition_mask] = 0.5 * (1 + np.cos(np.pi * normalized_transition))

    return lowpass_filter

def _apply_reflective_padding(image, padding_ratio=0.125):
    """
    Apply reflective padding to reduce FFT edge artifacts.

    Reflective padding mirrors the image content at boundaries, which helps
    minimize discontinuities that cause spectral leakage in FFT processing.

    Args:
        image (np.ndarray): Input 2D image array
        padding_ratio (float): Padding size as fraction of minimum image dimension

    Returns:
        tuple: (padded_image, crop_slice) where crop_slice can be used to
               extract the original region from the padded image
    """
    height, width = image.shape
    padding_size = int(min(height, width) * padding_ratio)

    # Apply symmetric padding on all sides
    padded_image = np.pad(image,
                         ((padding_size, padding_size), (padding_size, padding_size)),
                         mode='reflect')

    # Create slice objects for cropping back to original size
    crop_slice = (slice(padding_size, padding_size + height),
                  slice(padding_size, padding_size + width))

    return padded_image, crop_slice

def preprocess_dpc(
    dpc_image,
    padding_ratio=0.125,
    taper_ratio=0.08,
    lowpass_cutoff=0.35,
    lowpass_rolloff=0.08,
):
    """
    Comprehensive DPC (Differential Phase Contrast) image preprocessing pipeline.

    This function applies a series of preprocessing steps to reduce noise and artifacts
    while preserving important low-frequency information such as spherical aberrations:
    1. Remove DC bias (global mean)
    2. Apply reflective padding to minimize FFT edge effects
    3. Apply cosine edge tapering for smooth transitions
    4. Apply lowpass filtering in frequency domain
    5. Crop back to original dimensions

    The default parameters are optimized for typical X-ray grating self-images
    and generally do not require adjustment.

    Args:
        dpc_image (np.ndarray): Input DPC image (2D array)
        padding_ratio (float): Reflective padding size as fraction of minimum dimension
        taper_ratio (float): Edge taper width as fraction of image dimensions
        lowpass_cutoff (float): Lowpass filter cutoff frequency (0-0.5 relative to Nyquist)
        lowpass_rolloff (float): Lowpass filter rolloff width (relative to Nyquist)

    Returns:
        np.ndarray: Preprocessed DPC image with same shape and dtype as input

    Note:
        This preprocessing preserves low-frequency content including spherical
        aberrations and other important phase information while reducing high-frequency
        noise and artifacts.
    """
    # Remove DC bias to center the data around zero
    dpc_zero_mean = dpc_image - np.nanmean(dpc_image)

    # Apply reflective padding to reduce FFT boundary artifacts
    padded_image, original_crop_slice = _apply_reflective_padding(
        dpc_zero_mean, padding_ratio=padding_ratio
    )

    # Create and apply edge taper window for smooth transitions
    edge_taper_window = _create_cosine_edge_taper(
        padded_image.shape, taper_ratio=taper_ratio, dtype=dpc_image.dtype
    )
    tapered_image = padded_image * edge_taper_window

    # Create lowpass filter for noise reduction
    lowpass_filter = _create_raised_cosine_lowpass_filter(
        padded_image.shape,
        cutoff_frequency=lowpass_cutoff,
        rolloff_width=lowpass_rolloff,
        dtype=dpc_image.dtype
    )

    # Apply frequency domain filtering: FFT → filter → IFFT
    frequency_domain = fftshift(fft2(tapered_image))
    filtered_frequency = frequency_domain * lowpass_filter
    filtered_image = np.real(ifft2(ifftshift(filtered_frequency))).astype(
        dpc_image.dtype, copy=False
    )

    # Crop back to original image dimensions
    dpc_preprocessed = filtered_image[original_crop_slice]

    return dpc_preprocessed
