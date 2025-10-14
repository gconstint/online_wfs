import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def _create_cosine_edge_taper(image_shape, taper_ratio=0.08, dtype=np.float32):
    """
    Create a cosine edge taper window for smooth edge transitions in Fourier processing.

    The window applies a cosine-based tapering at the edges of the image to minimize
    spectral leakage and ringing artifacts in FFT operations. The central region
    maintains a value of 1, while the edges smoothly transition to 0.

    Args:
        image_shape (tuple): Shape of the image as (height, width)
        taper_ratio (float): Width of taper region as fraction of image dimensions (default: 0.08)
        dtype (np.dtype): Output array data type (default: np.float32)

    Returns:
        np.ndarray: 2D taper window with dimensions matching image_shape
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
    Create a raised-cosine lowpass filter for frequency domain filtering.

    Generates a smooth 2D frequency response with controlled transition between
    passband and stopband to minimize Gibbs phenomena. The filter is radially
    symmetric and centered at zero frequency after fftshift.

    Args:
        image_shape (tuple): Shape of the image as (height, width)
        cutoff_frequency (float): Normalized cutoff frequency [0-0.5] relative to Nyquist
        rolloff_width (float): Normalized transition bandwidth [0-0.5] relative to Nyquist
        dtype (np.dtype): Output array data type (default: np.float32)

    Returns:
        np.ndarray: 2D frequency domain filter with dimensions matching image_shape
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
    Apply reflective padding to minimize FFT edge discontinuities.

    Extends the image boundaries using reflection, which preserves edge continuity
    and reduces artifacts in frequency domain processing. The padding size is
    proportional to the smaller image dimension.

    Args:
        image (np.ndarray): Input 2D image array
        padding_ratio (float): Padding width relative to min(height, width)

    Returns:
        tuple: (padded_image, crop_slice), where:
            - padded_image (np.ndarray): Image with reflective padding
            - crop_slice (tuple): Slice objects to recover original dimensions
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
    Advanced DPC (Differential Phase Contrast) image preprocessing pipeline.

    Applies a sequence of optimized preprocessing steps to enhance DPC image quality:
    1. DC bias removal to normalize the signal baseline
    2. Reflective padding to minimize FFT edge artifacts
    3. Edge tapering for smooth boundary transitions
    4. Frequency domain lowpass filtering for noise reduction
    5. Restoration of original image dimensions

    Parameters are optimized for X-ray grating interferometry but can be adjusted
    for specific experimental conditions.

    Args:
        dpc_image (np.ndarray): Raw DPC image data (2D array)
        padding_ratio (float): Padding width relative to min(height, width)
        taper_ratio (float): Edge taper width relative to image dimensions
        lowpass_cutoff (float): Normalized cutoff frequency [0-0.5]
        lowpass_rolloff (float): Normalized transition bandwidth [0-0.5]

    Returns:
        np.ndarray: Preprocessed DPC image with preserved dimensions and dtype

    Notes:
        - Preserves low-frequency phase information including spherical aberrations
        - Reduces high-frequency noise while maintaining edge fidelity
        - Memory-efficient implementation with in-place operations where possible
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
