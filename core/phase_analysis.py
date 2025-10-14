import os
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def _reflect_and_pad_gradient_fields(delta_x, delta_y):
    """
    Reflect and pad gradient fields to generate continuous 2D reflection functions at boundaries.
    Preallocate target arrays and fill four quadrants at once to avoid copy overhead from multiple concatenations.
    """
    delta_x = np.asarray(delta_x)
    delta_y = np.asarray(delta_y)
    N, M = delta_x.shape

    # Preallocate output (2N x 2M)
    out_x = np.empty((2*N, 2*M), dtype=delta_x.dtype)
    out_y = np.empty((2*N, 2*M), dtype=delta_y.dtype)

    # Fill four quadrants for delta_x
    out_x[:N, :M] = delta_x
    out_x[N:2*N, :M] = delta_x[::-1, :]
    out_x[:N, M:2*M] = -delta_x[:, ::-1]
    out_x[N:2*N, M:2*M] = -delta_x[::-1, ::-1]

    # Fill four quadrants for delta_y (note: signs consistent with original implementation)
    out_y[:N, :M] = delta_y
    out_y[N:2*N, :M] = -delta_y[::-1, :]
    out_y[:N, M:2*M] = delta_y[:, ::-1]
    out_y[N:2*N, M:2*M] = -delta_y[::-1, ::-1]

    return out_x, out_y


def fc_method(delta_x, delta_y, reflected_pad=True, remove_tilt=True):
    """
    Improved Frankot-Chellappa algorithm for 2D phase integration.

    Integrates data using differential phases in x and y directions to reconstruct 2D phase distribution.
    Improved handling of zero frequency components to reduce baseline drift.

    Parameters:
        delta_x (ndarray): Differential phase in x direction, in radians
        delta_y (ndarray): Differential phase in y direction, in radians
        reflected_pad (bool): If True, applies reflection padding to input data to reduce boundary artifacts
        ensure_concave (bool): If True, ensures the phase is concave (concave facing upward)

    Returns:
        ndarray: 2D phase reconstructed from DPC data
    """
    # If reflection padding is needed, apply padding to delta_x and delta_y
    if reflected_pad:
        delta_x, delta_y = _reflect_and_pad_gradient_fields(delta_x, delta_y)

    # Get the size of the padded data
    NN, MM = delta_x.shape

    # Maintain smaller dtype and contiguous memory to reduce bandwidth
    delta_x = np.asarray(delta_x, dtype=np.float32, order='C')
    delta_y = np.asarray(delta_y, dtype=np.float32, order='C')

    # Frequency vectors (avoid large arrays and extra allocations from meshgrid)
    kx = (fftfreq(MM) * (2 * np.pi)).astype(np.float32)
    ky = (fftfreq(NN) * (2 * np.pi)).astype(np.float32)
    kx2 = kx * kx
    ky2 = ky * ky

    # Compute Fourier transforms (multithreaded)
    fx = fft2(delta_x, workers=os.cpu_count())
    fy = fft2(delta_y, workers=os.cpu_count())

    # Handle zero frequency components
    fx[0, 0] = 0
    fy[0, 0] = 0

    # Compute numerator and denominator in Fourier domain (using broadcasting to avoid large meshgrid)
    numerator = -1j * (fx * kx[np.newaxis, :] + fy * ky[:, np.newaxis])
    denominator = ky2[:, np.newaxis] + kx2[np.newaxis, :]

    # Special handling for zero frequency point to avoid division by zero
    denominator[0, 0] = 1.0
    numerator[0, 0] = 0.0

    # Remove tilt component (corresponding to low-frequency part)
    if remove_tilt:
        max_freq2 = float(ky2.max() + kx2.max())
        numerator = _apply_smooth_tilt_filter(numerator, denominator, max_freq2)

    # Perform inverse Fourier transform to obtain phase
    phase_2D = np.real(ifft2(numerator / denominator, workers=os.cpu_count()))

    # Further remove tilt term in spatial domain (fast version)
    if remove_tilt:
        phase_2D = _remove_spatial_tilt_fast(phase_2D)

    # If padding was applied, return the central region after cropping
    if reflected_pad:
        phase_2D = _crop_to_central_quarter(phase_2D)

    return phase_2D


def _apply_smooth_tilt_filter(numerator, denominator, max_freq2):
    """
    Smoothly removes very low-frequency tilt in the frequency domain to avoid explicit sqrt and meshgrid.
    numerator: complex array (H,W)
    denominator: wide sense frequency square wx^2+wy^2 (H,W) floating-point array
    max_freq2: max(wx^2+wy^2)
    """
    # Construct smooth filter: exp(- denom / ((max_freq*0.05)^2))
    # Equivalent to exp(- denominator / (max_freq2 * 0.05^2))
    eps = np.float32(0.05)
    cutoff = np.float32(0.02)
    denom = denominator.astype(np.float32, copy=False)
    scale = np.float32(max_freq2) * (eps * eps)
    filt = np.exp(- denom / (scale + 1e-20))
    # Completely remove the lowest frequency: denom < (max_freq*0.02)^2
    low_th = np.float32(max_freq2) * (cutoff * cutoff)
    filt = np.where(denom < low_th, 0.0, filt).astype(np.float32, copy=False)
    return numerator * filt


def _remove_spatial_tilt_fast(phase):
    """
    Quickly removes linear tilt (a + b*x + c*y) using least squares closed-form solution,
    avoiding the construction of a large design matrix A (H*W x 3).
    Numerically similar to _remove_spatial_tilt, but significantly faster and more memory-efficient.
    """
    H, W = phase.shape
    # Normalize coordinates to [-1,1]
    x = (np.arange(W, dtype=np.float32) - (W / 2)) * (2.0 / W)
    y = (np.arange(H, dtype=np.float32) - (H / 2)) * (2.0 / H)

    N = float(H * W)
    sum_x = float(x.sum()) * H
    sum_y = float(y.sum()) * W
    sum_xx = float((x * x).sum()) * H
    sum_yy = float((y * y).sum()) * W
    sum_xy = float(x.sum()) * float(y.sum())

    z = phase.astype(np.float64, copy=False)
    sum_z = float(z.sum())
    sum_xz = float(x.dot(z.sum(axis=0)))
    sum_yz = float(y.dot(z.sum(axis=1)))

    # Normal equations 3x3
    A = np.array([[N,     sum_x, sum_y],
                  [sum_x, sum_xx, sum_xy],
                  [sum_y, sum_xy, sum_yy]], dtype=np.float64)
    b = np.array([sum_z, sum_xz, sum_yz], dtype=np.float64)
    coeffs = np.linalg.solve(A, b)

    # Construct and remove plane (broadcasting, avoid large grid)
    plane = coeffs[0] + coeffs[1] * x[np.newaxis, :] + coeffs[2] * y[:, np.newaxis]
    return (z - plane).astype(phase.dtype, copy=False)



def _remove_spatial_tilt(phase):
    """
    Remove phase tilt (linear gradient) in spatial domain.

    Parameters:
        phase (ndarray): 2D phase map

    Returns:
        ndarray: Phase map with tilt removed
    """
    # Create coordinate grid
    height, width = phase.shape
    y, x = np.mgrid[0:height, 0:width]

    # Normalize coordinates to [-1, 1] range
    x_norm = 2 * (x - width / 2) / width
    y_norm = 2 * (y - height / 2) / height

    # Build linear term matrix [1, x, y]
    A = np.column_stack([
        np.ones(height * width),
        x_norm.flatten(),
        y_norm.flatten()
    ])

    # Solve least squares problem: phase = A * coeffs
    phase_flat = phase.flatten()
    coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_flat, rcond=None)

    # Reconstruct tilt term
    tilt_phase = A @ coeffs
    tilt_phase = tilt_phase.reshape(phase.shape)

    # Subtract tilt term from original phase
    phase_corrected = phase - tilt_phase

    return phase_corrected


def _crop_to_central_quarter(array):
    """
    Crop array to keep the central quarter region, removing padding.

    Parameters:
        array (ndarray): Array containing padded regions.

    Returns:
        ndarray: Cropped array containing the central quarter of the original data.
    """
    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]


def dpc_integration(dpc_x, dpc_y, remove_tilt=True):
    """
    DPC Integration Function

    Integrates differential phase contrast (DPC) measurements to reconstruct the phase.
    """

    phase = fc_method(dpc_x, dpc_y, reflected_pad=True, remove_tilt=remove_tilt)
    phase -= np.mean(phase)
    return -phase
