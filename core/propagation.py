import numpy as np
from os import cpu_count
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

# Module-level constant for parallel FFT
_CPU_COUNT = cpu_count() or 4


def two_steps_fresnel_method(
    E, wavelength, z, dx, dy, scale_factor_x=1, scale_factor_y=None
):
    """
    Calculate light field propagation using two-step Fresnel method, supporting different scaling in X and Y directions.

    Parameters:
    -----------
    E : ndarray
        Input light field
    wavelength : float
        Wavelength
    z : float
        Propagation distance
    dx : float
        X-direction sampling interval
    dy : float
        Y-direction sampling interval
    scale_factor_x : float, optional
        X-direction scaling factor, default is 1
    scale_factor_y : float, optional
        Y-direction scaling factor, if None uses scale_factor_x value

    Returns:
    --------
    ndarray
        Propagated light field
    """
    # If Y-direction scale factor not specified, use X-direction scale factor
    if scale_factor_y is None:
        scale_factor_y = scale_factor_x

    Nx, Ny = E.shape
    L1x = Nx * dx
    L1y = Ny * dy
    L2x = L1x * scale_factor_x
    L2y = L1y * scale_factor_y

    # Pre-compute constants
    inv_z_wavelength = 1.0 / (z * wavelength)
    pi_inv_z_wavelength = np.pi * inv_z_wavelength

    # Use 1D arrays + broadcasting instead of meshgrid (saves memory)
    x = dx * (np.arange(Nx, dtype=np.float64) - Nx // 2)
    y = dy * (np.arange(Ny, dtype=np.float64) - Ny // 2)

    # Step 1: Spatial domain phase modulation and FFT
    # Use broadcasting: x[:, None] * y[None, :] is equivalent to meshgrid
    coeff_x1 = (L1x - L2x) / L1x
    coeff_y1 = (L1y - L2y) / L1y
    phase1 = np.exp(
        1j
        * pi_inv_z_wavelength
        * (coeff_x1 * x[:, None] ** 2 + coeff_y1 * y[None, :] ** 2)
    )

    # FFT with multi-threading
    E1 = fftshift(fft2(E * phase1, workers=_CPU_COUNT))

    # Calculate frequency grid (using broadcasting)
    fx = fftshift(fftfreq(Nx, dx))
    fy = fftshift(fftfreq(Ny, dy))

    # Step 2: Frequency domain phase modulation and IFFT
    coeff_fx = L1x / L2x
    coeff_fy = L1y / L2y
    phase2 = np.exp(
        -1j
        * np.pi
        * wavelength
        * z
        * (coeff_fx * fx[:, None] ** 2 + coeff_fy * fy[None, :] ** 2)
    )
    # IFFT with multi-threading
    E2 = ifft2(ifftshift(phase2 * E1), workers=_CPU_COUNT)

    # Final phase correction
    # Pre-compute scaled coordinates
    x_scaled = x * scale_factor_x
    y_scaled = y * scale_factor_y

    amplitude_factor = np.sqrt(L1x * L1y / (L2x * L2y))
    propagation_phase = np.exp(1j * 2 * np.pi / wavelength * z)

    coeff_x3 = (L1x - L2x) / L2x
    coeff_y3 = (L1y - L2y) / L2y
    phase3_spatial = np.exp(
        -1j
        * pi_inv_z_wavelength
        * (coeff_x3 * x_scaled[:, None] ** 2 + coeff_y3 * y_scaled[None, :] ** 2)
    )

    # Combine final field calculation
    E_final = E2 * (amplitude_factor * propagation_phase * phase3_spatial)

    return E_final
