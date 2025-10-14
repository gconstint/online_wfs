import numpy as np


def two_steps_fresnel_method(E, wavelength, z, dx, dy, scale_factor_x=1, scale_factor_y=None):
    """
    Use the two-step Fresnel method to calculate the propagation of the light field, supporting different scale transformations in the X and Y directions
    
    Parameters:
    -----------
    E : ndarray
        Input light field
    wavelength : float
        Wavelength
    z : float
        Propagation distance
    dx : float
        Sampling interval in X direction
    dy : float
        Sampling interval in Y direction
    scale_factor_x : float, optional
        Scaling factor in X direction, default is 1
    scale_factor_y : float, optional
        Scaling factor in Y direction, if None, the value of scale_factor_x is used
    """
    
    # If the scaling factor in the Y direction is not specified, use the scaling factor in the X direction
    if scale_factor_y is None:
        scale_factor_y = scale_factor_x

    Nx = E.shape[0]
    Ny = E.shape[1]
    extent_x = Nx * dx
    extent_y = Ny * dy

    # Calculate the coordinate grid
    x = dx * (np.arange(Nx) - Nx//2)
    y = dy * (np.arange(Ny) - Ny//2)
    xx, yy = np.meshgrid(x, y)

    # Calculate the input and output plane sizes
    L1x = extent_x
    L1y = extent_y
    L2x = L1x * scale_factor_x
    L2y = L1y * scale_factor_y

    # Step 1: Spatial domain phase modulation and FFT
    phase1 = np.exp(1j * np.pi/(z * wavelength) * 
                    ((L1x-L2x)/L1x * xx**2 + (L1y-L2y)/L1y * yy**2))
    E1 = np.fft.fftshift(np.fft.fft2(E * phase1))

    # Calculate the frequency grid
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))
    fxx, fyy = np.meshgrid(fx, fy)

    # Step 2: Frequency domain phase modulation and IFFT
    phase2 = np.exp(-1j * np.pi * wavelength * z * 
                    (L1x/L2x * fxx**2 + L1y/L2y * fyy**2))
    E2 = np.fft.ifft2(np.fft.ifftshift(phase2 * E1))

    # Calculate the scaled coordinate grid
    xx_scaled = xx * scale_factor_x
    yy_scaled = yy * scale_factor_y

    # Final phase correction
    phase3 = np.sqrt(L1x * L1y / (L2x * L2y)) * np.exp(
        1j * 2*np.pi/wavelength * z -
        1j * np.pi/(z * wavelength) * 
        ((L1x-L2x)/L2x * xx_scaled**2 + (L1y-L2y)/L2y * yy_scaled**2)
    )

    # Calculate the final field
    E_final = E2 * phase3

    return E_final
