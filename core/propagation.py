import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq


def two_steps_fresnel_method(
    E, wavelength, z, dx, dy, scale_factor_x=1, scale_factor_y=None
):
    """
    使用两步菲涅尔法计算光场传播，支持X和Y方向不同的尺度变换

    Parameters:
    -----------
    E : ndarray
        输入光场
    wavelength : float
        波长
    z : float
        传播距离
    dx : float
        X方向采样间隔
    dy : float
        Y方向采样间隔
    scale_factor_x : float, optional
        X方向缩放因子，默认为1
    scale_factor_y : float, optional
        Y方向缩放因子，如果为None则使用scale_factor_x的值

    Returns:
    --------
    ndarray
        传播后的光场
    """
    # 如果未指定Y方向缩放因子，使用X方向的缩放因子
    if scale_factor_y is None:
        scale_factor_y = scale_factor_x

    Nx, Ny = E.shape
    L1x = Nx * dx
    L1y = Ny * dy
    L2x = L1x * scale_factor_x
    L2y = L1y * scale_factor_y

    # 预计算常量
    inv_z_wavelength = 1.0 / (z * wavelength)
    pi_inv_z_wavelength = np.pi * inv_z_wavelength

    # 使用 1D 数组 + broadcasting 代替 meshgrid (节省内存)
    x = dx * (np.arange(Nx, dtype=np.float64) - Nx // 2)
    y = dy * (np.arange(Ny, dtype=np.float64) - Ny // 2)

    # 第一步：空间域相位调制和FFT
    # 使用 broadcasting: x[:, None] * y[None, :] 等效于 meshgrid
    coeff_x1 = (L1x - L2x) / L1x
    coeff_y1 = (L1y - L2y) / L1y
    phase1 = np.exp(
        1j
        * pi_inv_z_wavelength
        * (coeff_x1 * x[:, None] ** 2 + coeff_y1 * y[None, :] ** 2)
    )

    # 使用 scipy.fft (通常比 numpy.fft 快)
    E1 = fftshift(fft2(E * phase1))

    # 计算频率网格 (使用 broadcasting)
    fx = fftshift(fftfreq(Nx, dx))
    fy = fftshift(fftfreq(Ny, dy))

    # 第二步：频域相位调制和IFFT
    coeff_fx = L1x / L2x
    coeff_fy = L1y / L2y
    phase2 = np.exp(
        -1j
        * np.pi
        * wavelength
        * z
        * (coeff_fx * fx[:, None] ** 2 + coeff_fy * fy[None, :] ** 2)
    )
    E2 = ifft2(ifftshift(phase2 * E1))

    # 最终相位修正
    # 预计算缩放后的坐标
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

    # 合并最终场计算
    E_final = E2 * (amplitude_factor * propagation_phase * phase3_spatial)

    return E_final
