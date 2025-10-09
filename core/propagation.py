import numpy as np


def two_steps_fresnel_method(E, wavelength, z, dx, dy, scale_factor_x=1, scale_factor_y=None):
    """
    使用两步菲涅尔法计算光场传播，支持X和Y方向不同的尺度变换
    
    Parameters:
    -----------
    p_energy : ndarray
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
    """
    
    # 如果未指定Y方向缩放因子，使用X方向的缩放因子
    if scale_factor_y is None:
        scale_factor_y = scale_factor_x

    Nx = E.shape[0]
    Ny = E.shape[1]
    extent_x = Nx * dx
    extent_y = Ny * dy

    # 计算坐标网格
    x = dx * (np.arange(Nx) - Nx//2)
    y = dy * (np.arange(Ny) - Ny//2)
    xx, yy = np.meshgrid(x, y)

    # 计算输入输出平面尺寸
    L1x = extent_x
    L1y = extent_y
    L2x = L1x * scale_factor_x
    L2y = L1y * scale_factor_y

    # 第一步：空间域相位调制和FFT
    phase1 = np.exp(1j * np.pi/(z * wavelength) * 
                    ((L1x-L2x)/L1x * xx**2 + (L1y-L2y)/L1y * yy**2))
    E1 = np.fft.fftshift(np.fft.fft2(E * phase1))

    # 计算频率网格
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))
    fxx, fyy = np.meshgrid(fx, fy)

    # 第二步：频域相位调制和IFFT
    phase2 = np.exp(-1j * np.pi * wavelength * z * 
                    (L1x/L2x * fxx**2 + L1y/L2y * fyy**2))
    E2 = np.fft.ifft2(np.fft.ifftshift(phase2 * E1))

    # 计算缩放后的坐标网格
    xx_scaled = xx * scale_factor_x
    yy_scaled = yy * scale_factor_y

    # 最终相位修正
    phase3 = np.sqrt(L1x * L1y / (L2x * L2y)) * np.exp(
        1j * 2*np.pi/wavelength * z -
        1j * np.pi/(z * wavelength) * 
        ((L1x-L2x)/L2x * xx_scaled**2 + (L1y-L2y)/L2y * yy_scaled**2)
    )

    # 计算最终场
    E_final = E2 * phase3

    return E_final
