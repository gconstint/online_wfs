import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def _reflect_and_pad_gradient_fields(delta_x, delta_y):
    """
    反射并填充梯度场 (保持物理符号正确性)
    优化版：使用预分配数组和直接赋值
    """
    h, w = delta_x.shape
    h2, w2 = h * 2, w * 2

    # 预分配输出数组
    gx = np.empty((h2, w2), dtype=delta_x.dtype)
    gy = np.empty((h2, w2), dtype=delta_y.dtype)

    # x 方向梯度填充
    # 左上 (原始)
    gx[:h, :w] = delta_x
    # 左下 (y 翻转)
    gx[h:, :w] = delta_x[::-1, :]
    # 右上 (x 翻转, 取负)
    gx[:h, w:] = -delta_x[:, ::-1]
    # 右下 (xy 翻转, 取负)
    gx[h:, w:] = -delta_x[::-1, ::-1]

    # y 方向梯度填充
    # 左上 (原始)
    gy[:h, :w] = delta_y
    # 左下 (y 翻转, 取负)
    gy[h:, :w] = -delta_y[::-1, :]
    # 右上 (x 翻转)
    gy[:h, w:] = delta_y[:, ::-1]
    # 右下 (xy 翻转, 取负)
    gy[h:, w:] = -delta_y[::-1, ::-1]

    return gx, gy


def _crop_to_central_quarter(array):
    """裁剪回左上角原始区域 (使用切片，更高效)"""
    h, w = array.shape
    return array[: h // 2, : w // 2]


def _ensure_concave_shape(phase):
    """
    强制相位为下凹形状（碗状，Concave Up）。
    通过对中心截线进行抛物线拟合来判断曲率方向。
    """
    H, W = phase.shape

    # 提取中心行和中心列
    mid_row = phase[H // 2, :]
    mid_col = phase[:, W // 2]

    # 使用 polyfit 进行快速二次拟合 (y = ax^2 + bx + c)
    # 只关心二次项系数 a
    x = np.arange(W)
    y = np.arange(H)

    coeff_x = np.polyfit(x, mid_row, 2)
    coeff_y = np.polyfit(y, mid_col, 2)

    # 综合判断：如果两个方向的曲率之和小于0，需要翻转
    if (coeff_x[0] + coeff_y[0]) < 0:
        return -phase

    return phase


def fc_method(delta_x, delta_y, reflected_pad=True):
    """
    Frankot-Chellappa 积分算法 (优化版)
    使用 scipy.fft 和 broadcasting 提升性能
    """
    # 1. 反射填充
    if reflected_pad:
        gx, gy = _reflect_and_pad_gradient_fields(delta_x, delta_y)
    else:
        gx, gy = delta_x, delta_y

    rows, cols = gx.shape

    # 2. 频率网格 (使用 scipy.fft.fftfreq 和 broadcasting)
    wx = fftfreq(cols) * (2 * np.pi)  # 1D array (cols,)
    wy = fftfreq(rows) * (2 * np.pi)  # 1D array (rows,)

    # 3. FFT
    fx = fft2(gx)
    fy = fft2(gy)

    # 4. 频域积分 (使用 broadcasting 代替 meshgrid)
    # wx[None, :] -> (1, cols), wy[:, None] -> (rows, 1)
    numerator = -1j * (wx[None, :] * fx + wy[:, None] * fy)
    denominator = wx[None, :] ** 2 + wy[:, None] ** 2

    # 处理奇点 (DC 分量)
    denominator[0, 0] = 1.0
    numerator[0, 0] = 0.0

    # 5. IFFT 并取实部
    phase_2D = np.real(ifft2(numerator / denominator))

    # 6. 裁剪
    if reflected_pad:
        phase_2D = _crop_to_central_quarter(phase_2D)

    return phase_2D


def dpc_integration(dpc_x, dpc_y, ensure_concave=True):
    """
    DPC 积分主函数

    Args:
        dpc_x, dpc_y: 梯度图
        ensure_concave (bool): 是否强制输出为下凹（碗状）相位。
                               对于会聚光束(Converging)，相位通常是下凹的。

    Returns:
        ndarray: 积分重建的相位图
    """
    # 1. 执行积分
    phase = fc_method(dpc_x, dpc_y, reflected_pad=True)

    # 2. 去除直流分量
    phase -= np.mean(phase)

    # 3. 强制形状约束
    if ensure_concave:
        phase = _ensure_concave_shape(phase)

    return phase
