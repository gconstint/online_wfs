import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def _reflect_and_pad_gradient_fields(delta_x, delta_y):
    """
    反射并填充梯度场 (保持物理符号正确性)
    """
    # x 方向的填充
    delta_x_c1 = np.concatenate((delta_x, delta_x[::-1, :]), axis=0)
    delta_x_c2 = np.concatenate((-delta_x[:, ::-1], -delta_x[::-1, ::-1]), axis=0)
    delta_x = np.concatenate((delta_x_c1, delta_x_c2), axis=1)

    # y 方向的填充
    delta_y_c1 = np.concatenate((delta_y, -delta_y[::-1, :]), axis=0)
    delta_y_c2 = np.concatenate((delta_y[:, ::-1], -delta_y[::-1, ::-1]), axis=0)
    delta_y = np.concatenate((delta_y_c1, delta_y_c2), axis=1)

    return delta_x, delta_y


def _crop_to_central_quarter(array):
    """裁剪回左上角原始区域"""
    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]


def _ensure_concave_shape(phase):
    """
    强制相位为下凹形状（碗状，Concave Up）。
    通过对中心截线进行抛物线拟合来判断曲率方向，这种方法对倾斜（Tilt）不敏感，非常稳健。
    """
    H, W = phase.shape

    # 提取中心行和中心列
    mid_row = phase[H // 2, :]
    mid_col = phase[:, W // 2]

    # 定义坐标轴
    x = np.arange(W)
    y = np.arange(H)

    # 使用 polyfit 进行快速二次拟合 (y = ax^2 + bx + c)
    # 我们只关心二次项系数 a
    # 如果 a > 0，开口向上 (凹面/Bowl)
    # 如果 a < 0，开口向下 (凸面/Hill)
    coeff_x = np.polyfit(x, mid_row, 2)
    coeff_y = np.polyfit(y, mid_col, 2)

    curvature_x = coeff_x[0]
    curvature_y = coeff_y[0]

    # 综合判断：如果两个方向的曲率之和小于0，说明整体是凸的，需要翻转
    if (curvature_x + curvature_y) < 0:
        return -phase

    return phase


def fc_method(delta_x, delta_y, reflected_pad=True):
    """
    Frankot-Chellappa 积分算法
    (已移除多余的去倾斜和滤波参数，保持核心纯净)
    """
    # 1. 反射填充
    if reflected_pad:
        gx, gy = _reflect_and_pad_gradient_fields(delta_x, delta_y)
    else:
        gx, gy = delta_x, delta_y

    rows, cols = gx.shape

    # 2. 频率网格
    wx = fftfreq(cols) * 2 * np.pi
    wy = fftfreq(rows) * 2 * np.pi
    wx_grid, wy_grid = np.meshgrid(wx, wy)

    # 3. FFT
    fx = fft2(gx)
    fy = fft2(gy)

    # 4. 频域积分
    numerator = -1j * (wx_grid * fx + wy_grid * fy)
    denominator = wx_grid**2 + wy_grid**2

    # 处理奇点
    denominator[0, 0] = 1.0
    numerator[0, 0] = 0.0

    # 5. IFFT
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
    """
    # 1. 执行积分
    phase = fc_method(dpc_x, dpc_y, reflected_pad=True)

    # 2. 去除直流分量
    phase -= np.mean(phase)

    # 3. 强制形状约束
    if ensure_concave:
        phase = _ensure_concave_shape(phase)

    return phase
