import os
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def _reflect_and_pad_gradient_fields(delta_x, delta_y):
    """
    反射并填充梯度场，以在边界处生成连续的二维反射函数。
    通过一次性预分配目标数组并填充四个象限，避免多次拼接带来的拷贝开销。
    """
    delta_x = np.asarray(delta_x)
    delta_y = np.asarray(delta_y)
    N, M = delta_x.shape

    # 预分配输出（2N x 2M）
    out_x = np.empty((2*N, 2*M), dtype=delta_x.dtype)
    out_y = np.empty((2*N, 2*M), dtype=delta_y.dtype)

    # 填充 delta_x 的四个象限
    out_x[:N, :M] = delta_x
    out_x[N:2*N, :M] = delta_x[::-1, :]
    out_x[:N, M:2*M] = -delta_x[:, ::-1]
    out_x[N:2*N, M:2*M] = -delta_x[::-1, ::-1]

    # 填充 delta_y 的四个象限（注意符号与原实现保持一致）
    out_y[:N, :M] = delta_y
    out_y[N:2*N, :M] = -delta_y[::-1, :]
    out_y[:N, M:2*M] = delta_y[:, ::-1]
    out_y[N:2*N, M:2*M] = -delta_y[::-1, ::-1]

    return out_x, out_y


def fc_method(delta_x, delta_y, reflected_pad=True, remove_tilt=True):
    """
    改进的Frankot-Chellappa算法用于二维相位积分。

    通过x和y方向上的微分相位对数据进行积分以重建二维相位分布。
    改进了零频率分量的处理以减少基线漂移。

    参数:
        delta_x (ndarray): x方向的微分相位，单位为弧度
        delta_y (ndarray): y方向的微分相位，单位为弧度
        reflected_pad (bool): 若为True，则对输入数据进行反射填充以减少边界伪影
        ensure_concave (bool): 若为True，则确保相位为下凹形状（凹面朝上）

    返回:
        ndarray: 从DPC数据重建的二维相位
    """
    # 如果需要反射填充，则对delta_x和delta_y进行填充处理
    if reflected_pad:
        delta_x, delta_y = _reflect_and_pad_gradient_fields(delta_x, delta_y)

    # 获取填充后的数据大小
    NN, MM = delta_x.shape

    # 保持较小dtype和连续内存，降低带宽
    delta_x = np.asarray(delta_x, dtype=np.float32, order='C')
    delta_y = np.asarray(delta_y, dtype=np.float32, order='C')

    # 频率向量（避免 meshgrid 带来的大数组与额外分配）
    kx = (fftfreq(MM) * (2 * np.pi)).astype(np.float32)
    ky = (fftfreq(NN) * (2 * np.pi)).astype(np.float32)
    kx2 = kx * kx
    ky2 = ky * ky

    # 计算傅里叶变换（多线程）
    fx = fft2(delta_x, workers=os.cpu_count())
    fy = fft2(delta_y, workers=os.cpu_count())

    # 处理零频率分量
    fx[0, 0] = 0
    fy[0, 0] = 0

    # 计算傅里叶域中的分子和分母（使用广播，避免大 meshgrid）
    numerator = -1j * (fx * kx[np.newaxis, :] + fy * ky[:, np.newaxis])
    denominator = ky2[:, np.newaxis] + kx2[np.newaxis, :]

    # 特殊处理零频率点，避免除以零
    denominator[0, 0] = 1.0
    numerator[0, 0] = 0.0

    # 移除倾斜分量 (对应于低频部分)
    if remove_tilt:
        max_freq2 = float(ky2.max() + kx2.max())
        numerator = _apply_smooth_tilt_filter(numerator, denominator, max_freq2)

    # 进行傅里叶逆变换以得到相位
    phase_2D = np.real(ifft2(numerator / denominator, workers=os.cpu_count()))

    # 在空间域中进一步去除倾斜项（快速版）
    if remove_tilt:
        phase_2D = _remove_spatial_tilt_fast(phase_2D)

    # 如果进行了填充，返回去除填充的中心区域
    if reflected_pad:
        phase_2D = _crop_to_central_quarter(phase_2D)

    return phase_2D


def _apply_smooth_tilt_filter(numerator, denominator, max_freq2):
    """
    在频域中平滑移除极低频倾斜项，避免显式 sqrt 与 meshgrid。
    numerator: 复数数组 (H,W)
    denominator: 宽义的频率平方 wx^2+wy^2 (H,W) 浮点数组
    max_freq2: max(wx^2+wy^2)
    """
    # 构造平滑滤波器：exp(- denom / ((max_freq*0.05)^2))
    # 等价于 exp(- denominator / (max_freq2 * 0.05^2))
    eps = np.float32(0.05)
    cutoff = np.float32(0.02)
    denom = denominator.astype(np.float32, copy=False)
    scale = np.float32(max_freq2) * (eps * eps)
    filt = np.exp(- denom / (scale + 1e-20))
    # 完全移除最低频率：denom < (max_freq*0.02)^2
    low_th = np.float32(max_freq2) * (cutoff * cutoff)
    filt = np.where(denom < low_th, 0.0, filt).astype(np.float32, copy=False)
    return numerator * filt


def _remove_spatial_tilt_fast(phase):
    """
    以最小二乘的闭式求解快速去除线性倾斜项（a + b*x + c*y），
    避免构造大设计矩阵 A（H*W x 3）。
    与 _remove_spatial_tilt 数值接近，但显著更快更省内存。
    """
    H, W = phase.shape
    # 归一化坐标到[-1,1]
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

    # 正规方程 3x3
    A = np.array([[N,     sum_x, sum_y],
                  [sum_x, sum_xx, sum_xy],
                  [sum_y, sum_xy, sum_yy]], dtype=np.float64)
    b = np.array([sum_z, sum_xz, sum_yz], dtype=np.float64)
    coeffs = np.linalg.solve(A, b)

    # 构建并移除平面（广播，避免大网格）
    plane = coeffs[0] + coeffs[1] * x[np.newaxis, :] + coeffs[2] * y[:, np.newaxis]
    return (z - plane).astype(phase.dtype, copy=False)



def _remove_spatial_tilt(phase):
    """
    在空间域中去除相位的倾斜项（线性梯度）

    参数:
        phase (ndarray): 2D相位图

    返回:
        ndarray: 去除倾斜项后的相位图
    """
    # 创建坐标网格
    height, width = phase.shape
    y, x = np.mgrid[0:height, 0:width]

    # 将坐标归一化到[-1, 1]范围
    x_norm = 2 * (x - width / 2) / width
    y_norm = 2 * (y - height / 2) / height

    # 构建线性项矩阵 [1, x, y]
    A = np.column_stack([
        np.ones(height * width),
        x_norm.flatten(),
        y_norm.flatten()
    ])

    # 求解最小二乘问题：phase = A * coeffs
    phase_flat = phase.flatten()
    coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_flat, rcond=None)

    # 重建倾斜项
    tilt_phase = A @ coeffs
    tilt_phase = tilt_phase.reshape(phase.shape)

    # 从原始相位中减去倾斜项
    phase_corrected = phase - tilt_phase

    return phase_corrected


def _crop_to_central_quarter(array):
    """
    裁剪数组，保留中心四分之一区域，以去除填充部分。

    参数:
        array (ndarray): 包含填充部分的数组。

    返回:
        ndarray: 裁剪后的数组，包含原数据的中心四分之一。
    """
    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]


def dpc_integration(dpc_x, dpc_y, remove_tilt=True):
    """
    DPC积分函数

    """

    phase = fc_method(dpc_x, dpc_y, reflected_pad=True, remove_tilt=remove_tilt)
    phase -= np.mean(phase)
    return -phase
