import cv2

import numpy as np

from numpy.fft import ifft2, ifftshift
from skimage.restoration import unwrap_phase


def calculate_peak_index(har_v, har_h, n_rows, n_columns, period_vert, period_hor):
    """
    Calculate the theoretical peak index(in pixel) for harmonic [har_v, har_h].
    Explain:
    1. n_rows // 2 和 n_columns // 2 is represented the center of the image.
    2. har_v, har_h is the harmonic index or the order of the harmonic.
    For instance, har_v = 1,har_v = 0 means the first harmonic in the vertical direction, and the zeroth harmonic in the
    horizontal direction.
    3. period_vert and period_hor is the period of the basic frequency in the harmonic.
    """
    results = [n_rows // 2 + har_v * period_vert,
               n_columns // 2 + har_h * period_hor]
    # print("the theoretical peak index is:", results)
    return results


def find_peak_in_region(img_fft, theoretical_idx, search_region):
    """
    Find the factual harmonic peak index (in pixel) within a local search window.

    Optimized: compute power only within the ROI to avoid full-spectrum |img_fft|.

    Args:
        img_fft: 2D complex spectrum (centered or not, irrelevant for peak search)
        theoretical_idx: [row, col] theoretical peak index
        search_region: int or [half_h, half_w] window half-size(s)
    """

    n_rows, n_columns = img_fft.shape
    i, j = theoretical_idx

    sy = int(search_region[0])
    sx = int(search_region[1])

    # Clamp ROI to image bounds
    y1 = max(0, i - sy)
    y2 = min(n_rows, i + sy + 1)
    x1 = max(0, j - sx)
    x2 = min(n_columns, j + sx + 1)

    local = img_fft[y1:y2, x1:x2]
    # Power spectrum (no sqrt): real^2 + imag^2
    power = (local.real * local.real) + (local.imag * local.imag)
    k = int(np.argmax(power))
    dy, dx = divmod(k, power.shape[1])
    return [y1 + dy, x1 + dx]


def accurate_harmonic_periods(img_fft, init_periods=None):
    """
    计算00、01和10谐波的峰值位置，并基于这些峰值计算准确的谐波周期。

    优化点：
    - 避免对整幅频谱计算 |img_fft|，仅在局部ROI内计算功率
    - 局部ROI搜索替代全图扫描，减少大矩阵分配与扫描

    Args:
        img_fft (ndarray): 频域图像（已FFT、是否shift不影响峰值局部搜索）
        init_periods (list): 初始谐波周期估计 [period_vert, period_hor]

    Returns:
        tuple: (
            accurate_period: [period_vert, period_hor] 准确的谐波周期,
            peak_positions: {'00': [y0, x0], '01': [y01, x01], '10': [y10, x10]} 各谐波峰值位置
        )
    """
    n_rows, n_columns = img_fft.shape

    period_vert, period_hor = init_periods

    # 设置搜索区域（若未传入，则基于周期估计自动设定）
    search_region = [max(10, int(min(period_vert, period_hor) // 4))] * 2

    # 3. 计算并精确定位三个谐波峰值
    peak_positions = {}

    # 3.1 计算00谐波（中心峰）
    idx_peak_00 = calculate_peak_index(0, 0, n_rows, n_columns, period_vert, period_hor)
    peak_00 = find_peak_in_region(img_fft, idx_peak_00, search_region)
    peak_positions['00'] = peak_00

    # 3.2 计算01谐波（水平方向第一谐波）
    idx_peak_01 = calculate_peak_index(0, 1, n_rows, n_columns, period_vert, period_hor)
    peak_01 = find_peak_in_region(img_fft, idx_peak_01, search_region)
    peak_positions['01'] = peak_01

    # 3.3 计算10谐波（垂直方向第一谐波）
    idx_peak_10 = calculate_peak_index(1, 0, n_rows, n_columns, period_vert, period_hor)
    peak_10 = find_peak_in_region(img_fft, idx_peak_10, search_region)
    peak_positions['10'] = peak_10

    # 4. 基于峰值位置计算准确的谐波周期
    exp_period_h = abs(peak_01[1] - peak_00[1])  # 水平周期
    exp_period_v = abs(peak_10[0] - peak_00[0])  # 垂直周期

    return [exp_period_v, exp_period_h], peak_positions


def rotate_image_by_peaks(img, peak_positions):
    """
    根据谐波峰值位置计算旋转角度并旋转图像

    Args:
        img (ndarray): 输入图像
        peak_positions (dict): 谐波峰值位置，包含 '00', '01', '10' 键

    Returns:
        ndarray: 旋转后的图像
        float: 旋转角度（度）
    """
    # 提取峰值位置
    peak_00 = peak_positions['00']  # [y, x]
    peak_01 = peak_positions['01']  # [y, x]
    peak_10 = peak_positions['10']  # [y, x]

    # 计算水平方向需要的旋转角度
    # 00峰应该与01峰具有相同的第一个坐标值（y坐标）
    delta_y_h = peak_01[0] - peak_00[0]  # 水平方向的y偏移
    delta_x_h = peak_01[1] - peak_00[1]  # 水平方向的x偏移
    angle_h = np.arctan2(delta_y_h, delta_x_h) * 180 / np.pi

    # 计算垂直方向需要的旋转角度
    # 00峰应该与10峰具有相同的第二个坐标值（x坐标）
    delta_y_v = peak_10[0] - peak_00[0]  # 垂直方向的y偏移
    delta_x_v = peak_10[1] - peak_00[1]  # 垂直方向的x偏移
    angle_v = np.arctan2(delta_y_v, delta_x_v) * 180 / np.pi - 90  # 垂直方向需要减去90度

    # 取两个方向角度的平均值
    angle = (angle_h + angle_v) / 2

    # 旋转图像（角度取反，因为我们要校正图像）
    rows, cols = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    return img_rotated, angle


def single_2D_grating_analyses(img_fft, harmonic_periods=None, unwrap_flag=True, plot_flag=True):
    """
    Process single 2D grating Talbot imaging img_data to extract harmonic images,
    dark field images, and phase contrast.

    Args:
        img (ndarray): Image to analyze.
        harmonic_periods (list): List containing harmonic periods [periodVert, periodHor].
        unwrap_flag (bool, optional): If True, unwrap phase images.
        plot_flag (bool, optional): If True, plot intermediate results.
        verbose (bool, optional): If True, print detailed processing information.

    Returns:
        int00 (ndarray): Zero-order harmonic intensity image (direct transmission).
        int01 (ndarray): First-order harmonic intensity image in the horizontal direction.
        int10 (ndarray): First-order harmonic intensity image in the vertical direction.
        darkField01 (ndarray): Dark field image for horizontal scattering.
        darkField10 (ndarray): Dark field image for vertical scattering.
        arg01 (ndarray): Phase contrast image in the horizontal direction.
        arg10 (ndarray): Phase contrast image in the vertical direction.
    """

    harm_img = single_grating_harmonic_images(img_fft, harmonic_periods)

    int00 = np.abs(harm_img[0])
    int01 = np.abs(harm_img[1])
    int10 = np.abs(harm_img[2])

    arg01 = unwrap_phase(np.angle(harm_img[1])) if unwrap_flag else np.angle(harm_img[1])
    arg10 = unwrap_phase(np.angle(harm_img[2])) if unwrap_flag else np.angle(harm_img[2])

    # remove piston only
    arg01 -= np.mean(arg01)
    arg10 -= np.mean(arg10)

    return int00, int01, int10, arg01, arg10


def single_grating_harmonic_images(img_fft, harmonic_period):
    """
    Obtain harmonic images from single 2D grating Talbot imaging.

    Args:
        img_fft (ndarray): Input image img_data.
        harmonic_period (list): List of harmonic periods [periodVert, periodHor].
        search_region (int, optional): Region size for harmonic peak search.
        plot_flag (bool, optional): If True, plot FFT results.
        verbose (bool, optional): If True, print detailed information.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Harmonic images for 00, 01, and 10 harmonics.
    """

    search_region = max(10, int(min(harmonic_period[0], harmonic_period[1]) // 4))  # 减小初始搜索范围
    # print("search_region", search_region)
    img_fft00 = extract_harmonic(img_fft, harmonic_period, harmonic_ij='00', search_region=search_region)
    img_fft01 = extract_harmonic(img_fft, harmonic_period, harmonic_ij='01', search_region=search_region)
    img_fft10 = extract_harmonic(img_fft, harmonic_period, harmonic_ij='10', search_region=search_region)

    img00 = ifft2(ifftshift(img_fft00), norm='ortho')
    img01 = ifft2(ifftshift(img_fft01), norm='ortho') if np.all(
        np.isfinite(img_fft01)) else img_fft01
    img10 = ifft2(ifftshift(img_fft10), norm='ortho') if np.all(
        np.isfinite(img_fft10)) else img_fft10

    return img00, img01, img10


def extract_harmonic(img_fft, harmonic_period, harmonic_ij='00', search_region=10):
    """
    优化版：Extract specific harmonic images from the Fourier-transformed img_data of a single grating Talbot image.

    改进：
    - 精确的亚像素峰值定位
    - 基于能量分布的自适应窗口
    - 改进的高斯加权
    - 修正窗口显示与实际选择区域的一致性
    - 支持矩形窗口（基于垂直和水平周期分别设置）
    """
    (n_rows, n_columns) = img_fft.shape
    har_v, har_h = int(harmonic_ij[0]), int(harmonic_ij[1])
    period_vert, period_hor = harmonic_period

    # 1. 获取理论峰值位置
    idx_peak_ij = calculate_peak_index(
        har_v, har_h, n_rows, n_columns, period_vert, period_hor)

    # 2. 精确定位峰值（使用局部最大值搜索）
    y0, x0 = idx_peak_ij
    y1, y2 = max(0, y0 - search_region), min(n_rows, y0 + search_region + 1)
    x1, x2 = max(0, x0 - search_region), min(n_columns, x0 + search_region + 1)

    # 在搜索区域内找到精确峰值
    local_region = np.abs(img_fft[y1:y2, x1:x2])
    max_idx = np.unravel_index(np.argmax(local_region), local_region.shape)
    y_c = y1 + max_idx[0]
    x_c = x1 + max_idx[1]

    # 设置窗口大小一致
    period_hor = period_vert = min(period_vert, period_hor)
    # 计算窗口大小（使用不同的垂直和水平尺寸）
    half_window_vert = period_vert // 2
    half_window_hor = period_hor // 2

    # 确保窗口在图像范围内
    y_start = int(max(0, y_c - half_window_vert))
    y_end = int(min(n_rows, y_c + half_window_vert))
    x_start = int(max(0, x_c - half_window_hor))
    x_end = int(min(n_columns, x_c + half_window_hor))

    # 4. 提取子区域并应用高斯加权
    sub_fft = img_fft[y_start:y_end, x_start:x_end]

    return sub_fft


def calculate_harmonic_periods(img_shape, pixel_size, pattern_period):
    """Calculate and adjust harmonic periods."""
    vert_harm = int((pixel_size[0] * img_shape[0]) / pattern_period)
    hor_harm = int((pixel_size[1] * img_shape[1]) / pattern_period)

    return [vert_harm, hor_harm]


def analyze_grating_data(img_fft, params, plot_flag=False):
    """Perform complete grating analysis."""

    results = single_2D_grating_analyses(img_fft, harmonic_periods=params["harmonic_periods"], unwrap_flag=True,
                                         plot_flag=plot_flag)

    # Calculate virtual pixel size and DPC
    virtual_pixel_size = [params['pixel_size'][0] * img_fft.shape[0] / results[0].shape[0],
                          params['pixel_size'][1] * img_fft.shape[1] / results[0].shape[1]]

    dpc_x = -results[3] * virtual_pixel_size[0] / \
            (params['det2sample'] * params['wavelength'])  # rad/像素
    dpc_y = -results[4] * virtual_pixel_size[1] / \
            (params['det2sample'] * params['wavelength'])  # rad/像素

    return *results[:3], dpc_x, dpc_y, virtual_pixel_size, params
