import numpy as np
import tifffile
from scipy import constants


def calculate_magnification_correction(params):
    """
    计算球面波DPC的理论比例因子

    Args:
        params: 包含光学系统参数的字典

    Returns:
        理论比例因子
    """
    d = params["det2sample"]  # 探测器到样品距离
    R = params["source_dist"]  # 光源到样品距离

    scale_factor = R / (R + d)
    return scale_factor


def load_images(
    image_path,
    dark_image_path=None,
    flat_image_path=None,
):
    """
    Load raw, dark, and flat images for image correction.

    Parameters
    ----------
    image_path : str
        Path to the sample image
    dark_image_path : str, optional
        Path to the dark image
    flat_image_path : str, optional
        Path to the flat image

    Returns
    -------
    tuple
        (sample_image, dark_image, flat_image) as float32 arrays
    """
    # Load and convert sample image
    image = tifffile.imread(image_path)
    image = np.array(image).astype(np.float32)

    # Handle dark image
    if dark_image_path:
        dark_image = tifffile.imread(dark_image_path)
        dark_image = np.array(dark_image).astype(np.float32)
    else:
        dark_image = None

    # Handle flat image
    if flat_image_path:
        flat_image = tifffile.imread(flat_image_path)
        flat_image = np.array(flat_image).astype(np.float32)
    else:
        flat_image = None

    return image, dark_image, flat_image


def calculate_wavelength(photon_energy):
    """Calculate the wavelength from photon energy."""
    hc = constants.value("inverse meter-electron volt relationship")  # h * c in eV * m
    return hc / photon_energy


def center_crop(img, target_size):
    """
    Center-crop a 2D image array to target_size x target_size.
    If the image is smaller than target_size in a dimension, that dimension is left as-is.
    Returns the cropped image.
    """
    height, width = img.shape
    if height > target_size:
        start_h = (height - target_size) // 2
        end_h = start_h + target_size
    else:
        start_h = 0
        end_h = height

    if width > target_size:
        start_w = (width - target_size) // 2
        end_w = start_w + target_size
    else:
        start_w = 0
        end_w = width

    return img[start_h:end_h, start_w:end_w]


def image_correction(
    image, flat=None, dark=None, epsilon=1e-8, normalize=True, **kwargs
):
    """
    通用平场校正函数

    参数
    ----
    image : ndarray
        自成像/样品图像
    flat : ndarray or None
        平场图像 (flat / open-beam)，可选
    dark : ndarray or None
        暗场图像 (dark)，可选
    epsilon : float
        避免除零的小常数
    normalize : bool
        是否将结果归一化到均值 ~1

    返回
    ----
    corrected_image : ndarray
        校正后的图像

    兼容性
    ----
    仍然兼容旧参数名: I_self, I_flat, I_dark, eps
    """

    # 转换为浮点数避免溢出
    image = np.asarray(image, dtype=np.float32)
    if flat is not None:
        flat = np.asarray(flat, dtype=np.float32)
    if dark is not None:
        dark = np.asarray(dark, dtype=np.float32)

    # ----------------------
    # Case 1: (image - dark) / (flat - dark)
    if flat is not None and dark is not None:
        numerator = image - dark
        denominator = flat - dark

    # Case 2: image / flat  (无 dark)
    elif flat is not None and dark is None:
        numerator = image
        denominator = flat

    # Case 3: image - dark  (无 flat)
    elif flat is None and dark is not None:
        numerator = image - dark
        denominator = np.ones_like(numerator)  # 不做平场，只做暗场扣除

    # Case 4: 只用 image
    else:
        numerator = image
        denominator = np.ones_like(numerator)

    # ----------------------
    # 防止除零
    valid_mask = denominator > epsilon
    corrected_image = np.zeros_like(numerator, dtype=np.float64)
    corrected_image[valid_mask] = numerator[valid_mask] / (
        denominator[valid_mask] + epsilon
    )

    # # 可选：归一化
    # if normalize and np.any(valid_mask):
    #     mean_value = np.mean(corrected_image[valid_mask])
    #     if mean_value > 0:
    #         corrected_image /= mean_value

    return corrected_image
