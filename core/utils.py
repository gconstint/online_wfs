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
    image = np.asarray(image, dtype=np.float64)

    # Case 3 and Case 4: No flat field correction needed
    if flat is None:
        if dark is not None:
            dark = np.asarray(dark, dtype=np.float64)
            return image - dark  # Simple dark subtraction, no division needed
        else:
            return image.copy()  # No correction needed, return copy

    # Case 1 and Case 2: Flat field correction
    flat = np.asarray(flat, dtype=np.float64)

    if dark is not None:
        dark = np.asarray(dark, dtype=np.float64)
        numerator = image - dark
        denominator = flat - dark
    else:
        numerator = image
        denominator = flat

    # Prevent division by near-zero values
    valid_mask = denominator > epsilon
    corrected_image = np.zeros_like(numerator, dtype=np.float64)
    corrected_image[valid_mask] = numerator[valid_mask] / (
        denominator[valid_mask] + epsilon
    )

    return corrected_image
