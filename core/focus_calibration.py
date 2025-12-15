"""
Focus Position Calibration Module

This module provides functions for calibrating the detector-to-focus distance
using Zernike defocus coefficient (C4) from wavefront analysis.

The calibration is based on the physical relationship between wavefront curvature
and the distance from detector to focus point.
"""

import numpy as np
from typing import Dict, Any, Union, Sequence

from core.zernike_analysis import perform_zernike_analysis


def calibrate_focus_position(
    fitted_phase: np.ndarray,
    roi_result: Dict[str, Any],
    params: Dict[str, Any],
    virtual_pixel_size: Union[list, Sequence[float]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    根据 fitted_phase 的 Zernike 系数标定焦点位置。

    对 fitted_phase 进行前 6 项 Zernike 拟合 (j=0 到 j=5)，获取:
    - C3 (Astigmatism 0°): 斜向像散
    - C4 (Defocus): 离焦
    - C5 (Astigmatism 45°): 正交像散

    基于这些系数计算标准焦点位置（基于 C4）和像散焦点位置（基于 C3/C4/C5）。

    Parameters
    ----------
    fitted_phase : np.ndarray
        抛物面拟合后的相位 [rad]
    roi_result : dict
        ROI 选择结果，包含 crop_info, aperture_center, aperture_radius_fraction
    params : dict
        系统参数，包含 wavelength, total_dist
    virtual_pixel_size : list
        虚拟像素尺寸 (py, px) [m]
    verbose : bool
        是否打印详细结果

    Returns
    -------
    dict
        标定结果，包含:
        - R, Delta_z, C4, C4_opd (标准焦点标定)
        - astigmatic_focus: 像散焦点结果 (R_x, R_y, Delta_z_x, Delta_z_y 等)
    """
    # 从 roi_result 中获取裁剪边界
    crop_info = roi_result["crop_info"]
    fitted_phase_cropped = fitted_phase[
        crop_info["row_start"] : crop_info["row_end"],
        crop_info["col_start"] : crop_info["col_end"],
    ]

    # 对 fitted_phase 进行 Zernike 分析（需要 j=0 到 j=5，共 6 项）
    fitted_zernike_coeffs, _, _, _, _, _ = perform_zernike_analysis(
        phase=fitted_phase_cropped,
        pixel_size=virtual_pixel_size,
        wavelength=params["wavelength"],
        num_terms=6,  # j=0 到 j=5，获取 C3, C4, C5
        aperture_center=roi_result["aperture_center"],
        aperture_radius_fraction=roi_result["aperture_radius_fraction"],
        use_radial_tukey_weight=True,
        verbose=verbose,
    )

    # 获取 Zernike 系数 (只需要 C4 和 C5)
    C4 = fitted_zernike_coeffs[4]  # Defocus (OSA j=4)
    C5 = fitted_zernike_coeffs[5]  # Astigmatism X-Y (OSA j=5)

    if verbose:
        print("\nFitted phase Zernike coefficients:")
        print(f"  C₄ (Defocus):     {C4:.6f} rad ({C4 / (2 * np.pi):.6f} λ)")
        print(f"  C₅ (Astig X-Y):   {C5:.6f} rad ({C5 / (2 * np.pi):.6f} λ)")

    # R0: 理想探测器到焦点的距离
    R0 = params["total_dist"]

    # r_max: 光束半径，使用 ROI 选择的物理半径
    cropped_size = min(roi_result["phase_error_cropped"].shape)
    r_max = (
        (cropped_size / 2)
        * roi_result["aperture_radius_fraction"]
        * np.mean(virtual_pixel_size)
    )

    # 1. 标准焦点位置标定（仅基于 C4）
    focus_result = calculate_focus_distance(
        C4=C4,
        R0=R0,
        r_max=r_max,
        wavelength=params["wavelength"],
        verbose=verbose,
    )

    # 2. 像散焦点位置标定（基于 C4/C5）
    astigmatic_focus_result = calculate_astigmatic_focus(
        C4=C4,
        C5=C5,
        R0=R0,
        r_max=r_max,
        wavelength=params["wavelength"],
        verbose=verbose,
    )

    # 合并结果
    result = focus_result.copy()
    result["astigmatic_focus"] = astigmatic_focus_result
    result["C5"] = C5

    return result


def calculate_focus_from_dpc(
    fit_params: list,
    wavelength: float,
    reference_distance: float = 0.465,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    根据 DPC 抛物面拟合参数计算焦点位置（曲率叠加法）。

    物理原理
    --------
    将距离转换成曲率 (K = 1/R) 来进行加减运算，因为曲率是可以线性叠加的：
        K_total = K_ideal + K_residual

    展开为距离公式：
        1/R_real = 1/R_0 + λ*a/π

    其中：
        - R_real: 真实焦点位置
        - R_0: 理想参考距离
        - λ: X 射线波长
        - a: 拟合得到的二次项系数 (单位 rad/m²)

    二次项系数 a 从拟合参数提取：
        a_x = A / Rx²
        a_y = A / Ry²

    Parameters
    ----------
    fit_params : list
        DPC 抛物面拟合参数 [x0, y0, Rx, Ry, A]
        - x0, y0: 抛物面中心位置 [m]
        - Rx, Ry: X/Y 方向虚拟半径 [m]
        - A: 振幅 [rad]
    wavelength : float
        X 射线波长 [m]
    reference_distance : float
        理想参考距离 R_0 [m]，默认 0.465 m
    verbose : bool
        是否打印详细结果

    Returns
    -------
    dict
        包含标定结果的字典：
        - R_x : float - X 方向焦点距离 [m]
        - R_y : float - Y 方向焦点距离 [m]
        - R_avg : float - 平均焦点距离 (R_x + R_y) / 2 [m]
        - Delta_x : float - X 方向焦点偏移量 R_x - R_0 [m]
        - Delta_y : float - Y 方向焦点偏移量 R_y - R_0 [m]
        - Delta_avg : float - 平均焦点偏移量 [m]
        - a_x : float - X 方向二次项系数 [rad/m²]
        - a_y : float - Y 方向二次项系数 [rad/m²]
        - R_0 : float - 参考距离 [m]
    """
    x0, y0, Rx, Ry, A = fit_params
    R_0 = reference_distance

    # 检查有效性
    if A == 0 or Rx <= 0 or Ry <= 0:
        if verbose:
            print(
                "Warning: Invalid fit parameters (A=0 or R<=0), cannot calculate focus."
            )
        return {
            "R_x": np.inf,
            "R_y": np.inf,
            "R_avg": np.inf,
            "Delta_x": np.nan,
            "Delta_y": np.nan,
            "Delta_avg": np.nan,
            "a_x": 0.0,
            "a_y": 0.0,
            "R_0": R_0,
        }

    # 从拟合参数提取二次项系数
    a_x = A / (Rx**2)  # rad/m²
    a_y = A / (Ry**2)  # rad/m²

    # 残余波前的曲率贡献
    K_residual_x = wavelength * a_x / np.pi  # 1/m
    K_residual_y = wavelength * a_y / np.pi  # 1/m

    # 总曲率 = 理想曲率 + 残余曲率
    K_ideal = 1.0 / R_0  # 1/m
    K_total_x = K_ideal + K_residual_x
    K_total_y = K_ideal + K_residual_y

    # 转换回距离: R_real = 1 / K_total
    R_x = 1.0 / K_total_x if abs(K_total_x) > 1e-14 else np.inf
    R_y = 1.0 / K_total_y if abs(K_total_y) > 1e-14 else np.inf

    # 焦点偏移量
    Delta_x = R_x - R_0
    Delta_y = R_y - R_0

    # 平均焦点位置和偏移量
    R_avg = (R_x + R_y) / 2.0
    Delta_avg = (Delta_x + Delta_y) / 2.0

    if verbose:
        print("\n" + "=" * 60)
        print("DPC 拟合焦点标定结果 (曲率叠加法)".center(60))
        print("=" * 60)
        print("计算公式: 1/R_real = 1/R_0 + λ*a/π")
        print("-" * 60)
        print("输入参数:")
        print(f"  - 参考距离 R_0:       {R_0:.6f} m ({R_0 * 1e3:.3f} mm)")
        print(f"  - 波长 λ:             {wavelength * 1e9:.4f} nm")
        print(f"  - 拟合振幅 A:         {A:.4f} rad")
        print(f"  - 虚拟半径 Rx:        {Rx * 1e6:.2f} μm")
        print(f"  - 虚拟半径 Ry:        {Ry * 1e6:.2f} μm")
        print("-" * 60)
        print("二次项系数 a = A/R²:")
        print(f"  - a_x = {a_x:.4e} rad/m²")
        print(f"  - a_y = {a_y:.4e} rad/m²")
        print("-" * 60)
        print("标定结果:")
        print(f"  - X 方向焦点距离 R_x: {R_x:.6f} m ({R_x * 1e3:.3f} mm)")
        print(f"  - Y 方向焦点距离 R_y: {R_y:.6f} m ({R_y * 1e3:.3f} mm)")
        print(f"  - 平均焦点距离 R_avg: {R_avg:.6f} m ({R_avg * 1e3:.3f} mm)")
        print(f"  - X 方向偏移量 ΔR_x:  {Delta_x * 1e3:.4f} mm")
        print(f"  - Y 方向偏移量 ΔR_y:  {Delta_y * 1e3:.4f} mm")
        print(f"  - 平均偏移量 ΔR_avg:  {Delta_avg * 1e3:.4f} mm")
        print("=" * 60 + "\n")

    return {
        "R_x": R_x,
        "R_y": R_y,
        "R_avg": R_avg,
        "Delta_x": Delta_x,
        "Delta_y": Delta_y,
        "Delta_avg": Delta_avg,
        "a_x": a_x,
        "a_y": a_y,
        "R_0": R_0,
    }


def calculate_focus_distance(
    C4: float,
    R0: float,
    r_max: float,
    wavelength: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    根据 Zernike 离焦系数 C4 计算探测器到焦点的实际距离。

    物理原理
    --------
    理想波前与实际波前的曲率差可以近似表示为（OPD 单位，米）：
        W_err(r) ≈ (r²/2R) - (r²/2R₀) = (r²/2) × (1/R - 1/R₀)

    相位与 OPD 的关系：φ = (2π/λ) × W

    Zernike 离焦系数与物理矢高的关系：
        Sag = √3 × C₄_opd

    注意：虽然 Z₄ = √3(2ρ² - 1) 的 ρ² 系数为 2√3，但这里的 "2" 是为了
    Zernike 多项式的正交归一化，不代表物理曲率的两倍。物理矢高应使用 √3 × C₄。

    匹配物理公式：
        √3 × C₄_opd = r_max² / (2R) - r_max² / (2R₀) = (r_max²/2) × (1/R - 1/R₀)

    求解得：
        R = (1/R₀ + 2√3 × C₄_opd / r_max²)⁻¹

    Parameters
    ----------
    C4 : float
        Zernike 离焦系数 (OSA/ANSI 索引 j=4)，单位：rad。
    R0 : float
        设定的理想探测器到焦点的距离 [m]。
    r_max : float
        光束在探测器上的半径（瞳径）[m]。
    wavelength : float
        波长 [m]。
    verbose : bool, optional
        是否打印详细计算结果。默认为 True。

    Returns
    -------
    dict
        包含标定结果的字典：
        - R : float - 实际探测器到焦点的距离 [m]
        - Delta_z : float - 焦点偏移量 Δz = R - R₀ [m]
        - Delta_z_approx : float - 线性近似的偏移量 [m]
        - C4 : float - 输入的 Zernike 离焦系数 [rad]
        - C4_opd : float - C4 转换为 OPD [m]

    Notes
    -----
    符号约定（焦点位置校准）：
    - 如果 Δz < 0 (R < R₀): 焦点太靠近探测器 → 焦点需要远离探测器
    - 如果 Δz > 0 (R > R₀): 焦点太远离探测器 → 焦点需要向探测器方向移动
    """
    sqrt3 = np.sqrt(3)

    # 将 C4 从弧度转换为 OPD（米）
    C4_opd = C4 * wavelength / (2 * np.pi)

    # 计算完整公式：R = (1/R₀ + 2√3 × C₄_opd / r_max²)⁻¹
    curvature_correction = 2 * sqrt3 * C4_opd / (r_max**2)

    # 处理极端情况避免除零
    inv_R = 1.0 / R0 + curvature_correction
    if abs(inv_R) < 1e-12:
        R = np.inf if inv_R >= 0 else -np.inf
    else:
        R = 1.0 / inv_R

    # 精确位置误差
    Delta_z = R - R0

    # 线性近似：Δz ≈ -2√3 × R₀² × C₄_opd / r_max²
    Delta_z_approx = -2 * sqrt3 * (R0**2) * C4_opd / (r_max**2)

    if verbose:
        print("\n" + "=" * 60)
        print("焦点位置标定结果".center(60))
        print("=" * 60)
        print("输入参数:")
        print(f"  - Zernike C₄ (相位): {C4:.6f} rad ({C4 / (2 * np.pi):.6f} λ)")
        print(f"  - Zernike C₄ (OPD):  {C4_opd * 1e9:.6f} nm")
        print(f"  - 理想距离 R₀:       {R0:.6f} m ({R0 * 1e3:.3f} mm)")
        print(f"  - 光束半径 r_max:    {r_max * 1e3:.3f} mm")
        print(f"  - 波长 λ:            {wavelength * 1e9:.4f} nm")
        print("-" * 60)
        print("标定结果:")
        print(f"  - 探测器到焦点实际距离 R:  {R:.6f} m ({R * 1e3:.3f} mm)")
        print(f"  - 焦点偏移量 Δz = R - R₀:  {Delta_z * 1e3:.6f} mm")
        print("=" * 60 + "\n")

    return {
        "R": R,
        "Delta_z": Delta_z,
        "Delta_z_approx": Delta_z_approx,
        "C4": C4,
        "C4_opd": C4_opd,
    }


def calculate_astigmatic_focus(
    C4: float,
    C5: float,
    R0: float,
    r_max: float,
    wavelength: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    计算像散情况下的 X/Y 方向焦点位置。

    当波前存在像散时，X 和 Y 方向的曲率半径不同，导致两个不同的焦点位置。
    本函数基于 Zernike 系数 C4 (Defocus) 和 C5 (Astigmatism) 计算两个方向的独立焦点位置。

    物理原理
    --------
    对于 OSA/ANSI 归一化的 Zernike 多项式：
    - Z₄ (n=2, m=0):  √3 × (2ρ² - 1) → 离焦 (平均曲率)
    - Z₅ (n=2, m=2):  √6 × ρ² × cos(2θ) ∝ (x² - y²) → 正交像散 (X-Y 曲率差)

    X/Y 方向的曲率半径：
        1/R_x = 1/R₀ + (2√3 × C₄_opd + 2√6 × C₅_opd) / r_max²
        1/R_y = 1/R₀ + (2√3 × C₄_opd - 2√6 × C₅_opd) / r_max²

    Parameters
    ----------
    C4 : float
        Zernike 离焦系数 (OSA j=4, Defocus) [rad]
    C5 : float
        Zernike 正交像散系数 (OSA j=5, Astigmatism X-Y) [rad]
    R0 : float
        设定的理想探测器到焦点的距离 [m]
    r_max : float
        光束在探测器上的半径（瞳径）[m]
    wavelength : float
        波长 [m]
    verbose : bool
        是否打印详细结果

    Returns
    -------
    dict
        包含以下键值:
        - R_x : float - X 方向曲率半径 [m]
        - R_y : float - Y 方向曲率半径 [m]
        - Delta_z_x : float - X 方向焦点偏移量 [m]
        - Delta_z_y : float - Y 方向焦点偏移量 [m]
        - astigmatism_distance : float - 像散距离 |R_x - R_y| [m]
        - C4, C5 : float - 输入的 Zernike 系数 [rad]
    """
    sqrt3 = np.sqrt(3)
    sqrt6 = np.sqrt(6)

    # 将 Zernike 系数从弧度转换为 OPD（米）
    C4_opd = C4 * wavelength / (2 * np.pi)
    C5_opd = C5 * wavelength / (2 * np.pi)

    inv_r_max_sq = 1.0 / (r_max**2)

    # 离焦贡献（两个方向相同）
    defocus_correction = 2 * sqrt3 * C4_opd * inv_r_max_sq

    # 像散贡献（两个方向相反）
    # C5 (cos 2θ ∝ x² - y²) 分量对应 X-Y 差异
    astig_correction = 2 * sqrt6 * C5_opd * inv_r_max_sq

    # X 方向曲率
    inv_R_x = 1.0 / R0 + defocus_correction + astig_correction
    if abs(inv_R_x) < 1e-12:
        R_x = np.inf if inv_R_x >= 0 else -np.inf
    else:
        R_x = 1.0 / inv_R_x

    # Y 方向曲率
    inv_R_y = 1.0 / R0 + defocus_correction - astig_correction
    if abs(inv_R_y) < 1e-12:
        R_y = np.inf if inv_R_y >= 0 else -np.inf
    else:
        R_y = 1.0 / inv_R_y

    # 焦点偏移量
    Delta_z_x = R_x - R0
    Delta_z_y = R_y - R0

    # 像散距离
    if np.isfinite(R_x) and np.isfinite(R_y):
        astigmatism_distance = abs(R_x - R_y)
    else:
        astigmatism_distance = np.inf

    if verbose:
        print("\n" + "=" * 70)
        print("像散焦点位置标定结果 (基于 Zernike C4/C5)".center(70))
        print("=" * 70)
        print("输入 Zernike 系数:")
        print(f"  - C₄ (Defocus):      {C4:.6f} rad ({C4 / (2 * np.pi):.6f} λ)")
        print(f"  - C₅ (Astig X-Y):    {C5:.6f} rad ({C5 / (2 * np.pi):.6f} λ)")
        print(f"  - 理想距离 R₀:       {R0:.6f} m ({R0 * 1e3:.3f} mm)")
        print(f"  - 光束半径 r_max:    {r_max * 1e3:.3f} mm")
        print("-" * 70)
        print("X 方向 (水平):")
        print(f"  - 曲率半径 R_x:      {R_x:.6f} m ({R_x * 1e3:.3f} mm)")
        print(f"  - 焦点偏移 Δz_x:     {Delta_z_x * 1e3:.6f} mm")
        print("Y 方向 (垂直):")
        print(f"  - 曲率半径 R_y:      {R_y:.6f} m ({R_y * 1e3:.3f} mm)")
        print(f"  - 焦点偏移 Δz_y:     {Delta_z_y * 1e3:.6f} mm")
        print("-" * 70)
        print(f"像散距离 |Rx - Ry|:    {astigmatism_distance * 1e3:.6f} mm")
        if astigmatism_distance < 1e-6:
            print("  → 近似球面波前，无明显像散")
        else:
            print("  → ⚠ 存在像散，X/Y 方向焦点位置不同")
        print("=" * 70 + "\n")

    return {
        "R_x": R_x,
        "R_y": R_y,
        "Delta_z_x": Delta_z_x,
        "Delta_z_y": Delta_z_y,
        "astigmatism_distance": astigmatism_distance,
        "C4": C4,
        "C5": C5,
        "C4_opd": C4_opd,
        "C5_opd": C5_opd,
    }
