"""
Undulator Source Distance Calculation Module

This module provides functions to calculate the Undulator source distance
using the Gaussian (thin lens) imaging formula based on SGI-measured focus position.

Optical Layout:
    Undulator → CRL → Focus → Grating → Detector
       |         |       |       |         |
       |<-L_src->|       |       |         |
                 |<-L_f->|       |         |
                         |<---- R -------->|

Where:
    - L_source: Undulator to CRL distance (object distance)
    - L_focus: CRL to focus distance (image distance)
    - R: Detector to focus distance (measured by SGI)
"""

from typing import Dict, Any, Optional


# CRL 材料常数 (常用材料的电子密度相关参数)
# 格式: material -> (Z/A ratio, density [g/cm³])
CRL_MATERIALS = {
    "Be": (4 / 9.012, 1.85),  # 铍
    "Al": (13 / 26.982, 2.70),  # 铝
    "C": (6 / 12.011, 2.26),  # 碳（金刚石）
    "Si": (14 / 28.086, 2.33),  # 硅
}


def calculate_delta_from_energy(
    energy_eV: float,
    material: str = "Be",
    custom_density: Optional[float] = None,
) -> float:
    """
    根据光子能量和材料计算折射率偏差 δ。

    使用简化公式（适用于远离吸收边的情况）：
        δ ≈ 2.701e-6 × (ρ [g/cm³]) × (Z/A) × (λ [Å])²

    Parameters
    ----------
    energy_eV : float
        光子能量 [eV]
    material : str, optional
        CRL 材料，可选 "Be", "Al", "C", "Si"。默认为 "Be"。
    custom_density : float, optional
        自定义密度 [g/cm³]。

    Returns
    -------
    float
        折射率偏差 δ（无量纲）
    """
    if material not in CRL_MATERIALS:
        raise ValueError(
            f"未知材料 '{material}'。可选材料: {list(CRL_MATERIALS.keys())}"
        )

    z_over_a, density = CRL_MATERIALS[material]
    if custom_density is not None:
        density = custom_density

    # 计算波长 [Å]
    hc = 12398.419  # eV·Å
    wavelength_angstrom = hc / energy_eV

    # δ ≈ 2.701e-6 × ρ × (Z/A) × λ²
    delta = 2.701e-6 * density * z_over_a * (wavelength_angstrom**2)

    return delta


def calculate_crl_focal_length(
    R: float,
    N: int,
    delta: Optional[float] = None,
    energy_eV: Optional[float] = None,
    material: str = "Be",
    verbose: bool = True,
) -> float:
    """
    计算 CRL 焦距: f = R / (2 × N × δ)

    Parameters
    ----------
    R : float
        单个透镜的曲率半径 [m]
    N : int
        透镜数量
    delta : float, optional
        折射率偏差 δ。如果未提供，根据 energy_eV 和 material 计算。
    energy_eV : float, optional
        光子能量 [eV]。当 delta 未提供时必需。
    material : str, optional
        CRL 材料。默认为 "Be"。
    verbose : bool, optional
        是否打印详细信息。

    Returns
    -------
    float
        CRL 焦距 [m]
    """
    if delta is None:
        if energy_eV is None:
            raise ValueError("必须提供 delta 或 energy_eV 来计算焦距")
        delta = calculate_delta_from_energy(energy_eV, material)

    f = R / (2 * N * delta)

    if verbose:
        print("\n" + "-" * 50)
        print("CRL 焦距计算")
        print("-" * 50)
        print(f"  曲率半径 R:     {R * 1e6:.1f} μm")
        print(f"  透镜数量 N:     {N}")
        print(f"  折射率偏差 δ:   {delta:.3e}")
        if energy_eV is not None:
            print(f"  光子能量:       {energy_eV:.0f} eV ({energy_eV / 1000:.2f} keV)")
            print(f"  材料:           {material}")
        print(f"  计算焦距 f:     {f * 1e3:.3f} mm ({f:.6f} m)")
        print("-" * 50 + "\n")

    return f


def calculate_source_distance(
    f: float,
    z_CRL: float,
    z_focus: Optional[float] = None,
    z_det: Optional[float] = None,
    R_measured: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    根据 SGI 测得的实际焦点位置，利用高斯成像公式反推 Undulator 源距离。

    光路布局: Undulator → CRL → 焦点 → Grating → 探测器

    计算步骤：
    1. z_focus = z_det - R_measured（焦点在探测器上游）
    2. L_focus = z_focus - z_CRL
    3. L_source = f × L_focus / (L_focus - f)

    Parameters
    ----------
    f : float
        CRL 焦距 [m]
    z_CRL : float
        CRL 位置 [m]
    z_focus : float, optional
        焦点位置 [m]。如果未提供，通过 z_det 和 R_measured 计算。
    z_det : float, optional
        探测器位置 [m]
    R_measured : float, optional
        SGI 测得的探测器到焦点距离 [m]

    Returns
    -------
    dict
        包含 L_source, L_focus, z_focus, z_source, f
    """
    # Step 1: 计算焦点位置
    if z_focus is not None:
        pass
    elif z_det is not None and R_measured is not None:
        z_focus = z_det - R_measured
    else:
        raise ValueError("必须提供 z_focus，或同时提供 z_det 和 R_measured")

    # Step 2: 计算像距
    L_focus = z_focus - z_CRL

    if L_focus <= f:
        raise ValueError(f"物理错误：L_focus ({L_focus:.6f} m) <= f ({f:.6f} m)。")

    # Step 3: 反推源距离
    L_source = (f * L_focus) / (L_focus - f)
    z_source = z_CRL - L_source

    if verbose:
        print("\n" + "=" * 70)
        print("Undulator 源距离计算结果 (高斯成像公式)".center(70))
        print("=" * 70)
        print("输入参数:")
        print(f"  - CRL 焦距 f:          {f * 1e3:.3f} mm ({f:.6f} m)")
        print(f"  - CRL 位置 z_CRL:      {z_CRL:.6f} m")
        print(f"  - 焦点位置 z_focus:    {z_focus:.6f} m")
        if z_det is not None and R_measured is not None:
            print(f"    (由 z_det={z_det:.3f}m, R_measured={R_measured:.6f}m 计算)")
        print("-" * 70)
        print("计算结果:")
        print("  - 像距 L_focus = z_focus - z_CRL:")
        print(f"      {L_focus * 1e3:.6f} mm ({L_focus:.6f} m)")
        print("  - L_focus - f (应为正且很小):")
        print(f"      {(L_focus - f) * 1e6:.3f} μm ({(L_focus - f) * 1e3:.6f} mm)")
        print("  - 物距 L_source (Undulator 到 CRL):")
        print(f"      {L_source:.3f} m")
        print("  - 光源绝对位置 z_source = z_CRL - L_source:")
        print(f"      {z_source:.6f} m")
        print("=" * 70 + "\n")

    return {
        "L_source": L_source,
        "L_focus": L_focus,
        "z_focus": z_focus,
        "z_source": z_source,
        "f": f,
    }


def calculate_undulator_source_distance(
    calibration_result: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    高层封装函数：从 calibration_result 和 params 中提取参数，计算 Undulator 源距离。

    光路布局: Undulator → CRL → 焦点 → Grating → 探测器

    需要的 params 参数：
    - 必需: crl_position
    - 焦距: crl_focal_length 或 (crl_radius + crl_lens_count)
    - 可选: detector_position, crl_material, wavelength

    Parameters
    ----------
    calibration_result : dict
        来自 calibrate_focus_position 的结果，包含 "R"（探测器到焦点距离）
    params : dict
        系统参数
    verbose : bool
        是否打印详细信息

    Returns
    -------
    dict or None
        包含源距离计算结果的字典，如果缺少必要参数则返回 None
    """
    # 检查必要参数
    has_crl_position = "crl_position" in params
    has_focal_length = "crl_focal_length" in params
    can_calc_focal_length = "crl_radius" in params and "crl_lens_count" in params

    if not has_crl_position:
        if verbose:
            print("信息: 未提供 crl_position，跳过源距离计算")
        return None

    if not (has_focal_length or can_calc_focal_length):
        if verbose:
            print(
                "信息: 未提供 crl_focal_length 或 (crl_radius + crl_lens_count)，跳过源距离计算"
            )
        return None

    # 计算或获取焦距
    if has_focal_length:
        crl_focal_length = params["crl_focal_length"]
    else:
        crl_material = params.get("crl_material", "Be")
        # 从波长反推能量
        hc = 12398.419e-10  # eV·m
        energy_eV = hc / params["wavelength"]

        crl_focal_length = calculate_crl_focal_length(
            R=params["crl_radius"],
            N=params["crl_lens_count"],
            energy_eV=energy_eV,
            material=crl_material,
            verbose=verbose,
        )

    # 获取探测器位置
    if "detector_position" in params:
        z_det = params["detector_position"]
    else:
        if verbose:
            print("警告: 未提供 detector_position，使用 crl_position + total_dist 估算")
        z_det = params["crl_position"] + params["total_dist"]

    # SGI 测得的探测器到焦点距离
    R_measured = calibration_result["R"]

    # 计算源距离
    result = calculate_source_distance(
        f=crl_focal_length,
        z_CRL=params["crl_position"],
        z_det=z_det,
        R_measured=R_measured,
        verbose=verbose,
    )

    # 添加焦距到结果
    result["crl_focal_length"] = crl_focal_length

    return result
