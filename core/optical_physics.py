def calculate_magnification_correction(params):
    """
    计算球面波DPC的理论比例因子

    Args:
        params: 包含光学系统参数的字典

    Returns:
        理论比例因子
    """

    d = params['det2sample']  # 探测器到样品距离
    R = params['source_dist']  # 光源到样品距离

    scale_factor = R / (R + d)
    return scale_factor


def calibrate_distance(params, magnification_grating, tol=0.05):
    """
    根据放大倍数校准距离参数

    已知:
    - magnification = (det2sample + source_dist) / source_dist
    - det2sample 是准确的

    Args:
        params (dict): 系统参数
        magnification_grating (float): 放大倍数
        tol (float): 允许的相对误差 (默认 5%)
    """
    # 原始值
    temp0 = params["source_dist"]
    params["focus_adjust"] = 0
    # 计算新的 source_dist
    temp1 = params["det2sample"] / (magnification_grating - 1)
    # print(f"Calculated source_dist: {temp1:.4f} m (old: {temp0:.4f} m)")

    # 合理性检查：相对误差
    if -1 < (temp1 - temp0) < 1:
        params["source_dist"] = temp1
        params["focus_adjust"] = temp0 - temp1
        # print("Source distance updated.")
    else:
        print("⚠️ Warning: Calculated source_dist deviates too much! "
        "Please check det2sample and grating_period.")

    # 保存旧值 & 更新总距离
    params['old_source_dist'] = temp0
    params["total_dist"] = params["det2sample"] + params["source_dist"]


    return params
