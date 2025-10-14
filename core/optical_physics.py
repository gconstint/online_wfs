def calculate_magnification_correction(params):
    """
    Calculate the theoretical scaling factor for spherical wave DPC

    Args:
        params: A dictionary containing optical system parameters

    Returns:
        The theoretical scaling factor
    """

    d = params['det2sample']  # Detector-to-sample distance
    R = params['source_dist']  # Source-to-sample distance

    scale_factor = R / (R + d)
    return scale_factor


def calibrate_distance(params, magnification_grating, tol=0.05):
    """
    Calibrate distance parameters based on magnification

    Known:
    - magnification = (det2sample + source_dist) / source_dist
    - det2sample is accurate

    Args:
        params (dict): System parameters
        magnification_grating (float): Magnification factor
        tol (float): Allowed relative error (default 5%)
    """
    # Original value
    temp0 = params["source_dist"]
    params["focus_adjust"] = 0
    # Calculate new source_dist
    temp1 = params["det2sample"] / (magnification_grating - 1)
    # print(f"Calculated source_dist: {temp1:.4f} m (old: {temp0:.4f} m)")

    # Sanity check: relative error
    if -1 < (temp1 - temp0) < 1:
        params["source_dist"] = temp1
        params["focus_adjust"] = temp0 - temp1
        # print("Source distance updated.")
    else:
        print("⚠️ Warning: Calculated source_dist deviates too much! "
        "Please check det2sample and grating_period.")

    # Save old value & update total distance
    params['old_source_dist'] = temp0
    params["total_dist"] = params["det2sample"] + params["source_dist"]


    return params
