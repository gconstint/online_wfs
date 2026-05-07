import numpy as np


def select_circular_roi(
    phase_error,
    fit_params,
    virtual_pixel_size,
    default_radius_fraction=0.9,
):
    """
    Perform interactive ROI selection and return masked phase_error with aperture parameters.

    Parameters
    ----------
    phase_error : np.ndarray
        Phase error data to select ROI from
    fit_params : tuple
        Parabolic fit parameters (x0, y0, Rx, Ry, A)
    virtual_pixel_size : tuple
        Pixel size (py, px) in meters
    wavelength : float
        Wavelength in meters
    interactive : bool, optional
        If True, show interactive ROI selector. If False, use default parameters.
        Default is True.
    default_radius_fraction : float, optional
        Default radius as fraction of image size when interactive=False.
        Default is 0.8 (80% of image).
    save_path : str, optional
        Directory path to save the ROI selector figure. The image will be saved as
        'roi_selection.png' in this directory. If None, figure is not saved.
        Default is None.
    verbose : bool, optional
        Whether to print status messages. Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - phase_error_masked : np.ndarray with NaN outside ROI
        - aperture_center : tuple (y, x) in meters
        - aperture_radius_fraction : float
    """
    # Use the parabolic fit center as the aperture center (fixed)
    x0, y0, Rx, Ry, A = fit_params

    # Initial parameters
    h, w = phase_error.shape
    py, px = virtual_pixel_size

    # Use default parameters
    center_x_m = x0
    center_y_m = y0
    max_radius = min(h * py, w * px) / 2.0
    radius_m = max_radius * default_radius_fraction

    # Create circular mask and apply NaN outside ROI
    # Coordinate system from phase_fit: y = (row_index - h/2) * py
    # So: row_index = y_phys / py + h/2
    yy, xx = np.mgrid[0:h, 0:w]
    cx_px = center_x_m / px + w / 2.0  # Physical x to pixel column
    cy_px = center_y_m / py + h / 2.0  # Physical y to pixel row
    radius_px = radius_m / min(px, py)

    dist = np.sqrt((xx - cx_px) ** 2 + (yy - cy_px) ** 2)
    roi_mask = dist <= radius_px

    # Crop to bounding box of ROI based on radius (not just mask extent)
    # This ensures we have complete circular ROI in the cropped image

    # Calculate crop size based on radius with some padding
    crop_radius_px = int(np.ceil(radius_px)) + 2  # Add 2 pixels padding
    crop_size = 2 * crop_radius_px

    # Center the crop around the ROI center
    cy_px_int = int(np.round(cy_px))
    cx_px_int = int(np.round(cx_px))

    # Calculate crop boundaries
    row_start = max(0, cy_px_int - crop_radius_px)
    row_end = min(h, cy_px_int + crop_radius_px)
    col_start = max(0, cx_px_int - crop_radius_px)
    col_end = min(w, cx_px_int + crop_radius_px)

    # Ensure we have the full crop size if possible
    actual_row_size = row_end - row_start
    actual_col_size = col_end - col_start

    # Adjust if we're at boundaries
    if actual_row_size < crop_size and row_start > 0:
        row_start = max(0, row_end - crop_size)
    if actual_row_size < crop_size and row_end < h:
        row_end = min(h, row_start + crop_size)

    if actual_col_size < crop_size and col_start > 0:
        col_start = max(0, col_end - crop_size)
    if actual_col_size < crop_size and col_end < w:
        col_end = min(w, col_start + crop_size)

    # Crop the phase_error
    phase_error_cropped = phase_error[row_start:row_end, col_start:col_end].copy()
    roi_mask_cropped = roi_mask[row_start:row_end, col_start:col_end]

    # Apply mask to cropped region (set outside ROI to NaN)
    phase_error_cropped[~roi_mask_cropped] = np.nan

    # Calculate new center in cropped coordinates (relative to cropped image center)
    crop_h, crop_w = phase_error_cropped.shape
    # Center offset from cropped image center
    new_cx_px = cx_px - col_start - crop_w / 2.0
    new_cy_px = cy_px - row_start - crop_h / 2.0
    new_center_x_m = new_cx_px * px
    new_center_y_m = new_cy_px * py

    # Calculate aperture parameters for cropped image
    # Center should be close to (0, 0) if cropping was centered correctly
    aperture_center = (new_center_y_m, new_center_x_m)  # (y, x) in meters
    max_cropped_radius_m = min(crop_h * py, crop_w * px) / 2.0
    aperture_radius_fraction = min(radius_m / max_cropped_radius_m, 0.99)  # Cap at 0.99

    return {
        "phase_error_cropped": phase_error_cropped,
        "aperture_center": aperture_center,
        "aperture_radius_fraction": aperture_radius_fraction,
        "crop_info": {
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
        },
    }
