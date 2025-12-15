import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class CircularROISelector:
    def __init__(self, image, center, initial_radius, pixel_size, wavelength):
        self.image = image
        self.pixel_size = pixel_size
        self.wavelength = wavelength

        # Convert to display units (mm)
        h, w = image.shape
        py, px = pixel_size
        self.extent = (
            -w * px * 1e3 / 2,
            w * px * 1e3 / 2,
            -h * py * 1e3 / 2,
            h * py * 1e3 / 2,
        )

        # ROI parameters (in mm for display)
        self.center_x = center[0] * 1e3  # mm (fixed)
        self.center_y = center[1] * 1e3  # mm (fixed)
        self.radius = initial_radius * 1e3  # mm (adjustable)

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Circular ROI Selection")

        # Display image
        phase_nm = image * wavelength / (2 * np.pi) * 1e9
        self.im = self.ax.imshow(
            phase_nm, cmap="RdBu_r", extent=self.extent, origin="lower"
        )
        plt.colorbar(self.im, ax=self.ax, label="Phase Error (nm)")

        # Create circle
        self.circle = Circle(
            (self.center_x, self.center_y),
            self.radius,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
        self.ax.add_patch(self.circle)

        # Mark center (fixed)
        (self.center_marker,) = self.ax.plot(
            self.center_x,
            self.center_y,
            "g+",
            markersize=15,
            markeredgewidth=2,
            label="Fit Center (fixed)",
        )

        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_title("Adjust Circular ROI Radius - Drag to Resize | Enter: Done")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Mouse interaction
        self.press = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Status text
        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        self.update_status()

        plt.tight_layout()

    def update_status(self):
        self.status_text.set_text(
            f"Center: ({self.center_x:.2f}, {self.center_y:.2f}) mm (fixed)\n"
            f"Radius: {self.radius:.2f} mm"
        )

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press = (event.xdata, event.ydata)

    def on_release(self, event):
        self.press = None
        self.update_status()
        self.fig.canvas.draw()

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return

        x0, y0 = self.press

        # Calculate distance from center to current mouse position
        dx = event.xdata - self.center_x
        dy = event.ydata - self.center_y
        new_radius = np.sqrt(dx**2 + dy**2)

        # Update radius (minimum 0.1 mm)
        self.radius = max(0.1, new_radius)

        # Update circle
        self.circle.radius = self.radius

        self.update_status()
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == "enter":
            plt.close(self.fig)

    def get_roi_params(self):
        """Return ROI parameters in meters"""
        return {
            "center_x": self.center_x * 1e-3,  # m
            "center_y": self.center_y * 1e-3,  # m
            "radius": self.radius * 1e-3,  # m
        }


def select_circular_roi(
    phase_error,
    fit_params,
    virtual_pixel_size,
    wavelength,
    interactive=True,
    default_radius_fraction=0.8,
    save_path=None,
    verbose=True,
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

    if interactive:
        if verbose:
            print("\n" + "=" * 70)
            print("INTERACTIVE CIRCULAR ROI SELECTION FOR ZERNIKE ANALYSIS")
            print("=" * 70)
            print(f"Parabolic fit center (fixed): ({x0 * 1e6:.2f}, {y0 * 1e6:.2f}) μm")
            print("\nInstructions:")
            print("  - Left click and drag: Adjust circle radius")
            print("  - Press 'Enter' or close window when done")
            print("  - Center is fixed at the parabolic fit center")

        initial_radius = (
            min(h * py, w * px) * 0.4
        )  # 40% of image size as initial radius

        # Create and show ROI selector
        selector = CircularROISelector(
            phase_error,
            (x0, y0),  # Fixed center
            initial_radius,
            virtual_pixel_size,
            wavelength,
        )

        plt.show()

        # Get selected ROI parameters
        roi_params = selector.get_roi_params()
        center_x_m = roi_params["center_x"]
        center_y_m = roi_params["center_y"]
        radius_m = roi_params["radius"]

        # Save figure AFTER user interaction if save_path is provided
        if save_path:
            img_path = os.path.join(save_path, "roi_selection.png")
            selector.fig.savefig(img_path, dpi=300, bbox_inches="tight")
            if verbose:
                print("MESSAGE: The ROI selection image is saved")
                print("-" * 50)
    else:
        # Use default parameters
        center_x_m = x0
        center_y_m = y0
        max_radius = min(h * py, w * px) / 2.0
        radius_m = max_radius * default_radius_fraction

        if verbose:
            print("\n" + "=" * 70)
            print("USING DEFAULT ROI PARAMETERS (INTERACTIVE MODE DISABLED)")
            print("=" * 70)

    if verbose:
        print(f"\nSelected ROI:")
        print(f"  Center: ({center_x_m * 1e6:.2f}, {center_y_m * 1e6:.2f}) μm (fixed)")
        print(f"  Radius: {radius_m * 1e6:.2f} μm")

    # Create circular mask and apply NaN outside ROI
    # Coordinate system from phase_fit: y = (row_index - h/2) * py
    # So: row_index = y_phys / py + h/2
    yy, xx = np.mgrid[0:h, 0:w]
    cx_px = center_x_m / px + w / 2.0  # Physical x to pixel column
    cy_px = center_y_m / py + h / 2.0  # Physical y to pixel row
    radius_px = radius_m / min(px, py)

    dist = np.sqrt((xx - cx_px) ** 2 + (yy - cy_px) ** 2)
    roi_mask = dist <= radius_px

    if verbose:
        print(
            f"  Valid pixels: {np.sum(roi_mask)} / {h * w} ({np.sum(roi_mask) / (h * w) * 100:.1f}%)"
        )

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

    if verbose:
        print(f"\nCropped region:")
        print(f"  Crop size: {crop_h} × {crop_w} pixels")
        print(
            f"  Crop bounds: rows[{row_start}:{row_end}], cols[{col_start}:{col_end}]"
        )
        print(
            f"  New center offset: ({new_center_x_m * 1e6:.2f}, {new_center_y_m * 1e6:.2f}) μm"
        )

    # Calculate aperture parameters for cropped image
    # Center should be close to (0, 0) if cropping was centered correctly
    aperture_center = (new_center_y_m, new_center_x_m)  # (y, x) in meters
    max_cropped_radius_m = min(crop_h * py, crop_w * px) / 2.0
    aperture_radius_fraction = min(radius_m / max_cropped_radius_m, 0.99)  # Cap at 0.99

    if verbose:
        print(
            f"  Aperture center (cropped coords): ({new_center_x_m * 1e6:.2f}, {new_center_y_m * 1e6:.2f}) μm"
        )
        print(
            f"  Aperture radius: {radius_m * 1e6:.2f} μm vs max {max_cropped_radius_m * 1e6:.2f} μm"
        )
        print(f"  Aperture radius fraction: {aperture_radius_fraction:.3f}")

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
