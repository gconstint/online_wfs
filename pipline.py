from os import cpu_count

import numpy as np
from scipy.fft import fft2, fftshift

from core import (
    calculate_and_visualize_beam, analyze_focus_sampling_from_beam,
    preprocess_dpc,
    analyze_grating_data, accurate_harmonic_periods, rotate_image_by_peaks, calculate_harmonic_periods,
    calculate_magnification_correction, calibrate_distance,
    dpc_integration,
    perform_wavefront_fitting,
    two_steps_fresnel_method,
    center_crop, image_correction, load_images,
)


def task(params: dict):
    # %%
    # Image Loading and Initial Processing
    # Load raw, dark, and flat field images for beam analysis
    img, dark, flat = load_images(params['image_path'], params["dark_image_path"],
                                  params["flat_image_path"])  # in np.float32 for better performance

    # %%
    # Image Normalization
    # Apply dark field subtraction and flat field correction
    img = image_correction(img, flat=flat, dark=dark, epsilon=1e-8, normalize=False)

    # %%
    # Image Preprocessing
    # Center-crop the image to standardized size for analysis
    img = center_crop(img, target_size=2048)

    # %%
    # Frequency Domain Analysis
    # Convert image to frequency domain using FFT
    img32 = np.asarray(img, dtype=np.float32, order='C')
    img_fft = fftshift(fft2(img32, norm='ortho', workers=cpu_count()))

    # %%
    # Grating Pattern Analysis
    # Calculate initial harmonic periods based on physical parameters
    harmonic_periods = calculate_harmonic_periods(
        img.shape,
        params['pixel_size'],
        params['pattern_period'],
    )
    # %%
    # Harmonic Period Refinement
    # Fine-tune the periods by analyzing frequency domain peaks
    harmonic_periods, peak_positions = accurate_harmonic_periods(img_fft, init_periods=harmonic_periods)
    # %%
    # Image Alignment
    # Rotate image to align with grating pattern axes
    img, rotation_angle = rotate_image_by_peaks(img, peak_positions)
    # %%
    # Pattern Period Update
    # Calculate refined pattern period from measured data
    p_x = params['pixel_size'][1] * img.shape[1] / harmonic_periods[1]
    p_y = params['pixel_size'][0] * img.shape[0] / harmonic_periods[0]
    params["pattern_period"] = float((p_x + p_y) / 2.0)

    params["harmonic_periods"] = harmonic_periods

    # %% md
    # ## Setup 2: Grating analysis and DPC calculation
    # ### Processing steps:
    # ### - analyze_grating_data: Extract 00, 01, 10 harmonics and calculate differential phase contrast (DPC)
    # ### - Compute magnification_grating from pattern_period and grating_period (must be > 1)
    # ### - calibrate_distance: Adjust source_dist based on measured magnification_grating
    # ### - calculate_magnification_correction: Apply scale factor correction to DPC signals
    # ### - preprocess_dpc: Apply filtering to reduce noise in DPC images
    # ### - dpc_integration: Integrate DPC gradients to reconstruct phase

    # %%
    # Extract harmonic components and calculate DPC signals
    # Analyze grating data to obtain:
    # - int00, int01, int10: Harmonic intensity components
    # - dpc_x, dpc_y: Differential phase contrast signals
    # - virtual_pixel_size: Effective pixel size after analysis
    results = analyze_grating_data(img_fft, params, plot_flag=False)
    int00, int01, int10, dpc_x, dpc_y, virtual_pixel_size, params = results

    # %%
    # Magnification Analysis
    # Calculate and validate the grating pattern magnification
    magnification_grating = params["pattern_period"] / params["grating_period"]
    assert magnification_grating > 1.0, f"magnification_grating({magnification_grating}) should be greater than 1.0"

    # %%
    # Distance Calibration
    # Adjust source distance based on measured magnification
    params = calibrate_distance(params, magnification_grating)

    # %%
    # Distance Relationship Analysis
    # Analyze and output focus adjustment parameter
    yield "output1", (params['focus_adjust'],)

    # %%
    # DPC Signal Processing
    # Apply scale factor correction and noise reduction
    scale_factor = calculate_magnification_correction(params)
    dpc_x *= scale_factor
    dpc_y *= scale_factor
    dpc_x_filt = preprocess_dpc(dpc_x)
    dpc_y_filt = preprocess_dpc(dpc_y)

    # %%
    # Phase Reconstruction
    # Convert DPC to physical units and integrate to obtain phase
    dpc_x_physical = dpc_x_filt * virtual_pixel_size[0]  # rad/m
    dpc_y_physical = dpc_y_filt * virtual_pixel_size[1]  # rad/m
    phase = dpc_integration(dpc_x_physical, dpc_y_physical)

    # %% md
    # ## Setup 3: Wavefront Analysis and Fitting
    # Process Overview:
    # 1. Fit an ideal spherical wavefront to the reconstructed phase
    # 2. Generate fitted phase, error measurements, and fitting parameters
    # 3. Compare original vs fitted phase data
    # 4. Analyze error profiles to identify potential aberrations

    # %%
    # Wavefront Analysis
    # Perform wavefront fitting and error analysis
    fitted_phase, phase_error, fit_params, = perform_wavefront_fitting(phase, virtual_pixel_size)

    # %%
    # Phase Analysis Output
    # Export phase error data with calibration parameters
    yield "output2", (phase_error, virtual_pixel_size, params['wavelength'])

    # %%
    # Beam Position Analysis
    # Calculate beam characteristics on self-imaging plane
    beam_position, beam_size = calculate_and_visualize_beam(int00, virtual_pixel_size,
                                                            title="self-image Position and Size Analysis")

    # %%
    # Focus Analysis Preparation
    # Calculate sampling parameters for focus plane analysis
    dx, dy = virtual_pixel_size[1], virtual_pixel_size[0]
    dx_focus_x, dx_focus_y, *_ = analyze_focus_sampling_from_beam(
        int00, dx, dy, params['wavelength'], params['total_dist'], beam_size)

    # %%
    # Beam Propagation and Focus Analysis
    # Combine amplitude and phase for beam propagation
    beam = int00 * np.exp(1j * phase)
    # Perform back-propagation to focus using Fresnel method
    focus_spot = two_steps_fresnel_method(beam, params['wavelength'], -params['total_dist'], dx, dy, 1)

    # %%
    # Focus Characterization
    # Calculate final focus position and spot size
    focus_position, focus_size = calculate_and_visualize_beam(np.abs(focus_spot), (dx_focus_x, dx_focus_y),
                                                              title="Focus Position and Size Analysis")
    # Export focus analysis results
    yield "output3", (focus_position, focus_size)
