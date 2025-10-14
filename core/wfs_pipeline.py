# load images
def pipeline(params):
    img, dark, flat = load_images(params['image_path'], params["dark_image_path"],
                                  params["flat_image_path"])  # in np.float32 for better performance
    # %%
    # Image normalization
    img = image_correction(img, flat=flat, dark=dark, epsilon=1e-8, normalize=False)
    # %%
    # Center-crop
    img = center_crop(img, target_size=2048)
    # %%

    # %%
    img32 = np.asarray(img, dtype=np.float32, order='C')
    img_fft = fftshift(fft2(img32, norm='ortho', workers=cpu_count()))
    # %%
    # Calculate initial harmonic periods
    harmonic_periods = calculate_harmonic_periods(
        img.shape,
        params['pixel_size'],
        params['pattern_period'],
    )
    # %%
    # Find accurate harmonic periods and peak positions
    harmonic_periods, peak_positions = accurate_harmonic_periods(img_fft, init_periods=harmonic_periods)
    # %%
    # Rotate image based on peak positions
    img, rotation_angle = rotate_image_by_peaks(img, peak_positions)
    # %%
    # Update params: estimate periods from X and Y, take the average for robustness
    p_x = params['pixel_size'][1] * img.shape[1] / harmonic_periods[1]
    p_y = params['pixel_size'][0] * img.shape[0] / harmonic_periods[0]
    params["pattern_period"] = float((p_x + p_y) / 2.0)

    params["harmonic_periods"] = harmonic_periods

    # %% md
    # ## Setup 2: grating analysis
    # ### Notes：
    # ### - analyze_grating_data: extract 00, 01, 10 harmonics and calculate DPC
    # ### - Compute magnification_grating from pattern_period and grating_period (should be > 1)
    # ### - Use calibrate_distance to adjust source_dist based on magnification_grating
    # ### - Use calculate_magnification_correction to apply scale factor to DPC
    # ### - preprocess_dpc: filter DPC images
    # ### - dpc_integration: integrate DPC to phase
    #
    # %%
    # Extract harmonics and calculate DPC
    results = analyze_grating_data(img_fft, params, plot_flag=False)
    int00, int01, int10, dpc_x, dpc_y, virtual_pixel_size, params = results
    # %%

    magnification_grating = params["pattern_period"] / params["grating_period"]
    assert magnification_grating > 1.0, f"magnification_grating({magnification_grating}) should be greater than 1.0"
    # %%
    # Adjust source_dist based on magnification_grating
    params = calibrate_distance(params, magnification_grating)
    # %%
    # calculate scale factor
    scale_factor = calculate_magnification_correction(params)
    # %%
    # Apply scale factor
    dpc_x *= scale_factor
    dpc_y *= scale_factor
    # Preprocess DPC images
    dpc_x_filt = preprocess_dpc(dpc_x)
    dpc_y_filt = preprocess_dpc(dpc_y)
    # %%
    # Convert DPC to physical units
    dpc_x_physical = dpc_x_filt * virtual_pixel_size[0]  # rad/m
    dpc_y_physical = dpc_y_filt * virtual_pixel_size[1]  # rad/m
    # Integrate DPC to phase
    phase = dpc_integration(dpc_x_physical, dpc_y_physical)

    # %% md
    # ## Setup 3.1. plot distance relationship
    # ### Visualize distance vs imaging geometry; validate magnification-based calibration
    # ### Use params['focus_adjust'] as a reference for focus position
    #
    # %%
    # Plot distance relationship
    # plot_distance_relationship(params)
    print("focus_adjust: ", params['focus_adjust'])

    # %% md
    # ## Setup 3.2. Wavefront fitting
    # ### Fit ideal spherical wavefront to reconstructed phase
    # ### - perform_wavefront_fitting: return fitted phase, error, and params
    # ### - plot_phase_fit_results: show original vs fitted and parameters
    # ### - plot_phase_error_profiles: show error profiles to diagnose aberrations
    #
    # %%
    # Wavefront fitting
    fitted_phase, phase_error, fit_params, = perform_wavefront_fitting(phase, virtual_pixel_size)
    # %%
    # Plotting
    # plot_phase_fit_results(phase, fitted_phase, fit_params, virtual_pixel_size, params['wavelength'])
    # %%
    # Plot error profiles in X and Y
    plot_phase_error_profiles(phase_error, virtual_pixel_size, params['wavelength'])

    # %% md
    # ## Setup 3.3. beam analysis
    # ### Estimate beam position and size on the self-imaging plane
    # ### calculate_and_visualize_beam: estimate centroid and effective size from intensity
    #
    # %%
    # Estimate beam position and size
    beam_position, beam_size = calculate_and_visualize_beam(int00, virtual_pixel_size,
                                                            title="self-image Position and Size Analysis")
    # print(beam_position, beam_size)
    # %% md
    # ## Setup 3.4. Focus analysis
    # ### Notes:
    # ### - analyze_focus_sampling_from_beam: compute focus sampling size
    # ### - two_steps_fresnel_method: backpropagate to evaluate focal spot
    # ### - calculate_and_visualize_beam: compute focus position and size
    #
    # %%
    # Calculate beam sampling size
    dx, dy = virtual_pixel_size[1], virtual_pixel_size[0]

    # calculate focus sampling size
    dx_focus_x, dx_focus_y, *_ = analyze_focus_sampling_from_beam(
        int00, dx, dy, params['wavelength'], params['total_dist'], beam_size)

    # Back propagate to focus
    beam = int00 * np.exp(1j * phase)
    focus_spot = two_steps_fresnel_method(beam, params['wavelength'], -params['total_dist'], dx, dy, 1)

    # Estimate focus position and size
    focus_position, focus_size = calculate_and_visualize_beam(np.abs(focus_spot), (dx_focus_x, dx_focus_y),
                                                              title="Focus Position and Size Analysis")

    return phase_error,
