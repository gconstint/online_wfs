# -*- coding: utf-8 -*-
"""
Parameter configurations for XGI Wavefront Sensor examples.

This module provides pre-configured parameter sets for different
experimental and simulation scenarios.
"""

from scipy import constants

def calculate_wavelength(photon_energy):
    """Calculate the wavelength from photon energy."""
    hc = constants.value("inverse meter-electron volt relationship")
    return hc / photon_energy


def get_params() -> dict:
    """
    Get parameters for experimental data analysis.

    Returns
    -------
    dict
        Configuration parameters for experimental XGI data
    """
    params = dict()

    # Detector Configuration
    params["pixel_size"] = [0.715e-6, 0.715e-6]  # Pixel size (m)
    params["wavelength"] = calculate_wavelength(7100)  # 7100 eV

    # Optical System Geometry
    params["det2sample"] = 0.35  # Grating-to-detector distance (m)
    params["total_dist"] = 6.5  # Source-to-detector distance (m)
    params["source_dist"] = params["total_dist"] - params["det2sample"]

    # Grating Parameters
    period = 18.38e-6  # Base grating period (m)
    params["grating_period"] = period / 2
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    # Data Source
    params["image_path"] = None
    params["dark_image_path"] = None
    params["flat_image_path"] = None

    # Pipeline Processing Parameters
    params["crop_size"] = 2048  # Center crop size (pixels)
    params["rotation_angle"] = 1.142  # Pre-computed rotation angle (degrees)
    params["lowpass_cutoff"] = 0.35  # DPC lowpass filter cutoff
    params["do_rotation"] = False  # Whether to perform rotation correction
    params["parallel"] = True  # Whether to use parallel execution

    return params
