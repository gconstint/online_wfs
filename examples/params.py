# -*- coding: utf-8 -*-
"""
Parameter configurations for XGI Wavefront Sensor examples.

This module provides pre-configured parameter sets for different
experimental and simulation scenarios.
"""

from pathlib import Path

# Try to import calculate_wavelength from core, fall back to local implementation
try:
    from core import calculate_wavelength
except ImportError:
    from scipy import constants

    def calculate_wavelength(photon_energy):
        """Calculate the wavelength from photon energy."""
        hc = constants.value("inverse meter-electron volt relationship")
        return hc / photon_energy


# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def get_exp_params() -> dict:
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
    params["image_path"] = str(DATA_DIR / "sample_exp.tif")
    params["dark_image_path"] = None
    params["flat_image_path"] = None

    return params


def get_sim_params() -> dict:
    """
    Get parameters for simulation data analysis.

    Returns
    -------
    dict
        Configuration parameters for simulated XGI data
    """
    import numpy as np

    params = dict()

    # Detector Configuration
    params["pixel_size"] = [1.3e-6, 1.3e-6]  # Pixel size (m)
    params["wavelength"] = calculate_wavelength(5000)  # 5000 eV

    # Optical System Geometry
    params["det2sample"] = 2.803  # Grating-to-detector distance (m)
    params["total_dist"] = 3.0  # Source-to-detector distance (m)
    params["source_dist"] = params["total_dist"] - params["det2sample"]

    # Grating Parameters
    period = 4e-6  # Base grating period (m)
    params["grating_period"] = period / np.sqrt(2)  # 45° rotation
    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    # Data Source
    params["image_path"] = str(DATA_DIR / "sample_sim.tif")
    params["dark_image_path"] = None
    params["flat_image_path"] = None

    return params


def get_custom_params(
    pixel_size: float,
    photon_energy: float,
    det2sample: float,
    total_dist: float,
    grating_period: float,
    image_path: str,
    dark_image_path: str = None,
    flat_image_path: str = None,
) -> dict:
    """
    Create custom parameters for XGI analysis.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in meters
    photon_energy : float
        Photon energy in eV
    det2sample : float
        Grating-to-detector distance in meters
    total_dist : float
        Source-to-detector distance in meters
    grating_period : float
        Grating period in meters
    image_path : str
        Path to the sample image
    dark_image_path : str, optional
        Path to dark field image
    flat_image_path : str, optional
        Path to flat field image

    Returns
    -------
    dict
        Configuration parameters
    """
    params = dict()

    params["pixel_size"] = [pixel_size, pixel_size]
    params["wavelength"] = calculate_wavelength(photon_energy)
    params["det2sample"] = det2sample
    params["total_dist"] = total_dist
    params["source_dist"] = total_dist - det2sample
    params["grating_period"] = grating_period
    params["pattern_period"] = grating_period * total_dist / params["source_dist"]
    params["image_path"] = image_path
    params["dark_image_path"] = dark_image_path
    params["flat_image_path"] = flat_image_path

    return params
