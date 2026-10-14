"""
Talbot distance calculation for spherical wave conditions.

This module provides core functions to calculate Talbot distances for phase
gratings under both plane wave and spherical wave conditions.

Main formulas:
1. Talbot distance under plane wave:
    Z_T0 = (1/eta²) * (n * d²) / (2 * λ)
2. Talbot distance under spherical wave:
    Z_T(R1) = (R1 * Z_T0) / (R1 - Z_T0)

Symbol definitions:
- d: grating period
- λ: wavelength
- eta: phase grating type parameter (eta=1: π/2 phase grating, eta=2: π phase grating)
- n: odd number sequence (1, 3, 5, ...)
- R1: distance from point source to grating
- Z_T0: fractional Talbot distance under plane wave conditions
- Z_T: Talbot distance under spherical wave conditions
"""

from scipy import constants


def calculate_wavelength(energy_ev: float) -> float:
    """
    Calculate wavelength from photon energy.

    Args:
        energy_ev: Photon energy in electron-volts (eV).

    Returns:
        Wavelength in meters (m).
    """
    hc = constants.value("inverse meter-electron volt relationship")
    return hc / energy_ev


def calculate_plane_wave_talbot_distance(
    grating_period: float, wavelength: float, eta: float, talbot_order: int
) -> float:
    """
    Calculate fractional Talbot distance for phase grating under plane wave conditions.

    Formula: Z_T0 = (1/eta²) * (n * d²) / (2 * λ)

    Args:
        grating_period: Grating period 'd' in meters (m).
        wavelength: Wavelength 'λ' in meters (m).
        eta: Phase grating parameter (1.0 for π/2 phase grating, 2.0 for π phase grating).
        talbot_order: Odd integer sequence 'n' (1, 3, 5, ...).

    Returns:
        Plane wave fractional Talbot distance 'Z_T0' in meters (m).

    Raises:
        ValueError: If talbot_order is not an odd number.
    """
    if talbot_order % 2 == 0:
        raise ValueError("Talbot order 'n' must be an odd number (1, 3, 5, ...).")

    z_t0 = (1 / eta**2) * (talbot_order * grating_period**2) / (2 * wavelength)
    return z_t0


def calculate_spherical_wave_talbot_distance(
    source_grating_dist: float, plane_wave_talbot_dist: float
) -> float:
    """
    Calculate Talbot distance under spherical wave conditions.
    This is the distance from the grating to the Talbot plane.

    Formula: Z_T(R1) = (R1 * Z_T0) / (R1 - Z_T0)

    Args:
        source_grating_dist: Distance from point source to grating 'R1' in meters (m).
        plane_wave_talbot_dist: Plane wave fractional Talbot distance 'Z_T0' in meters (m).

    Returns:
        Spherical wave Talbot distance 'Z_T' in meters (m).

    Raises:
        ValueError: If R1 <= Z_T0, which implies no valid real image formation downstream.
    """
    if source_grating_dist <= plane_wave_talbot_dist:
        raise ValueError("R1 must be greater than Z_T0 to ensure a valid solution.")

    z_t = (source_grating_dist * plane_wave_talbot_dist) / (
        source_grating_dist - plane_wave_talbot_dist
    )
    return z_t
