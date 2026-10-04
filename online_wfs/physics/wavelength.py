"""Photon energy to wavelength in SI units."""

from scipy import constants


def calculate_wavelength(photon_energy):
    """Convert positive photon energy [eV] to wavelength [m]."""
    return constants.value("inverse meter-electron volt relationship") / photon_energy
