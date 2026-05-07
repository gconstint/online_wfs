from scipy import constants


def calculate_wavelength(photon_energy: float) -> float:
    """Calculate wavelength from photon energy in eV."""
    if photon_energy <= 0:
        raise ValueError("photon_energy must be > 0")
    hc = constants.value("inverse meter-electron volt relationship")
    return hc / photon_energy


def slit_size_um(
    p_energy: float,
    distance_m: float,
    target_width_um: float,
) -> float:
    """
    Compute slit opening from the target width on the sensor:

        a ~= lambda * z / W

    where ``W`` is the desired illuminated width on the sensor.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if target_width_um <= 0:
        raise ValueError("target_width_um must be > 0")

    lam = calculate_wavelength(p_energy)
    width_m = target_width_um * 1e-6
    slit_m = lam * distance_m / width_m
    return slit_m * 1e6


def slit_size_from_detector_um(
    p_energy: float,
    distance_m: float,
    pixel_size_um: float,
    resolution: int,
    target_fraction: float = 1.0,
) -> float:
    """Compute slit opening from detector geometry and target illuminated fraction."""
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if target_fraction <= 0:
        raise ValueError("target_fraction must be > 0")

    detector_fov_um = pixel_size_um * resolution
    target_width_um = target_fraction * detector_fov_um
    return slit_size_um(p_energy, distance_m, target_width_um)


def main_lobe_width_um(
    p_energy: float,
    distance_m: float,
    slit_um: float,
) -> float:
    """Estimate the main-lobe width on the sensor from a slit opening."""
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if slit_um <= 0:
        raise ValueError("slit_um must be > 0")

    lam = calculate_wavelength(p_energy)
    slit_m = slit_um * 1e-6
    width_m = lam * distance_m / slit_m
    return width_m * 1e6



