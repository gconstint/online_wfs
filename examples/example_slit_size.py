from online_wfs.func.calculate_slit_size import (
    slit_size_from_detector_um,
    calculate_wavelength,
)

if __name__ == "__main__":
    p_energy = 9000  # photon energy in eV
    distance_m = 154.2  # slit -> sensor distance
    pixel_size_um = 0.48
    resolution = 5048
    target_fraction = 1.0

    wavelength_nm = calculate_wavelength(p_energy) * 1e9
    detector_fov_um = pixel_size_um * resolution
    target_width_um = target_fraction * detector_fov_um
    slit_um = slit_size_from_detector_um(
        p_energy,
        distance_m,
        pixel_size_um,
        resolution,
        target_fraction,
    )

    print(f"Energy                : {p_energy:.3f} eV")
    print(f"Distance              : {distance_m:.3f} m")
    print(f"Wavelength            : {wavelength_nm:.6f} nm")
    print(f"Detector FOV          : {detector_fov_um:.2f} um")
    print(f"Target fraction       : {target_fraction:.0%}")
    print(f"Target width          : {target_width_um:.2f} um")
    print(f"Slit size             : {slit_um:.2f} um")
