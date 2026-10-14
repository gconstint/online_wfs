"""Calculate Talbot distances: python -m examples.example_talbot_distance."""

from online_wfs.physics.talbot import calculate_wavelength
from tools.io import load_sample_config
from tools.print_talbot import print_talbot_distance


def main():
    """Use the packaged physical configuration without loading the detector image."""
    params = load_sample_config()

    energy_ev = params["p_energy"]
    grating_period = params["grating_period"]
    wavelength = calculate_wavelength(energy_ev)
    eta = 1.0  # 1 for a pi/2 phase grating, 2 for a pi phase grating.
    source_dist = params["source_dist"]
    talbot_orders = range(1, 30, 2)  # Odd orders: 1, 3, 5, ..., 29

    print_talbot_distance(grating_period, wavelength, eta, source_dist, talbot_orders)


if __name__ == "__main__":
    main()
