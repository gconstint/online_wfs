"""Calculate Talbot distances: python -m examples.example_talbot_distance."""

from importlib.resources import files
import json

from online_wfs.config import load_config
from online_wfs.physics.talbot import (
    calculate_wavelength,
    print_infos,
    print_talbot_distance,
)


def main():
    """Use the packaged physical configuration without loading the detector image."""
    config = files("online_wfs.data").joinpath("params.json")
    params = load_config(json.loads(config.read_text(encoding="utf-8")))

    energy_ev = params["p_energy"]
    grating_period = params["grating_period"]
    wavelength = calculate_wavelength(energy_ev)
    eta = 1.0  # 1 for a pi/2 phase grating, 2 for a pi phase grating.
    source_dist = params["source_dist"]
    talbot_orders = range(1, 30, 2)  # Odd orders: 1, 3, 5, ..., 29

    print_infos(energy_ev, wavelength, grating_period, eta, source_dist)
    print_talbot_distance(grating_period, wavelength, eta, source_dist, talbot_orders)
    print("=" * 65)


if __name__ == "__main__":
    main()
