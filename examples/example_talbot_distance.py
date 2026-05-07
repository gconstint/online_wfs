from online_wfs.func.calculate_talbot_distance import (
    calculate_wavelength,
    print_infos,
    print_talbot_distance,
)
from online_wfs.config import load_quick_config


def main():
    """Calculate and print Talbot distances using default parameters."""
    # --- Parameters (same as phase_talbot_distance.py) ---
    params = load_quick_config("examples/params.json")

    # --- Main calculation ---
    energy_ev = params["p_energy"]
    grating_period = params["grating_period"]
    wavelength = calculate_wavelength(energy_ev)
    eta = 1.0  # 1 if π/2 phase grating, 2 if π phase grating
    source_dist = params["source_dist"]
    talbot_orders = range(1, 10, 2)  # Odd orders: 1, 3, 5, ..., 29

    print_infos(energy_ev, wavelength, grating_period, eta, source_dist)
    print_talbot_distance(grating_period, wavelength, eta, source_dist, talbot_orders)
    print("=" * 65)


if __name__ == "__main__":
    main()
