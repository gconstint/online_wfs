"""Measure sample fringe contrast: python -m examples.example_contrast."""

from online_wfs.physics.contrast import calculate_contrast
from tools.io import load_sample


def main():
    image, params = load_sample()
    # Use the effective grating period derived from the physical grating angle.
    contrast = calculate_contrast(
        image,
        params["pixel_size"],
        params["grating_period"],
        params["source_dist"],
        params["det2sample"],
        search_region=None,
    )
    print(f"Mean fringe contrast: {contrast:.6g}")


if __name__ == "__main__":
    main()
