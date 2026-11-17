"""Calculate a slit opening: python -m examples.example_slit_size --help."""

import argparse
from collections.abc import Sequence

from online_wfs.physics.slit import calculate_slit_size


def build_parser() -> argparse.ArgumentParser:
    """Define physical parameters without a JSON configuration file."""
    parser = argparse.ArgumentParser(
        description="Calculate a single-slit opening from detector geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--energy-ev",
        type=float,
        default=9000.0,
        help="Photon energy in eV.",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        default=154.2,
        help="Slit-to-detector propagation distance in meters.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=0.48,
        help="Effective detector pixel size in micrometers.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=5048,
        help="Pixel count along the detector axis being sized.",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=1.0,
        help="Positive FOV multiplier setting the center-to-first-minimum distance.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Pass physical inputs to the calculator and display its report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = calculate_slit_size(
            energy_ev=args.energy_ev,
            distance_m=args.distance_m,
            pixel_size_um=args.pixel_size_um,
            resolution=args.resolution,
            target_fraction=args.target_fraction,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(result.format_report())


if __name__ == "__main__":
    main()
