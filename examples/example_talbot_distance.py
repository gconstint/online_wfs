"""Calculate grating-to-detector distances; run with --help for physical inputs."""

import argparse
from collections.abc import Sequence

from online_wfs.physics.talbot import calculate_talbot_distances


def build_parser() -> argparse.ArgumentParser:
    """Define physical inputs directly, without a configuration file."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate grating-to-detector Talbot distances for a 2-D pi/2 "
            "checkerboard grating, with the focus-to-detector distance fixed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-detector-dist",
        type=float,
        default=154.2,
        help="Fixed focus/source-to-detector distance L in meters.",
    )
    parser.add_argument(
        "--energy-ev",
        type=float,
        default=9000.0,
        help="Photon energy in eV.",
    )
    parser.add_argument(
        "--grating-period",
        type=float,
        default=18.38e-6,
        help="Nominal checkerboard X/Y period in meters; divided by sqrt(2) internally.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        choices=(1.0,),
        default=1.0,
        help="Phase parameter: 1 for the pi/2 checkerboard grating.",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Positive odd fractional Talbot orders.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Pass parsed physical inputs to the calculator and display its report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = calculate_talbot_distances(
            energy_ev=args.energy_ev,
            grating_period=args.grating_period,
            source_detector_dist=args.source_detector_dist,
            eta=args.eta,
            talbot_orders=args.orders,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(result.format_report())


if __name__ == "__main__":
    main()
