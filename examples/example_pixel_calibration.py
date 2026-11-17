"""Calibrate detector pixels: python -m examples.example_pixel_calibration --help."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from online_wfs.calibration import PixelSizeCalibrationResult, calibrate_pixel_size

from ._image_io import load_image


def build_parser() -> argparse.ArgumentParser:
    """Define a slit-reference image and physical inputs without JSON files."""
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate X/Y detector pixels from a slit spherical-wave reference. "
            "Default geometry matches the run 11 slit reference with 0.48 um initial pixels. "
            "The entire input image is processed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Slit-reference TIFF or NPZ ('image' array).",
    )
    parser.add_argument(
        "--energy-ev",
        type=float,
        default=10000.0,
        help="Photon energy in eV.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        nargs="+",
        default=[0.48],
        help="Initial detector pixels in um: one value or vertical horizontal (Y X).",
    )
    parser.add_argument(
        "--grating-period",
        type=float,
        default=18.38e-6,
        help="Nominal checkerboard period in meters; divided by sqrt(2) at 45 degrees.",
    )
    parser.add_argument(
        "--grating-angle",
        type=float,
        default=45.0,
        help="Nominal grating angle in degrees.",
    )
    parser.add_argument(
        "--source-grating-dist",
        "--source-dist",
        type=float,
        default=154.2,
        help="Known slit-to-grating distance R in meters.",
    )
    parser.add_argument(
        "--grating-detector-dist",
        "--det2sample",
        type=float,
        default=0.681,
        help="Grating-to-detector distance z in meters.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> PixelSizeCalibrationResult:
    """Load one image, calibrate with CLI geometry and print the compact report."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.pixel_size_um) not in (1, 2):
        parser.error("--pixel-size-um requires one value or vertical horizontal.")
    pixels_um = (
        args.pixel_size_um[0] if len(args.pixel_size_um) == 1 else args.pixel_size_um
    )
    params = {
        "p_energy": args.energy_ev,
        "pixel_size": np.asarray(pixels_um) * 1e-6,
        "grating_period": args.grating_period,
        "grating_angle": args.grating_angle,
        "source_dist": args.source_grating_dist,
        "det2sample": args.grating_detector_dist,
    }
    try:
        image = load_image(args.image)
        result = calibrate_pixel_size(image, params)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(result.format_report())
    return result


if __name__ == "__main__":
    main()
