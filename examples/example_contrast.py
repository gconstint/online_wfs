"""Measure real-time fringe contrast for one frame."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from online_wfs.automation import measure_realtime_contrast

from ._image_io import SAMPLE_IMAGE, load_image


def build_parser() -> argparse.ArgumentParser:
    """Define image and geometry inputs without reading JSON configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure mean fringe contrast for one frame from an aligned 45-degree "
            "checkerboard grating. The real-time API uses a fixed source-to-grating "
            "distance of 800 m."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=SAMPLE_IMAGE,
        help="Detector TIFF or NPZ ('image' array).",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        nargs="+",
        default=[0.715],
        help="Effective detector pixel size in um: one value or vertical horizontal.",
    )
    parser.add_argument(
        "--grating-period",
        type=float,
        default=18.38e-6,
        help="Nominal checkerboard period in meters; divided by sqrt(2) internally.",
    )
    parser.add_argument(
        "--grating-detector-dist",
        "--det2sample",
        type=float,
        default=0.35,
        help="Grating-to-detector distance for this frame in meters.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also print the image path and grating-to-detector distance.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Measure and print contrast at the supplied distance."""
    parser = build_parser()
    args = parser.parse_args(argv)
    pixels_um = args.pixel_size_um
    if len(pixels_um) not in (1, 2):
        parser.error("--pixel-size-um requires one value or vertical horizontal.")
    pixel_size_m = (
        pixels_um[0] * 1e-6
        if len(pixels_um) == 1
        else (pixels_um[0] * 1e-6, pixels_um[1] * 1e-6)
    )
    try:
        image_path = args.image.expanduser().resolve()
        image = load_image(image_path)
        contrast = measure_realtime_contrast(
            image,
            grating_detector_dist_m=args.grating_detector_dist,
            pixel_size=pixel_size_m,
            grating_period=args.grating_period,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.verbose:
        print(f"Image path: {image_path}")
        print(f"Grating-to-detector distance: {args.grating_detector_dist} m")
    print(f"Mean fringe contrast: {contrast:.6%}")


if __name__ == "__main__":
    main()
