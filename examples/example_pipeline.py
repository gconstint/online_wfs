"""Reconstruct one detector image: python -m examples.example_pipeline --help."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from online_wfs import Pipeline

from ._image_io import SAMPLE_IMAGE, load_image


def build_parser() -> argparse.ArgumentParser:
    """Define image and physical inputs without reading JSON configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an aligned 45-degree checkerboard detector image. "
            "Defaults use the packaged sample image and geometry. "
            "Calibration arrays must match the entire input image."
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
        "--dark-image", type=Path, help="Optional dark TIFF or NPZ ('dark' array)."
    )
    parser.add_argument(
        "--flat-image",
        type=Path,
        help="Optional flat TIFF or NPZ ('flat' array). Use the raw flat with a dark.",
    )
    parser.add_argument(
        "--energy-ev",
        type=float,
        default=7100,
        help="Photon energy in eV.",
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
        "--source-grating-dist",
        "--source-dist",
        type=float,
        default=6.5,
        help="Focus/source-to-grating distance in meters.",
    )
    parser.add_argument(
        "--grating-detector-dist",
        "--det2sample",
        type=float,
        default=0.35,
        help="Grating-to-detector distance in meters.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load one image, reconstruct with CLI geometry and display the results."""
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
        "source_dist": args.source_grating_dist,
        "det2sample": args.grating_detector_dist,
    }
    try:
        image = load_image(args.image)
        dark = load_image(args.dark_image, key="dark") if args.dark_image else None
        flat = load_image(args.flat_image, key="flat") if args.flat_image else None
        processor = Pipeline(params, dark=dark, flat=flat)
        result = processor.process(image)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Horizontal/vertical DPC shape:", result["dpc_x"].shape, result["dpc_y"].shape
    )
    print("Phase error RMS (rad):", np.nanstd(result["phase_error"]))
    print(
        "Residual phase RMS (rad):", np.sqrt(np.nanmean(result["residual_phase"] ** 2))
    )
    print(
        "Focus distance X/Y (m):",
        result["focus_distance_x_m"],
        result["focus_distance_y_m"],
    )


if __name__ == "__main__":
    main()
