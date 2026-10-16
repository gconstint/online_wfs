"""Calibrate X/Y detector pixels before running main.py reconstruction.

Run from the repository root: python cal_pixel_size.py.
Defaults use the run11 slit reference and 0.48 um initial pixels, without crop.
Dark/flat corrections require paths supplied through the CLI or a config.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from online_wfs.calibration import PixelSizeCalibrationResult, calibrate_pixel_size
from tools.io import load_config, load_inputs


DEFAULT_IMAGE = Path("/Users/guanzhh/Vscode_space/wfs_sgi/data/run11/img_010.tif")


def build_parser() -> argparse.ArgumentParser:
    """Take a saved config or CLI inputs; explicit flags override config values."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, help="Optional saved experimental configuration JSON."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Slit-reference TIFF or NPZ ('image' array).",
    )
    parser.add_argument(
        "--energy-ev", type=float, default=10000.0, help="Photon energy in eV."
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        nargs="+",
        default=[0.48],
        help="Initial pixels in um: one value, or vertical horizontal (Y X).",
    )
    parser.add_argument(
        "--grating-period",
        type=float,
        default=18.38e-6,
        help="Nominal grating period in meters.",
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
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument(
        "--crop-indexes",
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Slit-image ROI [y0,y1,x0,x1]; stops are exclusive. Default: full image.",
    )
    crop.add_argument(
        "--no-crop",
        action="store_true",
        help="Process the full image even if the config specifies a crop.",
    )
    parser.add_argument(
        "--dark-image",
        type=Path,
        help="Optional dark TIFF or NPZ ('dark' array).",
    )
    parser.add_argument(
        "--flat-image", type=Path, help="Optional flat TIFF or NPZ ('flat' array)."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the detector ROI, FFT peaks and calibrated pixel sizes without saving files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> PixelSizeCalibrationResult:
    """Load/crop a reference, call the array API and print independent X/Y pixels."""
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    saved = {}
    if config_args.config:
        try:
            saved = load_config(config_args.config)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        parser.set_defaults(
            image=Path(saved["image_path"]) if saved["image_path"] else None,
            energy_ev=saved["p_energy"],
            pixel_size_um=[value * 1e6 for value in saved["pixel_size"]],
            grating_period=saved["g_period"],
            grating_angle=saved["g_angle"],
            source_grating_dist=saved["source_dist"],
            grating_detector_dist=saved["det2sample"],
            crop_indexes=saved["crop_indexes"],
            dark_image=Path(saved["dark_image_path"])
            if saved["dark_image_path"]
            else None,
            flat_image=Path(saved["flat_image_path"])
            if saved["flat_image_path"]
            else None,
        )
    args = parser.parse_args(argv)
    if args.image is None:
        parser.error("provide --image or image_path in the configuration")
    if len(args.pixel_size_um) not in (1, 2):
        parser.error("--pixel-size-um expects one value or two values in Y X order")
    pixels = [value * 1e-6 for value in args.pixel_size_um]
    config = {
        **saved,
        "p_energy": args.energy_ev,
        "pixel_size": pixels[0] if len(pixels) == 1 else pixels,
        "g_period": args.grating_period,
        "g_angle": args.grating_angle,
        "source_dist": args.source_grating_dist,
        "det2sample": args.grating_detector_dist,
        "image_path": str(args.image.expanduser().resolve()),
        "dark_image_path": str(args.dark_image.expanduser().resolve())
        if args.dark_image
        else None,
        "flat_image_path": str(args.flat_image.expanduser().resolve())
        if args.flat_image
        else None,
        "crop_indexes": None if args.no_crop else args.crop_indexes,
    }
    try:
        image, params, corrections = load_inputs(config)
        result = calibrate_pixel_size(
            image, params, **corrections, include_arrays=args.plot
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(result.format_report(), flush=True)
    if args.plot:
        try:
            from tools.plot_harmonic_peaks import plot_harmonic_peaks

            plot_harmonic_peaks(
                result,
                image_label=config["image_path"],
                crop_indexes=config["crop_indexes"],
                show=True,
            )
        except ImportError:
            parser.error("--plot requires matplotlib; install the 'plots' extra.")
    return result


if __name__ == "__main__":
    main()
