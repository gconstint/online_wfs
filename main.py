"""Analyze a detector image, draw the requested maps, and save a compact result."""

import argparse
from pathlib import Path

from online_wfs import Pipeline
from tools.io import load_image, load_inputs, load_sample
from tools.plot_results import plot_results
from tools.save_results import save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="physical configuration JSON")
    parser.add_argument(
        "--image",
        type=Path,
        help="detector TIFF or NPZ bundle; defaults to the packaged sample or config image_path",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "output"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the figure in a local plotting window",
    )
    args = parser.parse_args()
    if args.config:
        image = load_image(args.image) if args.image else None
        image, params, calibration = load_inputs(args.config, image=image)
    else:
        image, params = load_sample()
        calibration = {}
        if args.image:
            image = load_image(args.image.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    result = Pipeline(params, **calibration).process(image)
    save_result(result, output_dir)
    plot_results(image, result, params, output_dir / "results.png", show=args.show)
    print(
        f"Focus X/Y: {result['focus_distance_x_m']:.6f} / {result['focus_distance_y_m']:.6f} m"
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
