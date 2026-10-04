"""Analyze a detector image, draw the requested maps, and save a compact result."""

import argparse
import json
from pathlib import Path

import numpy as np

from online_wfs import Pipeline, load_config
from online_wfs.data import load_sample
from online_wfs.io import load_image
from tools.plot_results import plot_results

# All maps use grating-plane coordinates; focus distances are grating-to-waist magnitudes.
ARRAY_UNITS = {
    "dpc_x": "rad/m",
    "dpc_y": "rad/m",
    "phase_error": "rad",
    "residual_phase": "rad",
}


def save_result(result, output_dir):
    """Save four full arrays and a six-field JSON with file references and units."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "arrays.npz", **{key: result[key] for key in ARRAY_UNITS}
    )
    report = {
        key: {"file": "arrays.npz", "unit": unit} for key, unit in ARRAY_UNITS.items()
    }
    report.update(
        focus_distance_x_m=result["focus_distance_x_m"],
        focus_distance_y_m=result["focus_distance_y_m"],
    )
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="physical configuration JSON")
    parser.add_argument(
        "--image",
        type=Path,
        help="detector TIFF; defaults to the packaged sample or config image_path",
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
        params = load_config(args.config)
        image_path = args.image or params["image_path"]
        if not image_path:
            parser.error("provide --image or image_path in the configuration")
        image = load_image(Path(image_path).expanduser().resolve())
    else:
        image, params = load_sample()
        if args.image:
            image = load_image(args.image.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    result = Pipeline(params).process(image)
    save_result(result, output_dir)
    plot_results(image, result, params, output_dir / "results.png", show=args.show)
    print(f"Horizontal focus distance: {result['focus_distance_x_m']:.6g} m")
    print(f"Vertical focus distance: {result['focus_distance_y_m']:.6g} m")
    print(f"Figure: {output_dir / 'results.png'}")
    print(f"Arrays: {output_dir / 'arrays.npz'}")
    print(f"Result: {output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
