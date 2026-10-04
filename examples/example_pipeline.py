"""Process the packaged sample: python -m examples.example_pipeline."""

import numpy as np

from online_wfs import Pipeline
from online_wfs.data import load_sample


def main():
    image, params = load_sample()
    processor = Pipeline(params)
    result = processor.process(image)
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
