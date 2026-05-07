# -*- coding: utf-8 -*-
from online_wfs.config import load_quick_config
from online_wfs.pipeline import task
from pprint import pprint


def main():
    """
    Example usage of the online_wfs package.
    """

    # STEP 0: load parameters from config file
    params = load_quick_config("examples/params.json")

    # STEP 1: Run the pipeline
    for checkpoint_name, result in task(params):
        if checkpoint_name == "focus_distances":
            print("Source distance (m)(X/Y):", result["R_x_m"], result["R_y_m"])

        if checkpoint_name == "beam_analysis":
            print(
                "Beam size (m)(X/Y):",
                result["beam_size"],
            )
            print(
                "Focus size (m)(X/Y):",
                result["focus_size"],
            )
        if checkpoint_name == "aberration_analysis":
            pprint(result["coefficient_table"])


if __name__ == "__main__":
    main()
