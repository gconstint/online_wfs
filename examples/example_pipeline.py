# -*- coding: utf-8 -*-
from online_wfs.config import load_quick_config
from online_wfs.pipeline import task


def main():
    """
    Example usage of the online_wfs package.
    """

    # STEP 0: load parameters from config file
    params = load_quick_config("examples/params.json")

    # STEP 1: Run the pipeline
    for checkpoint_name, _ in task(
        params,
        verbose=params["verbose"],
        show_plots=params["show_plots"],
        rotation_angle=params["rotation_angle"],
        lowpass_cutoff=params["lowpass_cutoff"],
        parallel=params["parallel"],
    ):
        print(f"Checkpoint: {checkpoint_name}")

    # # default 
    # for checkpoint_name, _ in task(
    #     params,
    #     verbose=False,
    #     show_plots=False,
    #     rotation_angle=None,
    #     lowpass_cutoff=0.35,
    #     parallel=True,
    # ):
    #     print(f"Checkpoint: {checkpoint_name}")

if __name__ == "__main__":
    main()
