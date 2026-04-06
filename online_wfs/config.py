import json

import numpy as np

from .core.utils import calculate_wavelength

def load_quick_config(json_path=None, params_dict=None):
    """Load and process quick-analysis configuration from a JSON file or dict.

    Args:
        json_path: Path to the JSON config file.
        params_dict: A pre-built parameter dictionary (e.g. from ipywidgets).
                     If provided, ``json_path`` is ignored.

    Returns:
        dict: Parameters dict with all derived fields populated.
    """
    if params_dict is not None:
        params = dict(params_dict)  # make a shallow copy to avoid mutating caller's dict
    elif json_path is not None:
        with open(json_path, "r") as f:
            params = json.load(f)
    else:
        raise ValueError("Either json_path or params_dict must be provided")

    # Derived parameters
    params["wavelength"] = calculate_wavelength(params["p_energy"])
    
    if "dx_final" in params:
        params["pixel_size"] = [params["dx_final"], params["dx_final"]]
    elif "pixel_size" in params:
        params["pixel_size"] = [params["pixel_size"], params["pixel_size"]]
    else:
        raise ValueError("Must provide either dx_final or pixel_size")

    params["total_dist"] = params["source_dist"] + params["det2sample"]

    # don't forget to set image paths
    # params["image_path"] = "xxx"
    # params["dark_image_path"] = None
    # params["flat_image_path"] = None

    # for pi/2 phase grating
    if params["g_angle"] == 45:
        params["grating_period"] = params["g_period"] / np.sqrt(2)
    else:
        params["grating_period"] = params["g_period"] / 2

    params["pattern_period"] = (
        params["grating_period"] * params["total_dist"] / params["source_dist"]
    )

    return params
