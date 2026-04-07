from online_wfs.func.calculate_contrast import calculate_contrast
from online_wfs.config import load_quick_config

import numpy as np
from PIL import Image

if __name__ == "__main__":

    # load parameters
    params = load_quick_config("examples/params.json")
    
    # load image
    img = Image.open(params["image_path"]).convert("L")
    img = np.array(img).astype(np.uint16)

    # don't change the codes below
    pixel_size = params["pixel_size"]
    g_period = params["g_period"]
    det2sample = params["det2sample"] # this value may be changed along the z-scan.
    source_dist = params["source_dist"]

    # main function: calculate contrast
    contrast = calculate_contrast(
        img,
        pixel_size,
        g_period,
        source_dist,
        det2sample,
        search_region=None,
        plot_flag=True,
    )

    # TODO: use the contrast value for further analysis, e.g. plot contrast vs det2sample in a z-scan.
