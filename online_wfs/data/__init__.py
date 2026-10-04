"""Packaged measured detector frame and its physical configuration."""

import json
from importlib.resources import as_file, files

from ..config import load_config
from ..io import load_image


def load_sample():
    """Return (image, params) from package resources, independent of working directory."""
    resources = files(__package__)
    params = load_config(
        json.loads(resources.joinpath("params.json").read_text(encoding="utf-8"))
    )
    with as_file(resources.joinpath("sample.tif")) as path:
        image = load_image(path)
    return image, params
