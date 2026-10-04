"""Measure complete Pipeline calls using the measured image shipped in the package."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from online_wfs import Pipeline, __version__
from online_wfs.data import load_sample


def _positive_count(value):
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=_positive_count, default=5)
    parser.add_argument("--warmup", type=_positive_count, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Load one real frame once. No synthetic images or copies of a frame sequence.
    image, params = load_sample()
    image.setflags(write=False)
    processor = Pipeline(params)
    for _ in range(args.warmup):
        processor.process(image)
        Pipeline(params).process(image)
    measured = {"reused": [], "fresh": []}
    for index in range(args.repetitions):
        order = ("reused", "fresh") if index % 2 else ("fresh", "reused")
        results = {}
        for name in order:
            start = perf_counter()
            selected = processor if name == "reused" else Pipeline(params)
            results[name] = selected.process(image)
            measured[name].append((perf_counter() - start) * 1000)
        for key in (
            "dpc_x",
            "dpc_y",
            "phase_error",
            "residual_phase",
            "focus_distance_x_m",
            "focus_distance_y_m",
        ):
            np.testing.assert_allclose(
                results["reused"][key], results["fresh"][key], atol=1e-10
            )
    records = {
        key: {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "samples_ms": values,
        }
        for key, values in measured.items()
    }
    report = dict(
        date=datetime.now(timezone.utc).isoformat(),
        package_version=__version__,
        mode="absolute",
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        input_resource="online_wfs/data/sample.tif",
        input_dtype=str(image.dtype),
        detector_shape=image.shape,
        input_array_sha256=hashlib.sha256(
            memoryview(np.ascontiguousarray(image)).cast("B")
        ).hexdigest(),
        physical_parameters={
            key: params[key]
            for key in (
                "p_energy",
                "pixel_size",
                "source_dist",
                "det2sample",
                "g_period",
                "g_angle",
            )
        },
        repetitions=args.repetitions,
        warmup_frames=args.warmup,
        results=records,
        scope="CPU; repeated packaged measured TIFF; complete process calls; fresh includes setup; excludes TIFF loading, camera acquisition and GPU",
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
