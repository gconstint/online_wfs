# External benchmark on packaged measured data

From the repository root, run:

```bash
python -m benchmarks.benchmark_online --output output/benchmark.json
```

The script reads `online_wfs/data/sample.tif` and its matching physical
configuration through package resources. It works with an installed package
without depending on the working directory's data paths. The measured frame
is 5068×5064, uint16. No images are generated or modified.

One TIFF load occurs before measurement. By default each path has two warmups
and five measured repetitions, with alternating reused/fresh order. A reused
processor retains aperture/Zernike state; a fresh call includes processor setup.
For a shorter check:

```bash
python -m benchmarks.benchmark_online --warmup 1 --repetitions 3
```

The report records the actual repetition counts, physical parameters, frame
shape/dtype/fingerprint, package version, environment, raw durations, median and
p95. Output equality is checked for DPC, phase, post-Zernike residual phase
and focus distances. This repeats one static measured frame and excludes TIFF
loading, acquisition and GPU execution. Production modules contain no clocks
or timing output fields.
