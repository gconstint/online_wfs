"""Plot detector data and four wavefront maps using physical coordinates."""

from pathlib import Path

import numpy as np

from online_wfs import load_config


def _extent(shape, sampling, bounds=None):
    """Pixel-edge coordinates for centers at (index - size/2) * sampling."""
    rows, columns = shape
    y0, y1, x0, x1 = (0, rows, 0, columns) if bounds is None else bounds
    py, px = sampling
    return [
        (x0 - columns / 2 - 0.5) * px * 1e6,
        (x1 - columns / 2 - 0.5) * px * 1e6,
        (y0 - rows / 2 - 0.5) * py * 1e6,
        (y1 - rows / 2 - 0.5) * py * 1e6,
    ]


def plot_results(image, result, params, path, *, show=False):
    """Save an overview; plotting and optional display remain outside the package.

    Result panels show the finite support bounding box plus one-pixel margins.
    Detector display is subsampled for rendering only; saved arrays are intact.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = load_config(params)
    shape = result["phase_error"].shape
    magnification = params["total_dist"] / params["source_dist"]
    sampling = tuple(
        params["pixel_size"][axis] * image.shape[axis] / shape[axis] / magnification
        for axis in (0, 1)
    )
    valid = np.isfinite(result["phase_error"])
    row, column = np.nonzero(valid)
    bounds = (
        max(0, row.min() - 1),
        min(shape[0], row.max() + 2),
        max(0, column.min() - 1),
        min(shape[1], column.max() + 2),
    )
    y0, y1, x0, x1 = bounds
    extent = _extent(shape, sampling, bounds)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), layout="constrained")
    detector_stride = max(1, int(np.ceil(max(image.shape) / 1200)))
    detector = axes[0, 0].imshow(
        image[::detector_stride, ::detector_stride],
        origin="lower",
        cmap="gray",
        extent=_extent(image.shape, params["pixel_size"]),
        interpolation="nearest",
    )
    axes[0, 0].set_title("Detector image")
    fig.colorbar(detector, ax=axes[0, 0], shrink=0.85, label="Detector counts")
    panels = (
        (axes[0, 1], "dpc_x", "Horizontal DPC", "rad/m"),
        (axes[0, 2], "dpc_y", "Vertical DPC", "rad/m"),
        (axes[1, 0], "phase_error", "Phase error", "rad"),
        (axes[1, 1], "residual_phase", "Residual phase", "rad"),
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ededed")
    for ax, key, title, unit in panels:
        values = result[key][y0:y1, x0:x1]
        limit = max(float(np.nanmax(np.abs(values))), np.finfo(float).eps)
        plotted = ax.imshow(
            values,
            origin="lower",
            cmap=cmap,
            extent=extent,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_title(title)
        bar = fig.colorbar(plotted, ax=ax, shrink=0.85, label=unit)
        bar.formatter.set_powerlimits((-3, 3))
        bar.update_ticks()
    for ax in (*axes[0], *axes[1, :2]):
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    focus = axes[1, 2]
    focus.axis("off")
    focus.text(
        0.08,
        0.78,
        "Distance to focus",
        fontsize=16,
        weight="semibold",
        transform=focus.transAxes,
    )
    focus.text(
        0.08,
        0.52,
        f"Horizontal: {result['focus_distance_x_m']:.6g} m\n\nVertical: {result['focus_distance_y_m']:.6g} m",
        fontsize=15,
        transform=focus.transAxes,
        va="center",
    )
    focus.text(
        0.08,
        0.22,
        "From grating plane\nGaussian beam estimate",
        fontsize=11,
        color="#555555",
        transform=focus.transAxes,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return {
        "result_shape": list(shape),
        "wavefront_sampling_yx_m": list(sampling),
        "map_bounds_y0_y1_x0_x1": [int(value) for value in bounds],
        "detector_display_stride": detector_stride,
    }
