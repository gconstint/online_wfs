"""UNUSED: experimental DCT unwrapping with integer branch repair.

Status: UNUSED by the production Pipeline. The active reconstruction still
uses SupportedPoisson.unwrap (skimage); this module is not imported there.

Interface: DCTUnwrapper(support_mask).unwrap(phase), matching the existing
SupportedPoisson.unwrap(phase) mask, output, and integer-anchor conventions.

Dependencies: NumPy and SciPy only. This file can be copied and imported on
its own; it has no Pipeline, skimage, file I/O, or plotting dependencies.

``unwrap_ls`` solves continuous unweighted least squares. The hybrid
``unwrap_dct_branch_corrected`` additionally selects integer 2*pi branches;
it therefore changes the original LS problem. Agreement with skimage has
been verified for two measured inputs, not guaranteed for arbitrary inputs.
"""

from os import cpu_count

import numpy as np
from scipy.fft import dctn, idctn, next_fast_len
from scipy.ndimage import label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import depth_first_order, minimum_spanning_tree
from scipy.sparse.linalg import LinearOperator, cg

__all__ = ["DCTUnwrapper", "unwrap_ls", "unwrap_dct_branch_corrected"]


class _UnweightedPoisson:
    """Open-boundary graph Poisson solve, with a DCT inverse/preconditioner."""

    def __init__(self, support):
        self.support = np.asarray(support, bool).copy()
        if self.support.ndim != 2 or min(self.support.shape) < 3:
            raise ValueError("phase grid must be two-dimensional and at least 3x3")
        if self.support.sum() < 4 or label(self.support)[1] != 1:
            raise ValueError("support must contain at least four connected pixels")
        self.wx = (self.support[:, :-1] & self.support[:, 1:]).astype(float)
        self.wy = (self.support[:-1] & self.support[1:]).astype(float)
        self.rectangle = bool(self.support.all())
        h, w = self.support.shape
        self.transform_shape = (
            (h, w) if self.rectangle else (next_fast_len(h), next_fast_len(w))
        )
        self.workers = min(4, cpu_count() or 1)
        th, tw = self.transform_shape
        self.eigenvalues = (
            2
            - 2 * np.cos(np.pi * np.arange(th)[:, None] / th)
            + 2
            - 2 * np.cos(np.pi * np.arange(tw)[None, :] / tw)
        )
        self.eigenvalues[0, 0] = 1

    def _inverse(self, rhs):
        spectrum = (
            dctn(
                rhs, type=2, norm="ortho", s=self.transform_shape, workers=self.workers
            )
            / self.eigenvalues
        )
        spectrum[0, 0] = 0
        field = idctn(spectrum, type=2, norm="ortho", workers=self.workers)
        return field[: rhs.shape[0], : rhs.shape[1]]

    def solve(self, dx, dy):
        dx, dy = np.asarray(dx, float), np.asarray(dy, float)
        if dx.shape != self.wx.shape or dy.shape != self.wy.shape:
            raise ValueError("edge differences must match the phase grid")
        horizontal, vertical = self.wx > 0, self.wy > 0
        if not np.isfinite(dx[horizontal]).all() or not np.isfinite(dy[vertical]).all():
            raise ValueError("edge differences must be finite on support")
        hx = np.where(horizontal, dx, 0) * self.wx
        hy = np.where(vertical, dy, 0) * self.wy
        rhs = np.zeros(self.support.shape)
        rhs[:, :-1] -= hx
        rhs[:, 1:] += hx
        rhs[:-1] -= hy
        rhs[1:] += hy
        del dx, dy, hx, hy, horizontal, vertical
        if self.rectangle:
            return self._inverse(rhs)

        def multiply(values):
            field = np.zeros(self.support.shape)
            field[self.support] = values
            hx = np.diff(field, axis=1) * self.wx
            hy = np.diff(field, axis=0) * self.wy
            output = np.zeros_like(field)
            output[:, :-1] -= hx
            output[:, 1:] += hx
            output[:-1] -= hy
            output[1:] += hy
            return output[self.support] + values.mean()

        def precondition(values):
            field = np.zeros(self.support.shape)
            field[self.support] = values - values.mean()
            output = self._inverse(field)[self.support]
            return output - output.mean() + values.mean()

        n = int(self.support.sum())
        operator = LinearOperator((n, n), matvec=multiply, dtype=float)
        preconditioner = LinearOperator((n, n), matvec=precondition, dtype=float)
        values = rhs[self.support]
        values -= values.mean()
        solution, info = cg(
            operator, values, M=preconditioner, rtol=1e-9, atol=0, maxiter=2000
        )
        if info != 0:
            raise RuntimeError(
                f"masked Poisson reconstruction did not converge (cg={info})"
            )
        output = np.full(self.support.shape, np.nan)
        output[self.support] = solution - solution.mean()
        return output


class DCTUnwrapper:
    """UNUSED in Pipeline; phase-only unwrap interface with a stored mask.

    Construct with the measured support_mask. Calling unwrap(phase) returns
    the same-shaped float64 field in radians, with NaN outside support and
    the wrapped branch preserved at the center or nearest supported pixel.
    Wrapped phase must be finite and in [-pi, pi] on measured support.
    """

    def __init__(self, support_mask, *, poisson_refine=True):
        self.support = np.asarray(support_mask, dtype=bool).copy()
        if self.support.ndim != 2 or min(self.support.shape) < 3:
            raise ValueError("phase grid must be two-dimensional and at least 3x3")
        if self.support.sum() < 4 or label(self.support)[1] != 1:
            raise ValueError("support must contain at least four connected pixels")
        self.poisson_refine = poisson_refine

    def unwrap(self, phase):
        """Unwrap in 2-D using the stored support; inputs are not modified."""
        phase = np.asarray(phase, dtype=float)
        if phase.shape != self.support.shape:
            raise ValueError("wrapped phase must match the reconstruction grid")
        if not np.all(np.isfinite(phase[self.support])):
            raise ValueError("wrapped phase must be finite on measured support")
        if np.any(np.abs(phase[self.support]) > np.pi + 1e-12):
            raise ValueError("wrapped phase must lie in [-pi, pi] on measured support")
        # Preserve valid rectangle data for the DCT initializer; like the
        # existing masked interface, missing/invalid exterior data is ignored.
        valid = np.isfinite(phase) & (np.abs(phase) <= np.pi + 1e-12)
        field = np.where(valid, phase, 0)
        return unwrap_dct_branch_corrected(
            field, self.support, poisson_refine=self.poisson_refine
        )


def principal(value):
    """Principal difference for phases in [-pi, pi], preserving exact +/-pi."""
    return np.where(
        value > np.pi,
        value - 2 * np.pi,
        np.where(value < -np.pi, value + 2 * np.pi, value),
    )


def anchor_index(support):
    """Center pixel, or the nearest supported pixel if the center is excluded."""
    anchor = (support.shape[0] // 2, support.shape[1] // 2)
    if not support[anchor]:
        indices = np.argwhere(support)
        anchor = tuple(indices[np.argmin(np.sum((indices - anchor) ** 2, axis=1))])
    return anchor


def align_anchor(field, phase, support):
    anchor = anchor_index(support)
    return np.where(support, field + phase[anchor] - field[anchor], np.nan)


def _validate_inputs(phase, support):
    phase, support = np.asarray(phase, float), np.asarray(support, bool)
    if phase.ndim != 2 or min(phase.shape) < 3 or not np.isfinite(phase).all():
        raise ValueError("wrapped phase must be a finite rectangle of at least 3x3")
    if np.any(np.abs(phase) > np.pi + 1e-12):
        raise ValueError("wrapped phase must lie in [-pi, pi]")
    if support.shape != phase.shape or not support.any():
        raise ValueError("support must be nonempty and match the phase grid")
    return phase, support


def unwrap_ls(phase, support, *, masked=False):
    """Solve min ||D u - principal(D phase)||^2 with unit edge weights.

    With masked=False, the full finite rectangle is solved by a single DCT-II
    Poisson inverse, and the output is masked afterwards. With masked=True,
    only edges inside one connected support are used, requiring CG with a
    DCT preconditioner. The continuous solution need not rewrap to the input.
    """
    phase, support = _validate_inputs(phase, support)
    solver = _UnweightedPoisson(support if masked else np.ones(phase.shape, bool))
    field = solver.solve(
        principal(np.diff(phase, axis=1)), principal(np.diff(phase, axis=0))
    )
    return align_anchor(field, phase, support)


def project_congruence(field, phase, support, *, circular_gauge=False):
    """Project onto phase + integer*2*pi and preserve the measured anchor."""
    if circular_gauge:
        offset = np.angle(np.mean(np.exp(1j * (phase[support] - field[support]))))
        field = field + offset
    candidate = phase + 2 * np.pi * np.rint((field - phase) / (2 * np.pi))
    anchor = anchor_index(support)
    candidate -= 2 * np.pi * np.rint((candidate[anchor] - phase[anchor]) / (2 * np.pi))
    return np.where(support, candidate, np.nan)


def phase_curvature_quality(phase, support):
    """Four-direction wrapped curvature using the skimage 0.25.2 layout.

    Its interior calculation includes masked-fill=0 in the current reference
    implementation. Outer-frame scores here are deterministic placeholders;
    border random initialization and tied-edge ordering are not reproduced.
    """
    values = np.where(support, phase, 0)
    c = values[1:-1, 1:-1]
    h = principal(values[1:-1, :-2] - c) - principal(c - values[1:-1, 2:])
    v = principal(values[:-2, 1:-1] - c) - principal(c - values[2:, 1:-1])
    d1 = principal(values[:-2, :-2] - c) - principal(c - values[2:, 2:])
    d2 = principal(values[:-2, 2:] - c) - principal(c - values[2:, :-2])
    quality = np.full(phase.shape, 1e9)
    quality[1:-1, 1:-1] = h * h + v * v + d1 * d1 + d2 * d2
    return quality


def correct_integer_branches(initial, phase, support):
    """Repair congruent branches along a minimum-curvature spanning tree."""
    indices = np.flatnonzero(support)
    node_ids = np.full(support.size, -1, np.int32)
    node_ids[indices] = np.arange(indices.size, dtype=np.int32)
    node_ids = node_ids.reshape(support.shape)
    quality = phase_curvature_quality(phase, support)
    horizontal = support[:, :-1] & support[:, 1:]
    vertical = support[:-1] & support[1:]
    rows = np.concatenate((node_ids[:, :-1][horizontal], node_ids[:-1][vertical]))
    columns = np.concatenate((node_ids[:, 1:][horizontal], node_ids[1:][vertical]))
    scores = np.concatenate(
        (
            (quality[:, :-1] + quality[:, 1:])[horizontal],
            (quality[:-1] + quality[1:])[vertical],
        )
    )
    # Sparse graphs represent exact zeros as absent edges.
    scores = np.maximum(scores, np.finfo(float).tiny)
    graph = coo_matrix(
        (scores, (rows, columns)), shape=(indices.size, indices.size)
    ).tocsr()
    del rows, columns, scores, quality
    tree = minimum_spanning_tree(graph)
    del graph
    root = int(node_ids[anchor_index(support)])
    order, parent = depth_first_order(tree + tree.T, root, directed=False)
    if order.size != indices.size:
        raise ValueError("branch correction requires one connected measured support")
    del tree, node_ids
    wrapped = phase.ravel()[indices]
    start = initial.ravel()[indices]
    corrections = np.zeros(indices.size, np.int32)
    two_pi = 2 * np.pi
    for node in order[1:]:
        predecessor = parent[node]
        raw = wrapped[node] - wrapped[predecessor]
        target = raw - two_pi if raw > np.pi else raw + two_pi if raw < -np.pi else raw
        current = start[node] - start[predecessor]
        corrections[node] = corrections[predecessor] + int(
            np.rint((target - current) / two_pi)
        )
    field = np.full(phase.shape, np.nan)
    field.ravel()[indices] = start + two_pi * corrections
    return project_congruence(field, phase, support)


def unwrap_dct_branch_corrected(phase, support, *, poisson_refine=True):
    """Return branch-corrected unwrapped carrier phase in radians.

    phase: finite 2-D wrapped rectangle in [-pi, pi], including outside support.
    support: boolean mask with at least four 4-connected measured pixels.
    Output: float64 array, NaN outside support, preserving the wrapped branch
    at the center (or its nearest supported pixel). Inputs are not modified.

    The DCT initializer is followed by modulo projection and curvature-guided
    tree branch repair. Optional masked Poisson reconstruction uses corrected
    edge differences, including integer jumps, followed by final projection.
    No reference phase is used. Exact skimage agreement on arbitrary border
    pixels, reliability ties, or other library versions is not guaranteed.
    """
    phase, support = _validate_inputs(phase, support)
    # Validate support once before allocating the spanning tree.
    solver = _UnweightedPoisson(support)
    initial = project_congruence(
        unwrap_ls(phase, support), phase, support, circular_gauge=True
    )
    corrected = correct_integer_branches(initial, phase, support)
    if not poisson_refine:
        return corrected
    reconstructed = solver.solve(np.diff(corrected, axis=1), np.diff(corrected, axis=0))
    return project_congruence(
        align_anchor(reconstructed, phase, support), phase, support
    )
