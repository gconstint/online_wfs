"""Grating-to-detector Talbot distances for a 2-D pi/2 checkerboard grating.

The geometry is focus/source -> grating -> detector: a divergent spherical wave
in the paraxial Fresnel approximation. For the specified 45-degree 2-D pi/2
checkerboard convention, nominal axial period P corresponds to effective
period d = P/sqrt(2), so the odd fractional Talbot distances are

    z0 = n * P**2 / (4 * wavelength),  n = 1, 3, 5, ...

All lengths are in meters. Only the specified pi/2 phase step (eta=1) is
supported. These are intensity-contrast planes. The period conversion
describes the specified checkerboard carrier convention.

With focus-to-grating distance R and grating-to-detector distance z, Fresnel
scaling gives z0 = R*z/(R+z). Thus z = R*z0/(R-z0) for fixed R. For a fixed
focus-to-detector distance L = R+z, the inverse problem is

    z = L/2 * (1 - sqrt(1 - 4*z0/L)),  R = L-z.

Only the shorter grating-to-detector solution is returned. It exists when
L >= 4*z0 and equals L/2 at equality. See Seaberg et al. (2019), equations (1)-(5):
https://doi.org/10.1107/S1600577519005721
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral, Real

from scipy import constants


def _require_positive_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


@dataclass(frozen=True)
class TalbotPosition:
    """One grating position that puts a fractional Talbot plane at the detector.

    All distances are in meters. ``grating_detector_dist`` is the requested
    Talbot propagation distance (``det2sample`` in the reconstruction
    configuration). ``source_grating_dist`` is the complementary distance
    from the focus to the grating (``source_dist``).
    """

    talbot_order: int
    plane_wave_talbot_dist: float
    source_grating_dist: float
    grating_detector_dist: float

    @property
    def magnification(self) -> float:
        """Return the geometric detector-to-grating magnification L/R."""
        return 1.0 + self.grating_detector_dist / self.source_grating_dist


@dataclass(frozen=True)
class TalbotOrderResult:
    """One requested order, including unreachable orders with no positions.

    There is one position for the shorter grating-to-detector solution, or
    none if the order is unreachable. All distances are in meters.
    """

    talbot_order: int
    plane_wave_talbot_dist: float
    positions: tuple[TalbotPosition, ...]

    @property
    def minimum_source_detector_dist(self) -> float:
        """Return the minimum total distance required to reach this order."""
        return 4.0 * self.plane_wave_talbot_dist

    @property
    def grating_detector_distances(self) -> tuple[float, ...]:
        """Return the shorter propagation distance, or an empty tuple."""
        return tuple(position.grating_detector_dist for position in self.positions)


@dataclass(frozen=True)
class TalbotResult:
    """Physical inputs, derived quantities and all requested order results.

    Energy is in eV; wavelength, periods and distances are in meters.
    """

    energy_ev: float
    grating_period: float
    source_detector_dist: float
    eta: float
    wavelength: float
    effective_grating_period: float
    order_results: tuple[TalbotOrderResult, ...]

    def format_report(self) -> str:
        """Build the complete distance report without printing or file I/O."""
        lines = [
            "Geometry: focus -> grating -> detector (divergent spherical wave)",
            f"Energy: {self.energy_ev:g} eV; wavelength: {self.wavelength * 1e9:.6f} nm; "
            f"eta: {self.eta:g}",
            f"Grating: 2-D pi/2 checkerboard; nominal period: {self.grating_period * 1e6:g} um; "
            f"effective period: {self.effective_grating_period * 1e6:.6f} um",
            f"Fixed focus-to-detector distance L: {self.source_detector_dist:g} m",
            "Primary result: grating-to-detector z; focus-to-grating R = L-z",
            f"{'Order':<7} {'grating-detector (m)':>22} "
            f"{'focus-grating (m)':>20} {'Z_T0 (m)':>12} {'M':>12}",
        ]
        for result in self.order_results:
            order, z0 = result.talbot_order, result.plane_wave_talbot_dist
            if not result.positions:
                lines.append(
                    f"{order:<7} {'-':>22} "
                    f"{'-':>20} {z0:>12.6f} {'-':>12}  "
                    f"no solution; requires L >= {result.minimum_source_detector_dist:.6f} m"
                )
                continue
            position = result.positions[0]
            lines.append(
                f"{order:<7} {position.grating_detector_dist:>22.6f} "
                f"{position.source_grating_dist:>20.6f} {z0:>12.6f} "
                f"{position.magnification:>12.6f}"
            )
        return "\n".join(lines)


def calculate_wavelength(energy_ev: float) -> float:
    """
    Calculate wavelength from photon energy.

    Args:
        energy_ev: Photon energy in electron-volts (eV).

    Returns:
        Wavelength in meters (m).
    """
    _require_positive_finite(energy_ev, "energy_ev")
    hc = constants.value("inverse meter-electron volt relationship")
    return hc / energy_ev


def calculate_effective_grating_period(grating_period: float) -> float:
    """Convert the nominal grating period to the period in the Talbot formula.

    Use the 45-degree 2-D checkerboard convention: nominal X/Y period divided
    by sqrt(2).
    Pass the nominal period, e.g. 18.38e-6 m, without dividing it beforehand.
    """
    _require_positive_finite(grating_period, "grating_period")
    return grating_period / sqrt(2.0)


def calculate_plane_wave_talbot_distance(
    grating_period: float,
    wavelength: float,
    eta: float,
    talbot_order: int,
) -> float:
    """
    Calculate the plane-wave fractional distance for a 2-D pi/2 checkerboard.

    Formula: Z_T0 = n * grating_period**2 / (4 * wavelength).

    Args:
        grating_period: Nominal grating period in meters (m).
        wavelength: Wavelength 'λ' in meters (m).
        eta: Must be 1.0 for the pi/2 phase step at the given energy.
        talbot_order: Odd integer sequence 'n' (1, 3, 5, ...).

    Returns:
        Plane wave fractional Talbot distance 'Z_T0' in meters (m).

    Raises:
        ValueError: If lengths are not finite and positive, eta is not 1,
            or talbot_order is not a positive odd integer.
    """
    effective_period = calculate_effective_grating_period(grating_period)
    _require_positive_finite(wavelength, "wavelength")
    _require_positive_finite(eta, "eta")
    if eta != 1:
        raise ValueError(
            "The checkerboard model supports a pi/2 phase step (eta=1) only."
        )
    if (
        isinstance(talbot_order, bool)
        or not isinstance(talbot_order, Integral)
        or talbot_order <= 0
        or talbot_order % 2 == 0
    ):
        raise ValueError("talbot_order must be a positive odd integer (1, 3, 5, ...).")

    z_t0 = (talbot_order * effective_period**2) / (2 * wavelength)
    return z_t0


def calculate_spherical_wave_talbot_distance(
    source_grating_dist: float, plane_wave_talbot_dist: float
) -> float:
    """
    Calculate Talbot distance under spherical wave conditions.
    This is the distance from the grating to the Talbot plane.

    Formula: Z_T(R1) = (R1 * Z_T0) / (R1 - Z_T0)

    Args:
        source_grating_dist: Distance from point source to grating 'R1' in meters (m).
        plane_wave_talbot_dist: Plane wave fractional Talbot distance 'Z_T0' in meters (m).

    Returns:
        Spherical wave Talbot distance 'Z_T' in meters (m).

    Raises:
        ValueError: If either distance is not finite and positive, or R1 <= Z_T0.
            The latter has no finite downstream plane for this positive order.
    """
    _require_positive_finite(source_grating_dist, "source_grating_dist")
    _require_positive_finite(plane_wave_talbot_dist, "plane_wave_talbot_dist")
    if source_grating_dist <= plane_wave_talbot_dist:
        raise ValueError("R1 must be greater than Z_T0 to ensure a valid solution.")

    z_t = (source_grating_dist * plane_wave_talbot_dist) / (
        source_grating_dist - plane_wave_talbot_dist
    )
    return z_t


def calculate_grating_detector_distances(
    source_detector_dist: float, plane_wave_talbot_dist: float
) -> tuple[float, ...]:
    """Return the shorter grating-to-detector distance for a fixed total L.

    All distances are in meters. Solve z*(L-z)/L = z0, where z0 is the plane-
    wave fractional Talbot distance. The returned tuple contains only the
    shorter solution, placing the grating closer to the detector. At
    L=4*z0, return L/2; at L<4*z0, return an empty tuple. Invalid inputs
    raise ValueError.
    """
    _require_positive_finite(source_detector_dist, "source_detector_dist")
    _require_positive_finite(plane_wave_talbot_dist, "plane_wave_talbot_dist")
    discriminant = 1.0 - 4.0 * (plane_wave_talbot_dist / source_detector_dist)
    if discriminant < 0:
        return ()

    # Rationalize the small root to avoid cancellation in 1-sqrt(...).
    short = 2.0 * plane_wave_talbot_dist / (1.0 + sqrt(discriminant))
    return (short,)


def _positions_from_talbot_distance(
    source_detector_dist: float, plane_wave_talbot_dist: float, talbot_order: int
) -> tuple[TalbotPosition, ...]:
    """Build the selected position from an already calculated order distance."""
    distances = calculate_grating_detector_distances(
        source_detector_dist, plane_wave_talbot_dist
    )
    return tuple(
        TalbotPosition(
            talbot_order=talbot_order,
            plane_wave_talbot_dist=plane_wave_talbot_dist,
            source_grating_dist=source_detector_dist - detector_dist,
            grating_detector_dist=detector_dist,
        )
        for detector_dist in distances
    )


def calculate_talbot_distances(
    *,
    energy_ev: float,
    grating_period: float,
    source_detector_dist: float,
    eta: float = 1.0,
    talbot_orders: Iterable[int] = (1, 3, 5),
) -> TalbotResult:
    """Calculate all requested grating-to-detector distances from physical inputs.

    This is the main API for the 2-D pi/2 checkerboard model. Pass photon
    energy [eV], nominal grating period [m] and fixed focus-to-detector
    distance [m]. Wavelength, effective period (nominal/sqrt(2)), per-order
    fractional distances, shorter grating-to-detector solutions and magnifications are
    calculated internally. ``eta`` must be 1; orders must be positive odd
    integers and default to 1, 3, 5.

    Returns:
        A result with common physical metadata and one ``order_results``
        entry per requested order, preserving input order. Each entry has
        zero or one position for the shorter grating-to-detector solution.
        Unreachable orders retain their plane-wave distance and minimum
        total distance. Call ``result.format_report()`` for the full text
        report, or read the structured fields directly.

    Raises:
        ValueError: If physical inputs or orders are invalid, or no orders
            are requested.
    """
    _require_positive_finite(source_detector_dist, "source_detector_dist")
    wavelength = calculate_wavelength(energy_ev)
    effective_period = calculate_effective_grating_period(grating_period)
    try:
        orders = tuple(talbot_orders)
    except TypeError as exc:
        raise ValueError(
            "talbot_orders must be an iterable of positive odd integers."
        ) from exc
    if not orders:
        raise ValueError(
            "talbot_orders must contain at least one positive odd integer."
        )

    order_results = []
    for order in orders:
        z0 = calculate_plane_wave_talbot_distance(
            grating_period, wavelength, eta, order
        )
        positions = _positions_from_talbot_distance(source_detector_dist, z0, order)
        order_results.append(
            TalbotOrderResult(
                talbot_order=order,
                plane_wave_talbot_dist=z0,
                positions=positions,
            )
        )
    return TalbotResult(
        energy_ev=energy_ev,
        grating_period=grating_period,
        source_detector_dist=source_detector_dist,
        eta=eta,
        wavelength=wavelength,
        effective_grating_period=effective_period,
        order_results=tuple(order_results),
    )
