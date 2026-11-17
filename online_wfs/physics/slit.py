"""Single-slit sizing in the paraxial Fraunhofer diffraction model.

For slit opening a, propagation distance z and wavelength lambda, the first
intensity minima occur at +/-y1, where y1 = lambda*z/a. The full main-lobe
width between these minima is 2*y1.

The existing sizing convention is retained: W = target_fraction * detector_FOV
is the center-to-first-minimum distance, giving a = lambda*z/W. It specifies a
diffraction length scale and does not enforce uniform illumination over W.
Use calculate_slit_size() to obtain all input and derived quantities together.
Energy is in eV; propagation distance and wavelength are in meters; detector
pixel size, transverse widths and slit opening are in micrometers.

Reference: OpenStax, University Physics Volume 3, sections 4.1-4.2.
https://openstax.org/books/university-physics-volume-3/pages/4-1-single-slit-diffraction
"""

from dataclasses import dataclass
from math import isfinite
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


def _opening_from_wavelength_um(
    wavelength: float, distance_m: float, target_width_um: float
) -> float:
    """Convert the first-minimum distance to an opening, using validated inputs."""
    width_m = target_width_um * 1e-6
    slit_m = wavelength * distance_m / width_m
    return slit_m * 1e6


@dataclass(frozen=True)
class SlitResult:
    """Physical inputs and derived detector/slit quantities for one detector axis."""

    energy_ev: float
    distance_m: float
    pixel_size_um: float
    resolution: int
    target_fraction: float
    wavelength: float
    detector_fov_um: float
    target_width_um: float
    slit_size_um: float

    @property
    def first_minimum_distance_um(self) -> float:
        """Return the center-to-first-minimum distance y1 [um]."""
        return self.target_width_um

    @property
    def main_lobe_width_um(self) -> float:
        """Return the full width between the two first minima [um]."""
        return 2.0 * self.first_minimum_distance_um

    def format_report(self) -> str:
        """Return the complete sizing report without printing or file I/O."""
        return "\n".join(
            (
                f"Energy                : {self.energy_ev:.3f} eV",
                f"Distance              : {self.distance_m:.3f} m",
                f"Wavelength            : {self.wavelength * 1e9:.6f} nm",
                f"Detector FOV          : {self.detector_fov_um:.2f} um",
                f"Target fraction       : {self.target_fraction:.0%}",
                f"Target width          : {self.target_width_um:.2f} um (center to first minimum)",
                f"Main-lobe full width  : {self.main_lobe_width_um:.2f} um",
                f"Slit size             : {self.slit_size_um:.2f} um",
            )
        )


def calculate_wavelength(photon_energy: float) -> float:
    """Calculate wavelength from photon energy in eV."""
    _require_positive_finite(photon_energy, "photon_energy")
    hc = constants.value("inverse meter-electron volt relationship")
    return hc / photon_energy


def slit_size_um(
    p_energy: float,
    distance_m: float,
    target_width_um: float,
) -> float:
    """
    Compute slit opening from the center-to-first-minimum distance W [um].

        a ~= lambda * z / W

    ``W`` is a one-sided diffraction scale, not the full main-lobe width.
    Photon energy is in eV; distance is in meters; the result is in um.
    """
    _require_positive_finite(distance_m, "distance_m")
    _require_positive_finite(target_width_um, "target_width_um")
    lam = calculate_wavelength(p_energy)
    return _opening_from_wavelength_um(lam, distance_m, target_width_um)


def slit_size_from_detector_um(
    p_energy: float,
    distance_m: float,
    pixel_size_um: float,
    resolution: int,
    target_fraction: float = 1.0,
) -> float:
    """Return the slit opening for W=target_fraction*detector_FOV, in um.

    W is the center-to-first-minimum distance. target_fraction is a positive
    FOV multiplier; values above 1 request a diffraction scale larger than
    the full detector FOV.
    """
    return calculate_slit_size(
        energy_ev=p_energy,
        distance_m=distance_m,
        pixel_size_um=pixel_size_um,
        resolution=resolution,
        target_fraction=target_fraction,
    ).slit_size_um


def first_minimum_distance_um(
    p_energy: float,
    distance_m: float,
    slit_um: float,
) -> float:
    """Return the center-to-first-minimum distance lambda*z/a [um]."""
    _require_positive_finite(distance_m, "distance_m")
    _require_positive_finite(slit_um, "slit_um")
    lam = calculate_wavelength(p_energy)
    slit_m = slit_um * 1e-6
    width_m = lam * distance_m / slit_m
    return width_m * 1e6


def main_lobe_width_um(
    p_energy: float,
    distance_m: float,
    slit_um: float,
) -> float:
    """Return the full first-minimum-to-first-minimum width 2*lambda*z/a [um]."""
    return 2.0 * first_minimum_distance_um(p_energy, distance_m, slit_um)


def calculate_slit_size(
    *,
    energy_ev: float,
    distance_m: float,
    pixel_size_um: float,
    resolution: int,
    target_fraction: float = 1.0,
) -> SlitResult:
    """Calculate a slit opening and its complete detector/diffraction metadata.

    Pass photon energy [eV], slit-to-detector distance [m], effective detector
    pixel size [um] and the positive integer pixel count for one detector axis.
    The positive target_fraction multiplies the detector FOV to define the
    center-to-first-minimum distance W. The opening is a=lambda*z/W [um].

    The result includes wavelength [m], detector FOV [um], W [um], opening
    [um] and full main-lobe width [um]. Call result.format_report() for text,
    or read its fields directly. Invalid inputs raise ValueError.
    """
    wavelength = calculate_wavelength(energy_ev)
    _require_positive_finite(distance_m, "distance_m")
    _require_positive_finite(pixel_size_um, "pixel_size_um")
    _require_positive_finite(target_fraction, "target_fraction")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, Integral)
        or resolution <= 0
    ):
        raise ValueError("resolution must be a positive integer pixel count.")

    detector_fov_um = pixel_size_um * resolution
    target_width_um = target_fraction * detector_fov_um
    opening_um = _opening_from_wavelength_um(wavelength, distance_m, target_width_um)
    return SlitResult(
        energy_ev=energy_ev,
        distance_m=distance_m,
        pixel_size_um=pixel_size_um,
        resolution=resolution,
        target_fraction=target_fraction,
        wavelength=wavelength,
        detector_fov_um=detector_fov_um,
        target_width_um=target_width_um,
        slit_size_um=opening_um,
    )
