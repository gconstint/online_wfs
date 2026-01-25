"""
Undulator Source Distance Calculation Module

This module provides functions to calculate the Undulator source distance
using the Gaussian (thin lens) imaging formula based on SGI-measured focus position.

Optical Layout:
    Undulator → CRL → Focus → Grating → Detector
       |         |       |       |         |
       |<-L_src->|       |       |         |
                 |<-L_f->|       |         |
                         |<---- R -------->|

Where:
    - L_source: Undulator to CRL distance (object distance)
    - L_focus: CRL to focus distance (image distance)
    - R: Detector to focus distance (measured by SGI)
"""

from typing import Dict, Any, Optional


# CRL material constants (electron density related parameters for common materials)
# Format: material -> (Z/A ratio, density [g/cm³])
CRL_MATERIALS = {
    "Be": (4 / 9.012, 1.85),  # Beryllium
    "Al": (13 / 26.982, 2.70),  # Aluminum
    "C": (6 / 12.011, 2.26),  # Carbon (diamond)
    "Si": (14 / 28.086, 2.33),  # Silicon
}


def calculate_delta_from_energy(
    energy_eV: float,
    material: str = "Be",
    custom_density: Optional[float] = None,
) -> float:
    """
    Calculate refractive index decrement δ based on photon energy and material.

    Uses simplified formula (valid far from absorption edges):
        δ ≈ 2.701e-6 × (ρ [g/cm³]) × (Z/A) × (λ [Å])²

    Parameters
    ----------
    energy_eV : float
        Photon energy [eV]
    material : str, optional
        CRL material, one of "Be", "Al", "C", "Si". Default is "Be".
    custom_density : float, optional
        Custom density [g/cm³].

    Returns
    -------
    float
        Refractive index decrement δ (dimensionless)
    """
    if material not in CRL_MATERIALS:
        raise ValueError(
            f"Unknown material '{material}'. Available materials: {list(CRL_MATERIALS.keys())}"
        )

    z_over_a, density = CRL_MATERIALS[material]
    if custom_density is not None:
        density = custom_density

    # Calculate wavelength [Å]
    hc = 12398.419  # eV·Å
    wavelength_angstrom = hc / energy_eV

    # δ ≈ 2.701e-6 × ρ × (Z/A) × λ²
    delta = 2.701e-6 * density * z_over_a * (wavelength_angstrom**2)

    return delta


def calculate_crl_focal_length(
    R: float,
    N: int,
    delta: Optional[float] = None,
    energy_eV: Optional[float] = None,
    material: str = "Be",
    verbose: bool = True,
) -> float:
    """
    Calculate CRL focal length: f = R / (2 × N × δ)

    Parameters
    ----------
    R : float
        Radius of curvature of a single lens [m]
    N : int
        Number of lenses
    delta : float, optional
        Refractive index decrement δ. If not provided, calculated from energy_eV and material.
    energy_eV : float, optional
        Photon energy [eV]. Required when delta is not provided.
    material : str, optional
        CRL material. Default is "Be".
    verbose : bool, optional
        Whether to print detailed information.

    Returns
    -------
    float
        CRL focal length [m]
    """
    if delta is None:
        if energy_eV is None:
            raise ValueError(
                "Must provide either delta or energy_eV to calculate focal length"
            )
        delta = calculate_delta_from_energy(energy_eV, material)

    f = R / (2 * N * delta)

    if verbose:
        print("\n" + "-" * 50)
        print("CRL Focal Length Calculation")
        print("-" * 50)
        print(f"  Radius of curvature R:     {R * 1e6:.1f} μm")
        print(f"  Number of lenses N:        {N}")
        print(f"  Refractive index δ:        {delta:.3e}")
        if energy_eV is not None:
            print(
                f"  Photon energy:             {energy_eV:.0f} eV ({energy_eV / 1000:.2f} keV)"
            )
            print(f"  Material:                  {material}")
        print(f"  Calculated focal length f: {f * 1e3:.3f} mm ({f:.6f} m)")
        print("-" * 50 + "\n")

    return f


def calculate_source_distance(
    f: float,
    z_CRL: float,
    z_focus: Optional[float] = None,
    z_det: Optional[float] = None,
    R_measured: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate Undulator source distance using Gaussian imaging formula from SGI-measured focus position.

    Optical layout: Undulator → CRL → Focus → Grating → Detector

    Calculation steps:
    1. z_focus = z_det - R_measured (focus is upstream of detector)
    2. L_focus = z_focus - z_CRL
    3. L_source = f × L_focus / (L_focus - f)

    Parameters
    ----------
    f : float
        CRL focal length [m]
    z_CRL : float
        CRL position [m]
    z_focus : float, optional
        Focus position [m]. If not provided, calculated from z_det and R_measured.
    z_det : float, optional
        Detector position [m]
    R_measured : float, optional
        SGI-measured detector-to-focus distance [m]

    Returns
    -------
    dict
        Contains L_source, L_focus, z_focus, z_source, f
    """
    # Step 1: Calculate focus position
    if z_focus is not None:
        pass
    elif z_det is not None and R_measured is not None:
        z_focus = z_det - R_measured
    else:
        raise ValueError("Must provide z_focus, or both z_det and R_measured")

    # Step 2: Calculate image distance
    L_focus = z_focus - z_CRL

    if L_focus <= f:
        raise ValueError(f"Physical error: L_focus ({L_focus:.6f} m) <= f ({f:.6f} m).")

    # Step 3: Calculate source distance
    L_source = (f * L_focus) / (L_focus - f)
    z_source = z_CRL - L_source

    if verbose:
        print("\n" + "=" * 70)
        print(
            "Undulator Source Distance Calculation (Gaussian Imaging Formula)".center(
                70
            )
        )
        print("=" * 70)
        print("Input parameters:")
        print(f"  - CRL focal length f:      {f * 1e3:.3f} mm ({f:.6f} m)")
        print(f"  - CRL position z_CRL:      {z_CRL:.6f} m")
        print(f"  - Focus position z_focus:  {z_focus:.6f} m")
        if z_det is not None and R_measured is not None:
            print(
                f"    (calculated from z_det={z_det:.3f}m, R_measured={R_measured:.6f}m)"
            )
        print("-" * 70)
        print("Calculation results:")
        print("  - Image distance L_focus = z_focus - z_CRL:")
        print(f"      {L_focus * 1e3:.6f} mm ({L_focus:.6f} m)")
        print("  - L_focus - f (should be positive and small):")
        print(f"      {(L_focus - f) * 1e6:.3f} μm ({(L_focus - f) * 1e3:.6f} mm)")
        print("  - Object distance L_source (Undulator to CRL):")
        print(f"      {L_source:.3f} m")
        print("  - Source absolute position z_source = z_CRL - L_source:")
        print(f"      {z_source:.6f} m")
        print("=" * 70 + "\n")

    return {
        "L_source": L_source,
        "L_focus": L_focus,
        "z_focus": z_focus,
        "z_source": z_source,
        "f": f,
    }


def calculate_undulator_source_distance(
    calibration_result: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    High-level wrapper function: Extract parameters from calibration_result and params to calculate Undulator source distance.

    Optical layout: Undulator → CRL → Focus → Grating → Detector

    Required params parameters:
    - Required: crl_position
    - Focal length: crl_focal_length or (crl_radius + crl_lens_count)
    - Optional: detector_position, crl_material, wavelength

    Parameters
    ----------
    calibration_result : dict
        Result from calibrate_focus_position, containing "R" (detector-to-focus distance)
    params : dict
        System parameters
    verbose : bool
        Whether to print detailed information

    Returns
    -------
    dict or None
        Dictionary containing source distance calculation results, or None if required parameters are missing
    """
    # Check required parameters
    has_crl_position = "crl_position" in params
    has_focal_length = "crl_focal_length" in params
    can_calc_focal_length = "crl_radius" in params and "crl_lens_count" in params

    if not has_crl_position:
        if verbose:
            print(
                "Info: crl_position not provided, skipping source distance calculation"
            )
        return None

    if not (has_focal_length or can_calc_focal_length):
        if verbose:
            print(
                "Info: crl_focal_length or (crl_radius + crl_lens_count) not provided, skipping source distance calculation"
            )
        return None

    # Calculate or get focal length
    if has_focal_length:
        crl_focal_length = params["crl_focal_length"]
    else:
        crl_material = params.get("crl_material", "Be")
        # Calculate energy from wavelength
        hc = 12398.419e-10  # eV·m
        energy_eV = hc / params["wavelength"]

        crl_focal_length = calculate_crl_focal_length(
            R=params["crl_radius"],
            N=params["crl_lens_count"],
            energy_eV=energy_eV,
            material=crl_material,
            verbose=verbose,
        )

    # Get detector position
    if "detector_position" in params:
        z_det = params["detector_position"]
    else:
        if verbose:
            print(
                "Warning: detector_position not provided, estimating using crl_position + total_dist"
            )
        z_det = params["crl_position"] + params["total_dist"]

    # SGI-measured detector-to-focus distance
    R_measured = calibration_result["R"]

    # Calculate source distance
    result = calculate_source_distance(
        f=crl_focal_length,
        z_CRL=params["crl_position"],
        z_det=z_det,
        R_measured=R_measured,
        verbose=verbose,
    )

    # Add focal length to result
    result["crl_focal_length"] = crl_focal_length

    return result
