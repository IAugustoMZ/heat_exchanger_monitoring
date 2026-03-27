"""
Heat transfer and pressure drop correlations for shell-and-tube heat exchangers.
================================================================================

All functions accept both Python scalars and NumPy arrays so they can be called
efficiently from the Method-of-Lines PDE solver (once per node per time step).

Tube-side
---------
* Reynolds number
* Prandtl number
* Nusselt number  — Dittus-Boelter (1930), with laminar / transitional fallbacks
* Friction factor — Churchill (1977), valid for ALL Re and any wall roughness
* Pressure drop   — Darcy-Weisbach per unit length

Shell-side  (Kern 1950)
-----------
* Equivalent hydraulic diameter (triangular / square pitch)
* Nusselt number
* Friction factor
* Pressure drop

Combined
--------
* Overall heat transfer coefficient U (flat-wall approximation, valid for
  t_wall << D_h — acceptable for PoC geometries)

References
----------
Dittus, F.W. & Boelter, L.M.K. (1930). Heat transfer in automobile radiators of
    the tubular type. University of California Publications in Engineering, 2, 443.

Churchill, S.W. (1977). Friction-factor equation spans all fluid-flow regimes.
    Chemical Engineering, 84(24), 91-92.

Kern, D.Q. (1950). Process Heat Transfer. McGraw-Hill.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tube-side dimensionless groups
# ---------------------------------------------------------------------------


def reynolds_number(
    mass_flow_rate: float | np.ndarray,
    hydraulic_diameter: float | np.ndarray,
    viscosity: float,
    flow_area: float | np.ndarray,
) -> float | np.ndarray:
    """
    Reynolds number for tube-side flow.

    Re = G * D_h / mu    where G = m_dot / A_flow  [kg/m^2/s]

    Parameters
    ----------
    mass_flow_rate : float or ndarray
        Total tube-side mass flow rate [kg/s].  Constant along z for
        incompressible-flow assumption.
    hydraulic_diameter : float or ndarray
        Current hydraulic diameter [m].  Varies with frost thickness delta_f(z,t).
    viscosity : float
        Dynamic viscosity [Pa·s].
    flow_area : float or ndarray
        Total tube-flow cross-section [m^2].  Decreases as frost grows.

    Returns
    -------
    Re : float or ndarray  (clipped to [1, inf) for numerical stability)
    """
    if np.any(np.asarray(flow_area) <= 0):
        raise ValueError("flow_area must be strictly positive.")
    if np.any(np.asarray(hydraulic_diameter) <= 0):
        raise ValueError("hydraulic_diameter must be strictly positive.")
    if viscosity <= 0:
        raise ValueError(f"viscosity must be positive, got {viscosity}.")

    G = mass_flow_rate / flow_area  # mass flux  [kg/m^2/s]
    Re = G * hydraulic_diameter / viscosity
    return np.maximum(Re, 1.0)


def prandtl_number(cp: float, viscosity: float, conductivity: float) -> float:
    """
    Prandtl number.

    Pr = cp * mu / k

    Parameters
    ----------
    cp : float  [J/kg/K]
    viscosity : float  [Pa·s]
    conductivity : float  [W/m/K]

    Returns
    -------
    Pr : float  (clipped to [0.1, 1000])
    """
    if conductivity <= 0:
        raise ValueError(f"conductivity must be positive, got {conductivity}.")
    if viscosity <= 0:
        raise ValueError(f"viscosity must be positive, got {viscosity}.")
    Pr = cp * viscosity / conductivity
    return float(np.clip(Pr, 0.1, 1000.0))


# ---------------------------------------------------------------------------
# Tube-side heat transfer: Dittus-Boelter
# ---------------------------------------------------------------------------


def nusselt_dittus_boelter(
    Re: float | np.ndarray,
    Pr: float,
    heating: bool = False,
) -> float | np.ndarray:
    """
    Tube-side Nusselt number — Dittus-Boelter correlation.

    Nu = 0.023 * Re^0.8 * Pr^n
         n = 0.4  (fluid is being heated)
         n = 0.3  (fluid is being cooled)

    Validity: Re > 10 000; 0.6 < Pr < 160; L/D > 10.

    Outside the turbulent regime a piecewise fallback is used:
    * Re < 2 300  → laminar fully-developed: Nu = 3.66 (Graetz / Sieder-Tate)
    * 2 300 ≤ Re < 10 000 → linear interpolation (conservative estimate)

    Parameters
    ----------
    Re : float or ndarray
    Pr : float
    heating : bool  True = fluid being heated, False = fluid being cooled.

    Returns
    -------
    Nu : float or ndarray
    """
    n = 0.4 if heating else 0.3
    Re = np.asarray(Re, dtype=float)
    scalar_input = Re.ndim == 0
    Re = np.atleast_1d(Re)

    Nu_lam = 3.66
    Nu_turb = 0.023 * (Re**0.8) * (Pr**n)

    alpha = np.clip((Re - 2300.0) / (10_000.0 - 2300.0), 0.0, 1.0)
    Nu = (1.0 - alpha) * Nu_lam + alpha * Nu_turb

    result = Nu if not scalar_input else float(Nu[0])
    return result


def heat_transfer_coefficient(
    Nu: float | np.ndarray,
    conductivity: float,
    hydraulic_diameter: float | np.ndarray,
) -> float | np.ndarray:
    """
    Heat transfer coefficient from Nusselt number.

    h = Nu * k / D_h

    Parameters
    ----------
    Nu : float or ndarray
    conductivity : float  [W/m/K]
    hydraulic_diameter : float or ndarray  [m]

    Returns
    -------
    h : float or ndarray  [W/m^2/K]
    """
    D_h = np.asarray(hydraulic_diameter)
    if np.any(D_h <= 0):
        raise ValueError("hydraulic_diameter must be strictly positive.")
    return Nu * conductivity / D_h


# ---------------------------------------------------------------------------
# Tube-side friction: Churchill (1977)
# ---------------------------------------------------------------------------


def friction_factor_churchill(
    Re: float | np.ndarray,
    relative_roughness: float = 1.0e-5,
) -> float | np.ndarray:
    """
    Churchill (1977) Darcy friction factor — valid for ALL Reynolds numbers
    (laminar, transitional, fully turbulent) and arbitrary wall roughness.

    f_D = 8 * [(8/Re)^12 + (A + B)^(-3/2)]^(1/12)

    where
        A = [2.457 * ln(1 / ((7/Re)^0.9 + 0.27*(eps/D)))]^16
        B = (37530 / Re)^16

    Limiting behaviour
    ------------------
    * Laminar   (low Re):  f_D → 64/Re  (Hagen-Poiseuille)
    * Turbulent (high Re): f_D → Colebrook-White (smooth: → Blasius)

    Reference
    ---------
    Churchill, S.W. (1977). Chemical Engineering, 84(24), 91-92.

    Parameters
    ----------
    Re : float or ndarray   (clipped to [1, inf) internally)
    relative_roughness : float  eps/D  (default 1e-5 ≈ drawn tubing)

    Returns
    -------
    f_D : float or ndarray  (Darcy-Weisbach friction factor, dimensionless)
    """
    Re = np.maximum(np.asarray(Re, dtype=float), 1.0)
    eps_D = max(float(relative_roughness), 0.0)

    inner = (7.0 / Re) ** 0.9 + 0.27 * eps_D
    # Guard against log(0); inner is always > 0 for finite Re and eps_D >= 0
    inner = np.maximum(inner, 1.0e-300)

    A = (2.457 * np.log(1.0 / inner)) ** 16
    B = (37530.0 / Re) ** 16

    f_D = 8.0 * ((8.0 / Re) ** 12 + (A + B) ** (-1.5)) ** (1.0 / 12.0)
    return f_D


def pressure_drop_per_unit_length(
    friction_factor: float | np.ndarray,
    mass_flux: float | np.ndarray,
    density: float,
    hydraulic_diameter: float | np.ndarray,
) -> float | np.ndarray:
    """
    Tube-side pressure drop per unit axial length (Darcy-Weisbach).

    dP/dz = f_D * G^2 / (2 * rho * D_h)    [Pa/m]

    The total pressure drop is obtained by numerically integrating this
    quantity over the N axial nodes in the PDE solver.

    Parameters
    ----------
    friction_factor : float or ndarray  f_D  (Darcy)
    mass_flux : float or ndarray        G = m_dot / A_flow  [kg/m^2/s]
    density : float                     rho  [kg/m^3]
    hydraulic_diameter : float or ndarray  D_h  [m]

    Returns
    -------
    dP_dz : float or ndarray  [Pa/m]
    """
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}.")
    D_h = np.asarray(hydraulic_diameter)
    if np.any(D_h <= 0):
        raise ValueError("hydraulic_diameter must be strictly positive.")

    return friction_factor * (np.asarray(mass_flux) ** 2) / (2.0 * density * D_h)


# ---------------------------------------------------------------------------
# Shell-side geometry
# ---------------------------------------------------------------------------


def equivalent_diameter_shell(
    pitch: float,
    tube_od: float,
    layout: str = "triangular",
) -> float:
    """
    Shell-side hydraulic (equivalent) diameter between tubes (Kern 1950).

    Triangular pitch:
        D_e = 4 * (sqrt(3)/4 * p^2 - pi/8 * d_o^2) / (pi/2 * d_o)

    Square pitch:
        D_e = 4 * (p^2 - pi*d_o^2/4) / (pi * d_o)

    Parameters
    ----------
    pitch : float   [m] tube centre-to-centre spacing
    tube_od : float [m] tube outer diameter
    layout : str    "triangular" (default) or "square"

    Returns
    -------
    D_e : float  [m]
    """
    if pitch <= tube_od:
        raise ValueError(
            f"pitch ({pitch} m) must be greater than tube_od ({tube_od} m)."
        )

    if layout == "triangular":
        D_e = (
            4.0
            * (np.sqrt(3) / 4.0 * pitch**2 - np.pi / 8.0 * tube_od**2)
            / (np.pi / 2.0 * tube_od)
        )
    elif layout == "square":
        D_e = (
            4.0
            * (pitch**2 - np.pi * tube_od**2 / 4.0)
            / (np.pi * tube_od)
        )
    else:
        raise ValueError(
            f"Unknown tube layout '{layout}'. Supported: 'triangular', 'square'."
        )

    if D_e <= 0:
        raise ValueError(
            f"Computed D_e = {D_e:.6f} m is non-positive. "
            "Check pitch > tube_od with a reasonable margin."
        )
    return float(D_e)


# ---------------------------------------------------------------------------
# Shell-side heat transfer: Kern (1950)
# ---------------------------------------------------------------------------


def nusselt_kern_shell(Re_s: float, Pr_s: float) -> float:
    """
    Shell-side Nusselt number — Kern (1950) correlation.

    Nu_s = 0.36 * Re_s^0.55 * Pr_s^(1/3)

    The viscosity correction factor (mu/mu_w)^0.14 is taken as unity for
    cryogenic LNG applications where wall-to-bulk viscosity differences are
    small (justified PoC assumption).

    Validity: 2e3 < Re_s < 1e6 for cross-flow over tube bundles.

    Parameters
    ----------
    Re_s : float  shell-side Reynolds number
    Pr_s : float  shell-side Prandtl number

    Returns
    -------
    Nu_s : float
    """
    Re_s = max(float(Re_s), 1.0)
    Pr_s = max(float(Pr_s), 0.1)
    return 0.36 * (Re_s**0.55) * (Pr_s ** (1.0 / 3.0))


def friction_factor_kern_shell(Re_s: float) -> float:
    """
    Shell-side friction factor — Kern (1950).

    f_s = exp(0.576 - 0.19 * ln(Re_s))

    Validity: 400 <= Re_s <= 1e6.

    Parameters
    ----------
    Re_s : float

    Returns
    -------
    f_s : float
    """
    Re_s = max(float(Re_s), 400.0)
    return float(np.exp(0.576 - 0.19 * np.log(Re_s)))


def pressure_drop_shell_side(
    f_s: float,
    mass_flux_s: float,
    density_s: float,
    shell_diameter: float,
    equiv_diameter: float,
    n_baffles: int,
) -> float:
    """
    Shell-side pressure drop — Kern (1950).

    dP_s = f_s * G_s^2 * D_s * (N_b + 1) / (2 * rho_s * D_e)

    Parameters
    ----------
    f_s : float            shell-side friction factor
    mass_flux_s : float    G_s = m_dot_s / A_cross [kg/m^2/s]
    density_s : float      [kg/m^3]
    shell_diameter : float [m]
    equiv_diameter : float [m]
    n_baffles : int        number of baffles

    Returns
    -------
    dP_s : float  [Pa]
    """
    if density_s <= 0:
        raise ValueError(f"density_s must be positive, got {density_s}.")
    if equiv_diameter <= 0:
        raise ValueError(f"equiv_diameter must be positive, got {equiv_diameter}.")

    return (
        f_s
        * mass_flux_s**2
        * shell_diameter
        * (n_baffles + 1)
        / (2.0 * density_s * equiv_diameter)
    )


# ---------------------------------------------------------------------------
# Overall heat transfer coefficient
# ---------------------------------------------------------------------------


def overall_heat_transfer_coefficient(
    h_tube: float | np.ndarray,
    h_shell: float,
    frost_thickness: float | np.ndarray,
    k_frost: float = 0.7,
    wall_thickness: float = 0.002,
    k_wall: float = 15.0,
) -> float | np.ndarray:
    """
    Overall heat transfer coefficient based on tube inner surface area.

    Flat-wall (planar) resistance network:
        1/U = 1/h_tube + delta_f/k_CO2 + t_wall/k_wall + 1/h_shell

    The cylindrical correction (log-mean area) is omitted here; it is
    negligible when t_wall << D_h, which holds for the STHE geometry
    parameterised in the scenarios (D_h ≈ 20 mm, t_wall = 2 mm).

    Parameters
    ----------
    h_tube : float or ndarray    [W/m^2/K]  tube-side (gas) coefficient
    h_shell : float              [W/m^2/K]  shell-side (LNG) coefficient
    frost_thickness : float or ndarray  [m]  delta_f >= 0
    k_frost : float      [W/m/K]  solid CO2 conductivity (Maqsood 2014: 0.7)
    wall_thickness : float  [m]   tube wall thickness (default 2 mm)
    k_wall : float          [W/m/K] stainless steel (default 15 W/m/K)

    Returns
    -------
    U : float or ndarray  [W/m^2/K]  (clipped to [0.1, inf))
    """
    delta_f = np.maximum(np.asarray(frost_thickness, dtype=float), 0.0)
    h_t = np.maximum(np.asarray(h_tube, dtype=float), 0.1)
    h_s = max(float(h_shell), 0.1)

    R_tube = 1.0 / h_t
    R_frost = delta_f / max(k_frost, 1.0e-9)
    R_wall = wall_thickness / max(k_wall, 1.0e-9)
    R_shell = 1.0 / h_s

    R_total = R_tube + R_frost + R_wall + R_shell
    U = 1.0 / R_total
    return np.maximum(U, 0.1)
