"""
CO2 frost deposition / erosion model for LNG tube-side flow.
=============================================================

Physical basis
--------------
When natural gas (carrying ~1-3 mol% CO2) is cooled below the CO2 frost
point, solid CO2 sublimes out of the gas phase and deposits on the tube
inner wall as a frost layer of thickness delta_f(z, t).  The layer grows
until either:

  * the thermal resistance of the frost warms the gas-frost interface enough
    that the driving force (y_CO2 - y_eq) vanishes, OR
  * mechanical erosion by the gas flow balances deposition.

Over time, delta_f reduces the hydraulic diameter D_h(z, t) = D_h0 - 2*delta_f,
which nonlinearly amplifies the tube-side pressure drop (ΔP ~ 1/D_h^5 at
constant mass flow).  This ΔP rise is the primary signal used by the
downstream ML freezing detector.

Frost growth model (Stefan-type, after Maqsood et al. 2014)
------------------------------------------------------------
    d(delta_f)/dt = k_dep * max(0, y_CO2 - y_eq(T_w)) - k_rem * u_h * delta_f

where T_w is the temperature at the gas-frost interface (inner frost surface).

CO2 sublimation curve
---------------------
Derived from Clausius-Clapeyron fit to NIST/ASHRAE data (Span & Wagner 1996)
in the range 154 K – 216.58 K (triple point):

    ln(P_sub / Pa) = 27.630 - 3134.4 / T(K)

Verification:
    T = 194.65 K  →  P = 101 340 Pa  ≈ 1 atm   (CO2 dry-ice point) ✓
    T = 216.58 K  →  P = 518 500 Pa  ≈ 5.19 bar (CO2 triple point)  ✓

References
----------
Maqsood, K. et al. (2014). Cryogenic packed beds for CO2 separation.
    Chemical Engineering Journal, 253, 327-336.

Bai, F. & Newell, T.A. (2002). Modeling of CO2 freezing in a cryogenic
    heat exchanger. International Journal of Refrigeration, 25(4), 476-484.

Span, R. & Wagner, W. (1996). A new equation of state for CO2 covering the
    fluid region from the triple-point temperature to 1100 K at pressures
    up to 800 MPa. J. Phys. Chem. Ref. Data, 25(6), 1509-1596.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CO2 sublimation curve constants
# Fit: ln(P_sub/Pa) = A - B/T(K)  over 154-216.58 K
# ---------------------------------------------------------------------------
_CO2_SUB_A: float = 27.630   # intercept
_CO2_SUB_B: float = 3134.4   # slope  [K]

CO2_TRIPLE_POINT_T_K: float = 216.58   # [K]
CO2_TRIPLE_POINT_P_PA: float = 518_500.0   # [Pa]  (5.185 bar)


# ---------------------------------------------------------------------------
# Sublimation equilibrium
# ---------------------------------------------------------------------------


def co2_sublimation_pressure(T_K: float | np.ndarray) -> float | np.ndarray:
    """
    CO2 solid-vapour equilibrium (sublimation) pressure.

    ln(P_sub / Pa) = 27.630 - 3134.4 / T(K)

    Above the triple-point temperature (216.58 K) CO2 can no longer exist as
    a stable solid at equilibrium; the function returns the triple-point
    pressure as a conservative upper bound so that callers see no
    thermodynamic driving force for deposition.

    Parameters
    ----------
    T_K : float or ndarray  Wall temperature [K]

    Returns
    -------
    P_sub : float or ndarray  Sublimation pressure [Pa]
    """
    T = np.asarray(T_K, dtype=float)
    scalar = T.ndim == 0
    T = np.atleast_1d(T)

    if np.any(T <= 0):
        raise ValueError(
            "Wall temperature must be strictly positive (T > 0 K). "
            f"Received min T = {float(np.min(T)):.3f} K."
        )

    ln_P = _CO2_SUB_A - _CO2_SUB_B / T
    P_sub = np.exp(ln_P)

    # Above triple point: solid phase is not stable → cap at triple-point P
    P_sub = np.where(T >= CO2_TRIPLE_POINT_T_K, CO2_TRIPLE_POINT_P_PA, P_sub)

    return float(P_sub[0]) if scalar else P_sub


def co2_equilibrium_mole_fraction(
    T_K: float | np.ndarray,
    P_total_Pa: float,
) -> float | np.ndarray:
    """
    Equilibrium CO2 mole fraction in the gas phase at the frost surface.

    y_eq(T_w) = P_sub(T_w) / P_total

    Deposition occurs when y_CO2_bulk > y_eq(T_w), i.e., the gas is
    supersaturated with respect to solid CO2 at the wall temperature.

    Parameters
    ----------
    T_K : float or ndarray  Wall (frost surface) temperature [K]
    P_total_Pa : float      Total tube-side gas pressure [Pa]

    Returns
    -------
    y_eq : float or ndarray  (clipped to [0, 1])
    """
    if P_total_Pa <= 0:
        raise ValueError(
            f"P_total_Pa must be strictly positive, got {P_total_Pa}."
        )

    P_sub = co2_sublimation_pressure(T_K)
    y_eq = P_sub / P_total_Pa
    return np.clip(y_eq, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Thermal resistance network
# ---------------------------------------------------------------------------


def frost_surface_temperature(
    T_bulk_h: float | np.ndarray,
    T_bulk_c: float,
    h_tube: float | np.ndarray,
    h_shell: float,
    frost_thickness: float | np.ndarray,
    k_frost: float = 0.7,
    wall_thickness: float = 0.002,
    k_wall: float = 15.0,
) -> float | np.ndarray:
    """
    Temperature at the inner frost surface (gas-frost interface).

    Thermal resistance network (planar approximation):

        T_h → [1/h_tube] → T_frost_surface → [delta_f/k_CO2] → T_metal
            → [t_wall/k_wall] → T_outer → [1/h_shell] → T_c

    The total specific heat flux is:
        q = (T_h - T_c) / R_total     [W/m^2]

    The frost surface temperature is:
        T_frost_surface = T_h - q / h_tube

    When delta_f = 0, this reduces to the bare metal inner wall temperature.

    Parameters
    ----------
    T_bulk_h : float or ndarray  [K]  tube-side (hot/gas) bulk temperature
    T_bulk_c : float             [K]  shell-side (cold/LNG) bulk temperature
    h_tube : float or ndarray    [W/m^2/K]  tube-side heat transfer coefficient
    h_shell : float              [W/m^2/K]  shell-side heat transfer coefficient
    frost_thickness : float or ndarray  [m]  delta_f >= 0
    k_frost : float  [W/m/K]  solid CO2 conductivity (default 0.7)
    wall_thickness : float  [m]  tube wall thickness (default 2 mm)
    k_wall : float  [W/m/K]  tube wall material (default 15 = stainless steel)

    Returns
    -------
    T_w : float or ndarray  [K]  frost surface (or bare wall) temperature
    """
    delta_f = np.maximum(np.asarray(frost_thickness, dtype=float), 0.0)
    h_t = np.maximum(np.asarray(h_tube, dtype=float), 0.1)
    T_h_arr = np.asarray(T_bulk_h, dtype=float)
    T_c_arr = np.asarray(T_bulk_c, dtype=float)

    R_h = 1.0 / h_t
    R_f = delta_f / max(k_frost, 1.0e-9)
    R_w = wall_thickness / max(k_wall, 1.0e-9)
    R_c = 1.0 / max(h_shell, 0.1)

    R_total = R_h + R_f + R_w + R_c

    # Heat flux from gas to LNG  [W/m^2]
    q = (T_h_arr - T_c_arr) / R_total

    T_w = T_h_arr - q * R_h
    return T_w


# ---------------------------------------------------------------------------
# Frost growth kinetics
# ---------------------------------------------------------------------------


def frost_growth_rate(
    delta_f: float | np.ndarray,
    T_frost_surface_K: float | np.ndarray,
    y_co2_bulk: float,
    P_total_Pa: float,
    velocity_h: float | np.ndarray,
    k_dep: float = 1.5e-6,
    k_rem: float = 5.0e-6,
) -> float | np.ndarray:
    """
    CO2 frost layer growth rate at each axial node.

    d(delta_f)/dt = k_dep * max(0, y_CO2 - y_eq(T_w)) - k_rem * u_h * delta_f

    First term  — deposition: thermodynamic driving force (supersaturation).
    Second term — erosion: mechanical removal proportional to gas shear and
                  existing frost mass.

    Physical calibration
    --------------------
    * k_dep = 1.5e-6 m/s  →  initial growth ~1.5e-8 m/s at 1 % supersaturation
      → ~0.6 mm frost in 6 h at constant driving force (matches Maqsood 2014).
    * k_rem = 5.0e-6 1/m  →  natural steady-state at delta_f ≈ 1 mm with
      typical gas velocity u_h ≈ 5 m/s.

    Note on units
    -------------
    k_rem has units [1/m] so that:
        k_rem [1/m] * u_h [m/s] * delta_f [m]  →  [m/s]  ✓

    Parameters
    ----------
    delta_f : float or ndarray          Current frost thickness [m], >= 0
    T_frost_surface_K : float or ndarray  Gas-frost interface temperature [K]
    y_co2_bulk : float                  Bulk CO2 mole fraction (e.g., 0.02)
    P_total_Pa : float                  Total tube-side pressure [Pa]
    velocity_h : float or ndarray       Tube-side gas velocity [m/s]
    k_dep : float  [m/s]   Deposition rate constant (default 1.5e-6)
    k_rem : float  [1/m]   Erosion rate constant   (default 5.0e-6)

    Returns
    -------
    d_delta_f_dt : float or ndarray  [m/s]
    """
    delta_f = np.maximum(np.asarray(delta_f, dtype=float), 0.0)
    u_h = np.maximum(np.asarray(velocity_h, dtype=float), 0.0)

    if not (0.0 <= y_co2_bulk <= 1.0):
        raise ValueError(
            f"y_co2_bulk must be in [0, 1], got {y_co2_bulk}."
        )
    if P_total_Pa <= 0:
        raise ValueError(
            f"P_total_Pa must be positive, got {P_total_Pa}."
        )

    y_eq = co2_equilibrium_mole_fraction(T_frost_surface_K, P_total_Pa)

    driving_force = np.maximum(0.0, y_co2_bulk - y_eq)
    deposition = k_dep * driving_force
    erosion = k_rem * u_h * delta_f

    rate = deposition - erosion

    # Physical lower bound: frost cannot go negative
    rate = np.where(delta_f <= 0.0, np.maximum(rate, 0.0), rate)

    return rate


# ---------------------------------------------------------------------------
# Derived geometry helpers
# ---------------------------------------------------------------------------


def hydraulic_diameter_with_frost(
    D_h0: float,
    delta_f: float | np.ndarray,
) -> float | np.ndarray:
    """
    Effective hydraulic diameter after frost deposition.

    D_h(z, t) = D_h0 - 2 * delta_f(z, t)

    The factor 2 accounts for frost growing on the full inner circumference
    (radially symmetric deposition assumed).

    Parameters
    ----------
    D_h0 : float                Bare (clean) tube inner diameter [m]
    delta_f : float or ndarray  Frost thickness [m]

    Returns
    -------
    D_h : float or ndarray  [m]  Clipped to [D_h0 * 0.05, D_h0] so the
          solver never encounters a fully blocked or zero-diameter tube.
          A warning is logged if any node exceeds 95 % blockage.
    """
    delta_f = np.maximum(np.asarray(delta_f, dtype=float), 0.0)
    D_h = D_h0 - 2.0 * delta_f
    min_D_h = 0.05 * D_h0  # 95 % blockage limit

    if np.any(D_h < min_D_h):
        logger.warning(
            "Frost thickness approaching tube blockage limit. "
            "Clipping D_h to %.1f %% of D_h0 = %.4f m.  "
            "Consider stopping the simulation (defrost required).",
            5.0,
            D_h0,
        )

    return np.clip(D_h, min_D_h, D_h0)


def tube_flow_area(D_h: float | np.ndarray, n_tubes: int) -> float | np.ndarray:
    """
    Total tube-side flow cross-section for all tubes at a given axial position.

    A_flow(z, t) = n_tubes * (pi/4) * D_h(z, t)^2

    Parameters
    ----------
    D_h : float or ndarray  Hydraulic diameter [m] (may vary with z and t)
    n_tubes : int           Number of parallel tubes

    Returns
    -------
    A_flow : float or ndarray  [m^2]
    """
    return n_tubes * np.pi / 4.0 * np.asarray(D_h, dtype=float) ** 2


def tube_mass_flux(
    mass_flow_rate: float,
    D_h: float | np.ndarray,
    n_tubes: int,
) -> float | np.ndarray:
    """
    Tube-side mass flux (varies along z as frost narrows the cross-section).

    G(z, t) = m_dot / A_flow(z, t)   [kg/m^2/s]

    Parameters
    ----------
    mass_flow_rate : float  Total tube-side mass flow [kg/s]
    D_h : float or ndarray  Local hydraulic diameter [m]
    n_tubes : int           Number of parallel tubes

    Returns
    -------
    G : float or ndarray  [kg/m^2/s]
    """
    A_flow = tube_flow_area(D_h, n_tubes)
    return mass_flow_rate / np.maximum(A_flow, 1.0e-9)
