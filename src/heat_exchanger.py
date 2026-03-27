"""
Transient 1-D PDE solver for counter-flow shell-and-tube heat exchanger
with CO2 frost deposition.
=======================================================================

Mathematical model
------------------
Three coupled PDEs are solved on the axial domain z ∈ [0, L], t ∈ [0, T_end]:

  (1) Tube-side (hot/gas) temperature T_h(z, t)   — flows in +z direction
  (2) Shell-side (cold/LNG) temperature T_c(z, t)  — flows in −z direction
  (3) Frost thickness delta_f(z, t)                — local ODE at each node

Spatio-temporal PDEs (1) and (2)
---------------------------------
∂T_h/∂t + u_h(z,t) * ∂T_h/∂z = -[U(z,t) * π * D_h(z,t)] /
                                   [ρ_h * cp_h * A_h(z,t)] * (T_h - T_c)

∂T_c/∂t − u_c       * ∂T_c/∂z = +[U(z,t) * π * D_h(z,t)] /
                                   [ρ_c * cp_c * A_sh     ] * (T_h - T_c)

Frost growth ODE (3)
---------------------
d(delta_f)/dt = k_dep * max(0, y_CO2 − y_eq(T_w)) − k_rem * u_h * delta_f

Numerical method — Method of Lines (MOL)
-----------------------------------------
* Spatial domain: N = 100 uniform axial nodes, spacing dz = L / (N-1)
* ∂T_h/∂z: first-order upwind (flow direction +z) → backward difference
* ∂T_c/∂z: first-order upwind (flow direction −z) → forward  difference
* Time integration: scipy.integrate.solve_ivp with stiff solver 'Radau'

Boundary conditions
-------------------
* T_h(z=0, t) = T_h_in  [K]  gas inlet (left boundary)
* T_c(z=L, t) = T_c_in  [K]  LNG inlet (right boundary, counter-flow)

State vector layout (length 3*N)
----------------------------------
  y[0 : N]       = T_h  (K)
  y[N : 2*N]     = T_c  (K)
  y[2*N : 3*N]   = delta_f  (m)

Diagnostics
-----------
After each successful solve, pressure drop (ΔP) and the spatial mean overall
heat transfer coefficient (U_bar) are computed for each saved time step and
returned as part of the SimulationResult dataclass.

References
----------
Maqsood, K. et al. (2014). Chemical Engineering Journal, 253, 327-336.
Bai, F. & Newell, T.A. (2002). Int. J. Refrigeration, 25(4), 476-484.
Churchill, S.W. (1977). Chemical Engineering, 84(24), 91-92.
Kern, D.Q. (1950). Process Heat Transfer. McGraw-Hill.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# NumPy 2.0 renamed trapz → trapezoid; support both versions transparently
_np_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]

from src.correlations import (
    friction_factor_churchill,
    heat_transfer_coefficient,
    nusselt_dittus_boelter,
    nusselt_kern_shell,
    overall_heat_transfer_coefficient,
    prandtl_number,
    pressure_drop_per_unit_length,
    reynolds_number,
)
from src.freezing_model import (
    co2_equilibrium_mole_fraction,
    frost_growth_rate,
    frost_surface_temperature,
    hydraulic_diameter_with_frost,
    tube_flow_area,
    tube_mass_flux,
)

logger = logging.getLogger(__name__)

N_NODES: int = 100  # axial spatial nodes


# ---------------------------------------------------------------------------
# Geometry and fluid property containers
# ---------------------------------------------------------------------------


@dataclass
class ShellAndTubeGeometry:
    """
    Fixed geometric parameters for the shell-and-tube heat exchanger.

    All lengths in metres; areas in m^2.
    """

    # --- Tube side ---
    tube_length: float = 6.0          # [m]   total tube length
    tube_id: float = 0.020            # [m]   inner diameter (clean)
    tube_od: float = 0.025            # [m]   outer diameter
    n_tubes: int = 200                # [-]   number of parallel tubes
    tube_roughness: float = 1.5e-5   # [m]   absolute wall roughness (drawn ss)

    # --- Shell side ---
    shell_id: float = 0.60            # [m]   internal shell diameter
    pitch: float = 0.032              # [m]   tube pitch (triangular)
    n_baffles: int = 12               # [-]   number of baffles
    baffle_spacing: float = 0.46      # [m]   baffle spacing ≈ L/(N_b+1)
    tube_layout: str = "triangular"   # "triangular" or "square"

    # --- Wall ---
    wall_thickness: float = 0.0025   # [m]   tube wall thickness
    k_wall: float = 15.0             # [W/m/K] stainless steel

    def __post_init__(self) -> None:
        if self.tube_id >= self.tube_od:
            raise ValueError(
                f"tube_id ({self.tube_id} m) must be less than "
                f"tube_od ({self.tube_od} m)."
            )
        if self.pitch <= self.tube_od:
            raise ValueError(
                f"pitch ({self.pitch} m) must exceed tube_od ({self.tube_od} m)."
            )

    @property
    def tube_wall_thickness(self) -> float:
        return (self.tube_od - self.tube_id) / 2.0

    @property
    def clean_flow_area_per_tube(self) -> float:
        """Cross-section of a single clean tube [m^2]."""
        return np.pi / 4.0 * self.tube_id**2

    @property
    def total_clean_flow_area(self) -> float:
        """Total tube-side flow cross-section (all tubes, clean) [m^2]."""
        return self.n_tubes * self.clean_flow_area_per_tube

    @property
    def shell_cross_flow_area(self) -> float:
        """
        Shell-side cross-flow area at baffle window (Kern approximation).

        A_s = (D_s - n_t_row * d_o) * B
        Simplified as: A_s = D_s * B_s * (1 - d_o/pitch)
        """
        return (
            self.shell_id
            * self.baffle_spacing
            * (1.0 - self.tube_od / self.pitch)
        )


@dataclass
class FluidProperties:
    """
    Fluid physical properties (assumed constant — PoC simplification).

    For a production model, replace with CoolProp lookups.
    """

    # --- Tube side (natural gas, methane-rich) ---
    rho_h: float = 32.0        # [kg/m^3]  at ~40 bar, 200 K
    cp_h: float = 2_300.0      # [J/kg/K]  gas cp
    mu_h: float = 1.2e-5       # [Pa·s]    dynamic viscosity
    k_h: float = 0.035         # [W/m/K]   thermal conductivity

    # --- Shell side (LNG / mixed refrigerant) ---
    rho_c: float = 430.0       # [kg/m^3]  liquid MR ~ 430 kg/m^3
    cp_c: float = 2_100.0      # [J/kg/K]
    mu_c: float = 1.8e-4       # [Pa·s]    liquid viscosity
    k_c: float = 0.12          # [W/m/K]

    # --- CO2 frost ---
    k_frost: float = 0.7       # [W/m/K]  dense solid CO2 (Maqsood 2014)


@dataclass
class OperatingConditions:
    """
    Inlet conditions and operating parameters for one simulation run.
    """

    # --- Inlet temperatures ---
    T_h_in: float = 250.0        # [K]  hot gas inlet temperature
    T_c_in: float = 120.0        # [K]  cold LNG inlet temperature

    # --- Mass flow rates ---
    mdot_h: float = 5.0          # [kg/s]  tube-side gas
    mdot_c: float = 8.0          # [kg/s]  shell-side LNG

    # --- Gas composition ---
    y_co2: float = 0.02          # [-]  CO2 mole fraction (2 %)
    P_total_Pa: float = 40.0e5   # [Pa]  tube-side operating pressure (40 bar)

    # --- Frost kinetics ---
    k_dep: float = 1.5e-6        # [m/s]  deposition rate constant
    k_rem: float = 5.0e-6        # [1/m]  erosion rate constant

    # --- Initial frost (for chained / defrost scenarios) ---
    initial_frost_m: float = 1.0e-6  # [m] uniform initial frost thickness seed

    # --- Simulation duration ---
    t_end_s: float = 6.0 * 3600  # [s]   6 hours
    n_t_out: int = 360           # [-]   number of output time steps

    def t_span(self) -> tuple[float, float]:
        return (0.0, self.t_end_s)

    def t_eval(self) -> np.ndarray:
        return np.linspace(0.0, self.t_end_s, self.n_t_out)


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """
    All outputs from a single simulation run.

    Arrays have shape  (n_t_out, N_NODES)  unless stated otherwise.
    """

    # Time axis [s]  shape (n_t_out,)
    t: np.ndarray = field(default_factory=lambda: np.array([]))

    # Axial coordinates [m]  shape (N_NODES,)
    z: np.ndarray = field(default_factory=lambda: np.array([]))

    # State fields  shape (n_t_out, N_NODES)
    T_h: np.ndarray = field(default_factory=lambda: np.array([]))
    T_c: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_f: np.ndarray = field(default_factory=lambda: np.array([]))

    # Derived diagnostics  shape (n_t_out, N_NODES)
    U_field: np.ndarray = field(default_factory=lambda: np.array([]))
    dP_dz: np.ndarray = field(default_factory=lambda: np.array([]))

    # Scalar time series  shape (n_t_out,)
    delta_P_total: np.ndarray = field(default_factory=lambda: np.array([]))
    U_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_f_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_f_max: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    scenario_name: str = ""
    run_id: int = 0
    elapsed_wall_time_s: float = 0.0
    solver_message: str = ""
    success: bool = False
    early_termination: bool = False  # True when blockage event fired


# ---------------------------------------------------------------------------
# Core ODE right-hand side (Method of Lines)
# ---------------------------------------------------------------------------


class HeatExchangerODE:
    """
    Encapsulates the Method-of-Lines discretisation of the STHE PDE system.

    Instantiated once per simulation; the __call__ method is passed to
    scipy.integrate.solve_ivp as the RHS function f(t, y).
    """

    def __init__(
        self,
        geometry: ShellAndTubeGeometry,
        fluid: FluidProperties,
        operating: OperatingConditions,
        n_nodes: int = N_NODES,
    ) -> None:
        self.geo = geometry
        self.fluid = fluid
        self.ops = operating
        self.N = n_nodes

        self.z = np.linspace(0.0, geometry.tube_length, n_nodes)
        self.dz = self.z[1] - self.z[0]

        # Pre-compute constant shell-side properties
        from src.correlations import (
            equivalent_diameter_shell,
            friction_factor_kern_shell,
        )

        self.D_e = equivalent_diameter_shell(
            geometry.pitch, geometry.tube_od, geometry.tube_layout
        )

        Pr_c = prandtl_number(fluid.cp_c, fluid.mu_c, fluid.k_c)
        G_s = operating.mdot_c / geometry.shell_cross_flow_area
        Re_s = G_s * self.D_e / fluid.mu_c
        Nu_s = nusselt_kern_shell(Re_s, Pr_c)
        self.h_shell = float(heat_transfer_coefficient(Nu_s, fluid.k_c, self.D_e))

        # Shell-side pressure drop (constant — no fouling on shell)
        f_s = friction_factor_kern_shell(Re_s)
        from src.correlations import pressure_drop_shell_side

        self.dP_shell = pressure_drop_shell_side(
            f_s, G_s, fluid.rho_c,
            geometry.shell_id, self.D_e, geometry.n_baffles,
        )

        # Pre-compute physical temperature guard bounds
        self._T_min: float = max(operating.T_c_in - 20.0, 50.0)   # 50 K absolute floor
        self._T_max: float = operating.T_h_in + 50.0               # generous ceiling
        self._max_frost: float = 0.45 * geometry.tube_id / 2.0    # 90 % blockage limit

        # NaN/Inf warning throttle: only emit the first N_WARN_MAX warnings
        self._nan_warn_count: int = 0
        self._NAN_WARN_MAX: int = 5

        logger.info(
            "HeatExchangerODE initialised: N=%d nodes, dz=%.4f m, "
            "h_shell=%.1f W/m²K, dP_shell=%.1f Pa",
            n_nodes, self.dz, self.h_shell, self.dP_shell,
        )

    def initial_state(self) -> np.ndarray:
        """
        Linear temperature profiles + zero frost as initial condition.

        T_h: linear from T_h_in (z=0) to T_h_in − 20 K (z=L)
        T_c: linear from T_c_in (z=L) to T_c_in + 15 K (z=0)
        delta_f: uniform 1e-6 m (numerical seed to avoid zero-division)
        """
        N = self.N
        T_h0 = np.linspace(self.ops.T_h_in, self.ops.T_h_in - 20.0, N)
        T_c0 = np.linspace(self.ops.T_c_in + 15.0, self.ops.T_c_in, N)
        delta_f0 = np.full(N, max(self.ops.initial_frost_m, 1.0e-6))
        return np.concatenate([T_h0, T_c0, delta_f0])

    # ------------------------------------------------------------------
    # RHS: this is the core physics, called ~10^5 times per simulation
    # ------------------------------------------------------------------

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG002
        N = self.N
        geo = self.geo
        fluid = self.fluid
        ops = self.ops

        # ------------------------------------------------------------------
        # GUARDRAIL 1: Sanitize non-finite state entries before any physics.
        # The stiff Radau solver occasionally evaluates speculative trial
        # points outside physical bounds; clipping prevents correlation
        # functions from receiving NaN/Inf and producing undefined RHS values.
        # ------------------------------------------------------------------
        n_bad = int(np.sum(~np.isfinite(y)))
        if n_bad > 0:
            self._nan_warn_count += 1
            if self._nan_warn_count <= self._NAN_WARN_MAX:
                logger.warning(
                    "Non-finite state at t=%.1f s (%d/%d elements). "
                    "Clipping to physical bounds.",
                    t, n_bad, len(y),
                )
            elif self._nan_warn_count == self._NAN_WARN_MAX + 1:
                logger.warning(
                    "Non-finite state warnings suppressed after %d occurrences.",
                    self._NAN_WARN_MAX,
                )
            y = np.where(np.isfinite(y), y, 0.0)

        # Unpack state vector
        T_h = np.clip(y[0:N], self._T_min, self._T_max)
        T_c = np.clip(y[N : 2 * N], self._T_min, self._T_max)
        # GUARDRAIL 2: Cap frost below the 90%-blockage limit so hydraulic
        # diameter never reaches zero (the blockage termination event fires
        # before this, but the clip is a last-resort safety net).
        delta_f = np.clip(y[2 * N : 3 * N], 0.0, self._max_frost)

        # -----------------------------------------------------------------
        # 1. Geometry: hydraulic diameter and flow area at each node
        # -----------------------------------------------------------------
        D_h = hydraulic_diameter_with_frost(geo.tube_id, delta_f)   # (N,)
        A_flow = tube_flow_area(D_h, geo.n_tubes)                    # (N,)
        G_h = tube_mass_flux(ops.mdot_h, D_h, geo.n_tubes)          # (N,)  [kg/m²/s]
        u_h = G_h / fluid.rho_h                                      # (N,)  [m/s]

        # -----------------------------------------------------------------
        # 2. Tube-side heat transfer coefficient (Dittus-Boelter)
        # -----------------------------------------------------------------
        Pr_h = prandtl_number(fluid.cp_h, fluid.mu_h, fluid.k_h)
        Re_h = reynolds_number(ops.mdot_h, D_h, fluid.mu_h, A_flow)
        Nu_h = nusselt_dittus_boelter(Re_h, Pr_h, heating=False)    # gas is cooled
        h_tube = heat_transfer_coefficient(Nu_h, fluid.k_h, D_h)    # (N,)

        # -----------------------------------------------------------------
        # 3. Overall U and heat exchange source terms
        # -----------------------------------------------------------------
        U = overall_heat_transfer_coefficient(
            h_tube, self.h_shell, delta_f,
            k_frost=fluid.k_frost,
            wall_thickness=geo.wall_thickness,
            k_wall=geo.k_wall,
        )  # (N,)

        # Total heat transfer perimeter per unit axial length [m/m]:
        # = n_tubes * pi * D_h(z,t)   (each tube contributes pi*D_h)
        total_perimeter = geo.n_tubes * np.pi * D_h

        # Volumetric heat exchange coefficient [1/s]:
        #   Q_h = U * total_perimeter / (rho * cp * A_flow) * delta_T  [K/s]
        # Equivalently for tube side: U * 4 / (D_h * rho_h * cp_h)
        Q_per_vol_h = U * total_perimeter / (fluid.rho_h * fluid.cp_h * A_flow)
        A_sh = geo.shell_cross_flow_area
        Q_per_vol_c = U * total_perimeter / (fluid.rho_c * fluid.cp_c * A_sh)

        delta_T = T_h - T_c  # local temperature driving force

        # -----------------------------------------------------------------
        # 4. Frost surface temperature and growth rate
        # -----------------------------------------------------------------
        T_w = frost_surface_temperature(
            T_h, T_c, h_tube, self.h_shell, delta_f,
            k_frost=fluid.k_frost,
            wall_thickness=geo.wall_thickness,
            k_wall=geo.k_wall,
        )
        d_delta_f = frost_growth_rate(
            delta_f, T_w, ops.y_co2, ops.P_total_Pa, u_h,
            k_dep=ops.k_dep, k_rem=ops.k_rem,
        )

        # -----------------------------------------------------------------
        # 5. Method of Lines: spatial derivatives (first-order upwind)
        # -----------------------------------------------------------------
        # T_h flows in +z direction → backward (upwind) difference
        dTh_dz = np.empty(N)
        dTh_dz[0] = (T_h[0] - self.ops.T_h_in) / self.dz   # BC: T_h[z=0] = T_h_in
        dTh_dz[1:] = (T_h[1:] - T_h[:-1]) / self.dz

        # T_c flows in -z direction → forward (upwind relative to flow) difference
        dTc_dz = np.empty(N)
        dTc_dz[-1] = (self.ops.T_c_in - T_c[-1]) / self.dz  # BC: T_c[z=L] = T_c_in
        dTc_dz[:-1] = (T_c[1:] - T_c[:-1]) / self.dz

        # -----------------------------------------------------------------
        # 6. Assemble time derivatives
        # -----------------------------------------------------------------
        # Gas velocity at each node [m/s]
        # u_h(z,t) = m_dot / (rho_h * A_flow(z,t))  — already computed above

        dTh_dt = -u_h * dTh_dz - Q_per_vol_h * delta_T
        dTc_dt = +u_h_c(ops.mdot_c, fluid.rho_c, A_sh) * dTc_dz + Q_per_vol_c * delta_T

        dydt = np.concatenate([dTh_dt, dTc_dt, d_delta_f])

        # GUARDRAIL 3: Replace any non-finite derivative with zero so the
        # solver step is rejected (step-size control) rather than aborting.
        if not np.all(np.isfinite(dydt)):
            logger.warning(
                "Non-finite RHS at t=%.1f s — zeroing %d elements.",
                t, int(np.sum(~np.isfinite(dydt))),
            )
            dydt = np.where(np.isfinite(dydt), dydt, 0.0)

        return dydt


def u_h_c(mdot_c: float, rho_c: float, A_sh: float) -> float:
    """Shell-side velocity magnitude [m/s] (scalar, constant)."""
    return mdot_c / (rho_c * A_sh)


# ---------------------------------------------------------------------------
# Blockage termination event (scipy solve_ivp events API)
# ---------------------------------------------------------------------------


def _make_blockage_event(tube_id: float, n_nodes: int):
    """
    Return a terminal scipy event that fires when the maximum frost thickness
    exceeds 45 % of the tube radius (i.e. 90 % diameter blockage).

    At this point the hydraulic diameter is only 10 % of its clean value and
    the pressure drop is ~10^5× elevated — a physically unreachable steady
    state.  Terminating early preserves the usable partial trajectory and
    avoids solver stiffness explosion.

    The event function returns a value that is positive under normal operation
    and crosses zero (going negative) when the blockage limit is reached.
    Setting terminal=True and direction=-1 stops integration on that crossing.
    """
    limit = 0.45 * tube_id / 2.0   # 45 % of tube radius
    _2N = n_nodes * 2
    _3N = n_nodes * 3

    def blockage_event(t: float, y: np.ndarray) -> float:  # noqa: ARG001
        max_frost = float(np.max(np.maximum(y[_2N:_3N], 0.0)))
        return limit - max_frost   # crosses zero from + to − as frost grows

    blockage_event.terminal = True    # type: ignore[attr-defined]
    blockage_event.direction = -1     # type: ignore[attr-defined]
    return blockage_event


# ---------------------------------------------------------------------------
# Diagnostic post-processing
# ---------------------------------------------------------------------------


def compute_diagnostics(
    t_arr: np.ndarray,
    states: np.ndarray,
    ode: HeatExchangerODE,
) -> dict[str, np.ndarray]:
    """
    Compute ΔP, U field, and summary statistics for all saved time steps.

    Parameters
    ----------
    t_arr : ndarray  shape (n_t,)
    states : ndarray shape (n_t, 3*N)
    ode : HeatExchangerODE

    Returns
    -------
    dict with keys:
        U_field       (n_t, N)
        dP_dz         (n_t, N)          [Pa/m]
        delta_P_total (n_t,)            [Pa]  tube-side total
        U_mean        (n_t,)
        delta_f_mean  (n_t,)
        delta_f_max   (n_t,)
    """
    N = ode.N
    n_t = len(t_arr)
    geo = ode.geo
    fluid = ode.fluid
    ops = ode.ops

    Pr_h = prandtl_number(fluid.cp_h, fluid.mu_h, fluid.k_h)

    U_field = np.empty((n_t, N))
    dP_dz_field = np.empty((n_t, N))

    for i in range(n_t):
        delta_f = np.maximum(states[i, 2 * N : 3 * N], 0.0)

        D_h = hydraulic_diameter_with_frost(geo.tube_id, delta_f)
        A_flow = tube_flow_area(D_h, geo.n_tubes)
        G_h = tube_mass_flux(ops.mdot_h, D_h, geo.n_tubes)

        Re_h = reynolds_number(ops.mdot_h, D_h, fluid.mu_h, A_flow)
        Nu_h = nusselt_dittus_boelter(Re_h, Pr_h, heating=False)
        h_tube = heat_transfer_coefficient(Nu_h, fluid.k_h, D_h)

        U = overall_heat_transfer_coefficient(
            h_tube, ode.h_shell, delta_f,
            k_frost=fluid.k_frost,
            wall_thickness=geo.wall_thickness,
            k_wall=geo.k_wall,
        )
        U_field[i] = U

        eps_D = geo.tube_roughness / D_h
        f_D = friction_factor_churchill(Re_h, relative_roughness=float(np.mean(eps_D)))
        dP_dz_field[i] = pressure_drop_per_unit_length(f_D, G_h, fluid.rho_h, D_h)

    dz = ode.dz
    delta_P_total = _np_trapezoid(dP_dz_field, dx=dz, axis=1)

    return {
        "U_field": U_field,
        "dP_dz": dP_dz_field,
        "delta_P_total": delta_P_total,
        "U_mean": U_field.mean(axis=1),
        "delta_f_mean": np.maximum(states[:, 2 * N : 3 * N], 0.0).mean(axis=1),
        "delta_f_max": np.maximum(states[:, 2 * N : 3 * N], 0.0).max(axis=1),
    }


# ---------------------------------------------------------------------------
# Public simulation entry point
# ---------------------------------------------------------------------------


def run_simulation(
    geometry: ShellAndTubeGeometry,
    fluid: FluidProperties,
    operating: OperatingConditions,
    scenario_name: str = "unnamed",
    run_id: int = 0,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-7,
) -> SimulationResult:
    """
    Run a single transient simulation of the frosting STHE.

    Guardrails
    ----------
    * Blockage event: terminates early when max frost > 90 % diameter blockage;
      the partial result is returned with early_termination=True and success=True
      so that ML pipelines can still use the trajectory.
    * Retry logic: on solver failure, retries up to 2 more times with
      progressively relaxed tolerances (rtol/atol × 10 each attempt).
    * NaN/Inf clipping: the ODE RHS clips temperatures and frost thickness to
      physical bounds before invoking any correlation (see HeatExchangerODE).

    Parameters
    ----------
    geometry : ShellAndTubeGeometry
    fluid : FluidProperties
    operating : OperatingConditions
    scenario_name : str   label for logging and output
    run_id : int          numeric run identifier (0 = nominal, 1+ = perturbed)
    rtol : float          initial relative tolerance for solve_ivp
    atol : float          initial absolute tolerance for solve_ivp

    Returns
    -------
    SimulationResult
        Contains full spatiotemporal fields and derived pressure/U signals.
        Always check result.success before using the data.
        If result.early_termination is True, the time axis is shorter than
        operating.t_end_s — truncated when blockage threshold was reached.
    """
    logger.info("Starting simulation: %s  (run_id=%d)", scenario_name, run_id)
    t0_wall = time.perf_counter()

    ode = HeatExchangerODE(geometry, fluid, operating, n_nodes=N_NODES)
    y0 = ode.initial_state()
    t_span = operating.t_span()
    t_eval = operating.t_eval()
    blockage_event = _make_blockage_event(geometry.tube_id, N_NODES)

    # Retry schedule: (rtol, atol) for each attempt
    _RETRY_TOLERANCES = [
        (rtol,        atol),          # attempt 1 — standard
        (rtol * 10,   atol * 10),     # attempt 2 — relaxed ×10
        (rtol * 100,  atol * 100),    # attempt 3 — very relaxed ×100
    ]

    sol = None
    last_exc: str = ""
    for attempt, (cur_rtol, cur_atol) in enumerate(_RETRY_TOLERANCES, start=1):
        if attempt > 1:
            logger.warning(
                "Retrying '%s' (attempt %d/3) with relaxed tolerances "
                "rtol=%.0e atol=%.0e",
                scenario_name, attempt, cur_rtol, cur_atol,
            )
            # Re-instantiate ODE so warning counters reset
            ode = HeatExchangerODE(geometry, fluid, operating, n_nodes=N_NODES)

        try:
            sol = solve_ivp(
                ode,
                t_span,
                y0,
                method="Radau",
                t_eval=t_eval,
                events=[blockage_event],
                rtol=cur_rtol,
                atol=cur_atol,
                dense_output=False,
                max_step=60.0,
            )
        except Exception as exc:
            last_exc = str(exc)
            logger.error(
                "solve_ivp exception on attempt %d for '%s': %s",
                attempt, scenario_name, exc,
            )
            sol = None
            continue

        # status=1  → terminated by blockage event (success=True, partial)
        # status=0  → integration span completed successfully
        # status=-1 → integration failure → retry
        if sol is not None and sol.status != -1:
            break   # good result, exit retry loop

        last_exc = getattr(sol, "message", "unknown solver failure")
        logger.warning("Attempt %d failed: %s", attempt, last_exc)

    elapsed = time.perf_counter() - t0_wall

    if sol is None or sol.status == -1:
        logger.error(
            "All retry attempts exhausted for scenario '%s'. "
            "Last error: %s",
            scenario_name, last_exc,
        )
        return SimulationResult(
            scenario_name=scenario_name,
            run_id=run_id,
            success=False,
            solver_message=last_exc,
            elapsed_wall_time_s=elapsed,
        )

    # Detect early blockage termination
    early_term = sol.status == 1
    if early_term:
        logger.warning(
            "Scenario '%s' terminated early at t=%.0f s (%.1f %% of planned) "
            "due to tube blockage (max frost > 90 %% blockage). "
            "Partial result returned — valid for ML training.",
            scenario_name,
            sol.t[-1] if len(sol.t) else 0.0,
            100.0 * sol.t[-1] / operating.t_end_s if len(sol.t) else 0.0,
        )

    states = sol.y.T          # shape (n_t_stored, 3*N)
    N = N_NODES
    t_arr = sol.t

    if len(t_arr) == 0:
        logger.error(
            "Solver for '%s' returned zero time steps — cannot build result.",
            scenario_name,
        )
        return SimulationResult(
            scenario_name=scenario_name,
            run_id=run_id,
            success=False,
            solver_message="Zero time steps returned by solver.",
            elapsed_wall_time_s=elapsed,
        )

    logger.info(
        "Simulation '%s' (run_id=%d) completed in %.1f s  "
        "(%d steps stored, early_term=%s).",
        scenario_name, run_id, elapsed, len(t_arr), early_term,
    )

    diag = compute_diagnostics(t_arr, states, ode)

    result = SimulationResult(
        t=t_arr,
        z=ode.z,
        T_h=states[:, 0:N],
        T_c=states[:, N : 2 * N],
        delta_f=np.maximum(states[:, 2 * N : 3 * N], 0.0),
        U_field=diag["U_field"],
        dP_dz=diag["dP_dz"],
        delta_P_total=diag["delta_P_total"],
        U_mean=diag["U_mean"],
        delta_f_mean=diag["delta_f_mean"],
        delta_f_max=diag["delta_f_max"],
        scenario_name=scenario_name,
        run_id=run_id,
        elapsed_wall_time_s=elapsed,
        solver_message=sol.message,
        success=True,
        early_termination=early_term,
    )
    return result
