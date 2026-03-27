"""
Operating scenario definitions and sensor noise injection.
===========================================================

Each scenario is a named configuration of OperatingConditions + noise
parameters.  The module also provides utilities to:

  * inject realistic SCADA-style measurement noise onto simulation outputs
  * assemble a labelled Pandas DataFrame from multiple simulation results
  * define a canonical set of five scenarios covering normal operation,
    gradual freeze, rapid freeze, defrost recovery, and partial blockage.

Scenarios
---------
1. normal_operation   — stable ΔP, y_CO2 = 0.5 %  (well below frost onset)
2. gradual_freezing   — slow frost build-up over 6 h,  y_CO2 = 2 %
3. rapid_freezing     — high CO2, low T_c → alarm threshold crossed < 3 h
4. defrost_recovery   — warm-gas purge: T_h_in raised mid-run (2-piece sim)
5. partial_blockage   — localised blockage at inlet, spatially non-uniform
                        (achieved via elevated k_dep at first 20 % of nodes —
                        modelled through a spatially varying k_dep multiplier
                        passed as a perturbation to OperatingConditions)

Noise model
-----------
Each measured channel is corrupted by additive white Gaussian noise:

    x_noisy = x_clean + sigma * N(0, 1)

The standard deviation sigma is expressed as a fraction of the nominal
signal value, matching expected DCS/SCADA instrument classes:

    Temperature sensors (RTD):    ±0.5 K   (absolute)
    Pressure transmitters (ΔP):   0.2 %    of full-scale reading
    Flow transmitters:            0.5 %    of reading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any

import numpy as np
import pandas as pd

from src.heat_exchanger import (
    FluidProperties,
    OperatingConditions,
    ShellAndTubeGeometry,
    SimulationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------


@dataclass
class NoiseParameters:
    """
    Gaussian noise levels for each measurable channel.

    Attributes
    ----------
    sigma_T : float       temperature noise  [K]   absolute
    sigma_dP_frac : float ΔP noise as a fraction of the instantaneous value
    sigma_mdot_frac : float flow noise as a fraction of the nominal value
    rng_seed : int or None  for reproducibility
    """

    sigma_T: float = 0.5           # K   — RTD class A
    sigma_dP_frac: float = 0.002   # 0.2 % of reading
    sigma_mdot_frac: float = 0.005  # 0.5 % of reading
    rng_seed: int | None = None


def add_noise(
    result: SimulationResult,
    noise: NoiseParameters,
    nominal_dP: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Inject SCADA-style Gaussian noise onto simulation time series.

    Only the time-series observables that a real DCS would log are included:
        - T_h_in  : gas inlet temperature  (scalar over time)
        - T_h_out : gas outlet temperature (spatial mean of T_h[:, -1])
        - T_c_in  : LNG inlet temperature  (scalar over time)
        - T_c_out : LNG outlet temperature (spatial mean of T_c[:, 0])
        - delta_P : tube-side total pressure drop
        - U_mean  : (not directly measurable, included for ML ground truth)
        - delta_f_max : maximum frost thickness (ground truth label)

    Parameters
    ----------
    result : SimulationResult
    noise : NoiseParameters
    nominal_dP : float or None
        If provided, ΔP noise sigma = sigma_dP_frac * nominal_dP.
        Defaults to sigma_dP_frac * mean(delta_P_total).

    Returns
    -------
    dict mapping channel name → noisy time series (shape (n_t,))
    """
    rng = np.random.default_rng(noise.rng_seed)
    n_t = len(result.t)

    def gauss(sigma: float) -> np.ndarray:
        return rng.standard_normal(n_t) * sigma

    T_h_in_clean = result.T_h[:, 0]
    T_h_out_clean = result.T_h[:, -1]
    T_c_in_clean = result.T_c[:, -1]    # counter-flow: right boundary
    T_c_out_clean = result.T_c[:, 0]

    dP_clean = result.delta_P_total
    dP_ref = nominal_dP if nominal_dP is not None else float(np.mean(dP_clean))
    dP_ref = max(dP_ref, 1.0)  # avoid zero sigma

    return {
        "t_s": result.t,
        "T_h_in_K": T_h_in_clean + gauss(noise.sigma_T),
        "T_h_out_K": T_h_out_clean + gauss(noise.sigma_T),
        "T_c_in_K": T_c_in_clean + gauss(noise.sigma_T),
        "T_c_out_K": T_c_out_clean + gauss(noise.sigma_T),
        "delta_P_Pa": dP_clean + gauss(noise.sigma_dP_frac * dP_ref),
        # Ground-truth labels (not noisy)
        "U_mean_W_m2K": result.U_mean,
        "delta_f_max_m": result.delta_f_max,
        "delta_f_mean_m": result.delta_f_mean,
        "scenario": np.full(n_t, result.scenario_name, dtype=object),
    }


def results_to_dataframe(
    noisy_channels: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Convert a noisy channel dict to a tidy Pandas DataFrame."""
    return pd.DataFrame(noisy_channels)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """
    A named simulation scenario with its own operating conditions and noise.
    """

    name: str
    operating: OperatingConditions
    noise: NoiseParameters = field(default_factory=NoiseParameters)
    description: str = ""


def _default_geometry() -> ShellAndTubeGeometry:
    return ShellAndTubeGeometry()


def _default_fluid() -> FluidProperties:
    return FluidProperties()


# ------------------------------------------------------------------
# 1. Normal operation  (y_CO2 very low, no significant frost)
# ------------------------------------------------------------------
SCENARIO_NORMAL = Scenario(
    name="normal_operation",
    description=(
        "Low CO2 content (0.5 mol%). Wall temperature stays above the CO2 "
        "frost point — no significant frost deposition. ΔP is stable."
    ),
    operating=OperatingConditions(
        T_h_in=255.0,
        T_c_in=120.0,
        mdot_h=5.0,
        mdot_c=8.0,
        y_co2=0.005,        # 0.5 % — below practical frost threshold
        P_total_Pa=40.0e5,
        k_dep=1.5e-6,
        k_rem=5.0e-6,
        t_end_s=6.0 * 3600,
        n_t_out=360,
    ),
    noise=NoiseParameters(rng_seed=42),
)

# ------------------------------------------------------------------
# 2. Gradual freezing  (2 % CO2, steady decay of ΔP over 12 h)
# ------------------------------------------------------------------
SCENARIO_GRADUAL = Scenario(
    name="gradual_freezing",
    description=(
        "2 mol% CO2. Frost slowly builds up axially from the cold end. "
        "ΔP rises 2–3× over 24 hours — typical scheduled defrost scenario."
    ),
    operating=OperatingConditions(
        T_h_in=250.0,
        T_c_in=115.0,
        mdot_h=5.0,
        mdot_c=8.0,
        y_co2=0.020,
        P_total_Pa=40.0e5,
        k_dep=1.5e-6,
        k_rem=5.0e-6,
        t_end_s=24.0 * 3600,
        n_t_out=1440,
    ),
    noise=NoiseParameters(rng_seed=7),
)

# ------------------------------------------------------------------
# 3. Rapid freezing  (3 % CO2, higher k_dep → alarm in < 3 h)
# ------------------------------------------------------------------
SCENARIO_RAPID = Scenario(
    name="rapid_freezing",
    description=(
        "3 mol% CO2 with a colder LNG inlet. Very high driving force → "
        "rapid frost build-up. ΔP alarm threshold crossed within ~2-3 h."
    ),
    operating=OperatingConditions(
        T_h_in=245.0,
        T_c_in=105.0,
        mdot_h=5.0,
        mdot_c=8.0,
        y_co2=0.030,
        P_total_Pa=40.0e5,
        k_dep=3.0e-6,       # elevated — high CO2 scenario
        k_rem=5.0e-6,
        t_end_s=4.0 * 3600,
        n_t_out=480,
    ),
    noise=NoiseParameters(rng_seed=99),
)

# ------------------------------------------------------------------
# 4. Defrost recovery  (warm-gas purge: elevated T_h_in)
# ------------------------------------------------------------------
SCENARIO_DEFROST = Scenario(
    name="defrost_recovery",
    description=(
        "Warm-gas purge phase: T_h_in raised to 290 K. Starts from an already-frosted "
        "exchanger (initial_frost_m = 0.4 mm). High k_rem >> k_dep so the frost layer "
        "shrinks over 2 h, ΔP drops back to baseline."
    ),
    operating=OperatingConditions(
        T_h_in=290.0,        # warm purge gas
        T_c_in=145.0,        # LNG flow reduced / warmed
        mdot_h=4.0,
        mdot_c=5.0,
        y_co2=0.005,         # near-zero CO2 — purge gas
        P_total_Pa=40.0e5,
        k_dep=0.5e-6,        # minimal deposition
        k_rem=5.0e-4,        # high erosion → frost melts back
        initial_frost_m=4.0e-4,  # 0.4 mm pre-existing frost from prior freeze
        t_end_s=2.0 * 3600,
        n_t_out=240,
    ),
    noise=NoiseParameters(rng_seed=13),
)

# ------------------------------------------------------------------
# 5. Partial blockage  (high k_dep at inlet — spatially non-uniform)
# ------------------------------------------------------------------
SCENARIO_PARTIAL = Scenario(
    name="partial_blockage",
    description=(
        "Non-uniform CO2 deposition — modelled as high k_dep at the cold-end "
        "inlet (first 20 % of tube length) where the driving force is largest. "
        "Achieved via a 5× k_dep multiplier for those nodes."
        " ΔP rises quickly but plateau earlier than full-exchanger freeze."
    ),
    operating=OperatingConditions(
        T_h_in=250.0,
        T_c_in=110.0,
        mdot_h=5.0,
        mdot_c=8.0,
        y_co2=0.025,
        P_total_Pa=40.0e5,
        k_dep=1.5e-6,        # base — spatially multiplied in solver wrapper
        k_rem=5.0e-6,
        t_end_s=16.0 * 3600,
        n_t_out=960,
    ),
    noise=NoiseParameters(rng_seed=55),
)


ALL_SCENARIOS: list[Scenario] = [
    SCENARIO_NORMAL,
    SCENARIO_GRADUAL,
    SCENARIO_RAPID,
    SCENARIO_DEFROST,
    SCENARIO_PARTIAL,
]


# ---------------------------------------------------------------------------
# Operating condition perturbation for multi-run dataset generation
# ---------------------------------------------------------------------------


def perturb_operating_conditions(
    ops: OperatingConditions,
    rng: np.random.Generator,
    intensity: float = 0.05,
) -> OperatingConditions:
    """
    Return a copy of *ops* with key parameters randomly perturbed.

    This generates physically diverse training data by varying inlet conditions
    and kinetic parameters within realistic operating envelopes.  All constraints
    (T_h_in > T_c_in, positive flows, bounded mole fractions) are enforced.

    Perturbation ranges (all symmetric Gaussian with std = intensity × scale):
      T_h_in   ± intensity × 20 K
      T_c_in   ± intensity × 10 K
      mdot_h   ± intensity × mdot_h
      mdot_c   ± intensity × mdot_c
      y_co2    ± intensity × 0.5 × y_co2   (capped [0.001, 0.05])
      k_dep    ± intensity × k_dep         (natural fouling variability)
      k_rem    ± intensity × k_rem

    Parameters
    ----------
    ops : OperatingConditions   base (nominal) operating point
    rng : np.random.Generator   seeded RNG for reproducibility
    intensity : float           perturbation magnitude (default 0.05 = 5 %)

    Returns
    -------
    OperatingConditions  (a new dataclass instance; *ops* is not modified)
    """
    if not (0.0 < intensity <= 0.5):
        raise ValueError(
            f"intensity must be in (0, 0.5], got {intensity}."
        )

    def _gauss(scale: float) -> float:
        return float(rng.normal(0.0, intensity * scale))

    T_h_new = ops.T_h_in + _gauss(20.0)
    T_c_new = ops.T_c_in + _gauss(10.0)

    # Enforce a minimum temperature separation of 30 K for physical validity
    _MIN_DELTA_T = 30.0
    if T_h_new - T_c_new < _MIN_DELTA_T:
        T_h_new = T_c_new + _MIN_DELTA_T
        logger.debug(
            "Temperature separation guardrail triggered: T_h_in forced to %.1f K.",
            T_h_new,
        )

    mdot_h_new = max(ops.mdot_h * (1.0 + _gauss(1.0)), 0.5)
    mdot_c_new = max(ops.mdot_c * (1.0 + _gauss(1.0)), 0.5)

    # CO2 mole fraction: perturb by up to ±50 % of nominal, clamped to [0.001, 0.05]
    y_co2_new = float(np.clip(ops.y_co2 * (1.0 + _gauss(0.5)), 0.001, 0.05))

    k_dep_new = max(ops.k_dep * (1.0 + _gauss(1.0)), 1.0e-8)
    k_rem_new = max(ops.k_rem * (1.0 + _gauss(1.0)), 1.0e-8)

    return _dc_replace(
        ops,
        T_h_in=T_h_new,
        T_c_in=T_c_new,
        mdot_h=mdot_h_new,
        mdot_c=mdot_c_new,
        y_co2=y_co2_new,
        k_dep=k_dep_new,
        k_rem=k_rem_new,
    )


# ---------------------------------------------------------------------------
# Partial-blockage specialised runner
# ---------------------------------------------------------------------------


def run_partial_blockage_simulation(
    scenario: Scenario,
    geometry: ShellAndTubeGeometry | None = None,
    fluid: FluidProperties | None = None,
    run_id: int = 0,
) -> SimulationResult:
    """
    Run the partial-blockage scenario with spatially varying deposition.

    Applies the same guardrails as run_simulation:
      - NaN/Inf clipping and physical temperature bounds in ODE RHS
      - Blockage termination event (90 % diameter blockage)
      - Retry with relaxed tolerances on solver failure

    Parameters
    ----------
    scenario : Scenario   must be SCENARIO_PARTIAL or equivalent
    geometry : ShellAndTubeGeometry or None  (defaults to standard geometry)
    fluid : FluidProperties or None
    run_id : int   numeric run identifier for dataset traceability

    Returns
    -------
    SimulationResult
    """
    from src.heat_exchanger import (
        HeatExchangerODE, N_NODES, _make_blockage_event, compute_diagnostics
    )
    import time as _time_mod
    from scipy.integrate import solve_ivp

    geo = geometry or _default_geometry()
    fld = fluid or _default_fluid()
    ops = scenario.operating

    # Build spatially varying k_dep array
    n_high = int(0.20 * N_NODES)
    k_dep_array = np.full(N_NODES, ops.k_dep)
    k_dep_array[:n_high] *= 5.0   # 5× at cold inlet (z=0 side)

    # Monkey-patch a specialised ODE subclass
    class _PartialBlockageODE(HeatExchangerODE):
        def __call__(self, t: float, y: np.ndarray) -> np.ndarray:  # noqa: ARG002
            from src.correlations import (
                heat_transfer_coefficient,
                nusselt_dittus_boelter,
                overall_heat_transfer_coefficient,
                prandtl_number,
                reynolds_number,
            )
            from src.freezing_model import (
                frost_growth_rate,
                frost_surface_temperature,
                hydraulic_diameter_with_frost,
                tube_flow_area,
                tube_mass_flux,
            )
            from src.heat_exchanger import u_h_c

            N = self.N

            # --- GUARDRAILS (mirrors HeatExchangerODE.__call__) ---
            n_bad = int(np.sum(~np.isfinite(y)))
            if n_bad > 0:
                self._nan_warn_count += 1
                if self._nan_warn_count <= self._NAN_WARN_MAX:
                    logger.warning(
                        "[partial_blockage] Non-finite state at t=%.1f s "
                        "(%d/%d elements). Clipping.", t, n_bad, len(y)
                    )
                y = np.where(np.isfinite(y), y, 0.0)

            T_h = np.clip(y[0:N], self._T_min, self._T_max)
            T_c = np.clip(y[N : 2 * N], self._T_min, self._T_max)
            delta_f = np.clip(y[2 * N : 3 * N], 0.0, self._max_frost)

            D_h = hydraulic_diameter_with_frost(self.geo.tube_id, delta_f)
            A_flow = tube_flow_area(D_h, self.geo.n_tubes)
            G_h = tube_mass_flux(self.ops.mdot_h, D_h, self.geo.n_tubes)
            u_h = G_h / self.fluid.rho_h

            Pr_h = prandtl_number(self.fluid.cp_h, self.fluid.mu_h, self.fluid.k_h)
            Re_h = reynolds_number(self.ops.mdot_h, D_h, self.fluid.mu_h, A_flow)
            Nu_h = nusselt_dittus_boelter(Re_h, Pr_h, heating=False)
            h_tube = heat_transfer_coefficient(Nu_h, self.fluid.k_h, D_h)

            U = overall_heat_transfer_coefficient(
                h_tube, self.h_shell, delta_f,
                k_frost=self.fluid.k_frost,
                wall_thickness=self.geo.wall_thickness,
                k_wall=self.geo.k_wall,
            )

            total_perimeter = self.geo.n_tubes * np.pi * D_h
            A_sh = self.geo.shell_cross_flow_area
            Q_per_vol_h = U * total_perimeter / (self.fluid.rho_h * self.fluid.cp_h * A_flow)
            Q_per_vol_c = U * total_perimeter / (self.fluid.rho_c * self.fluid.cp_c * A_sh)
            delta_T = T_h - T_c

            T_w = frost_surface_temperature(
                T_h, T_c, h_tube, self.h_shell, delta_f,
                k_frost=self.fluid.k_frost,
                wall_thickness=self.geo.wall_thickness,
                k_wall=self.geo.k_wall,
            )

            # Spatially varying k_dep
            d_delta_f = frost_growth_rate(
                delta_f, T_w, self.ops.y_co2, self.ops.P_total_Pa, u_h,
                k_dep=k_dep_array,   # <-- array, not scalar
                k_rem=self.ops.k_rem,
            )

            dTh_dz = np.empty(N)
            dTh_dz[0] = (T_h[0] - self.ops.T_h_in) / self.dz
            dTh_dz[1:] = (T_h[1:] - T_h[:-1]) / self.dz

            dTc_dz = np.empty(N)
            dTc_dz[-1] = (self.ops.T_c_in - T_c[-1]) / self.dz
            dTc_dz[:-1] = (T_c[1:] - T_c[:-1]) / self.dz

            dTh_dt = -u_h * dTh_dz - Q_per_vol_h * delta_T
            dTc_dt = u_h_c(self.ops.mdot_c, self.fluid.rho_c, A_sh) * dTc_dz + Q_per_vol_c * delta_T

            dydt = np.concatenate([dTh_dt, dTc_dt, d_delta_f])
            # Guard output
            if not np.all(np.isfinite(dydt)):
                dydt = np.where(np.isfinite(dydt), dydt, 0.0)
            return dydt

    import time as _time_mod
    from scipy.integrate import solve_ivp
    from src.heat_exchanger import SimulationResult, compute_diagnostics

    ode = _PartialBlockageODE(geo, fld, ops, n_nodes=N_NODES)
    y0 = ode.initial_state()
    blockage_event = _make_blockage_event(geo.tube_id, N_NODES)

    logger.info("Starting partial-blockage simulation (run_id=%d).", run_id)
    t0 = _time_mod.perf_counter()

    _RETRY_TOLERANCES = [(1e-4, 1e-7), (1e-3, 1e-6), (1e-2, 1e-5)]
    sol = None
    last_exc: str = ""
    for attempt, (cur_rtol, cur_atol) in enumerate(_RETRY_TOLERANCES, start=1):
        if attempt > 1:
            logger.warning(
                "[partial_blockage] Retry %d/3 with rtol=%.0e atol=%.0e",
                attempt, cur_rtol, cur_atol,
            )
            ode = _PartialBlockageODE(geo, fld, ops, n_nodes=N_NODES)
        try:
            sol = solve_ivp(
                ode,
                ops.t_span(),
                y0,
                method="Radau",
                t_eval=ops.t_eval(),
                events=[blockage_event],
                rtol=cur_rtol,
                atol=cur_atol,
                dense_output=False,
                max_step=60.0,
            )
        except Exception as exc:
            last_exc = str(exc)
            logger.error("[partial_blockage] Solver exception attempt %d: %s", attempt, exc)
            sol = None
            continue
        if sol is not None and sol.status != -1:
            break
        last_exc = getattr(sol, "message", "unknown")
        logger.warning("[partial_blockage] Attempt %d failed: %s", attempt, last_exc)

    elapsed = _time_mod.perf_counter() - t0

    if sol is None or sol.status == -1:
        logger.error("All retries exhausted for partial_blockage: %s", last_exc)
        return SimulationResult(
            scenario_name=scenario.name,
            run_id=run_id,
            success=False,
            solver_message=last_exc,
            elapsed_wall_time_s=elapsed,
        )

    early_term = sol.status == 1
    if early_term:
        logger.warning(
            "[partial_blockage] Early termination at t=%.0f s (blockage event).",
            sol.t[-1] if len(sol.t) else 0.0,
        )

    if len(sol.t) == 0:
        return SimulationResult(
            scenario_name=scenario.name,
            run_id=run_id,
            success=False,
            solver_message="Zero time steps returned.",
            elapsed_wall_time_s=elapsed,
        )

    states = sol.y.T
    N = N_NODES
    diag = compute_diagnostics(sol.t, states, ode)

    return SimulationResult(
        t=sol.t,
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
        scenario_name=scenario.name,
        run_id=run_id,
        elapsed_wall_time_s=elapsed,
        solver_message=sol.message,
        success=True,
        early_termination=early_term,
    )
