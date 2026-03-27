"""
data_service.py
===============
Loads and serves scenario data from the combined dataset.
Provides row-by-row iteration for streaming simulation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HX geometry and fluid property constants (mirrors ShellAndTubeGeometry /
# FluidProperties in src/heat_exchanger.py — kept here so the backend
# container has no dependency on the simulation source tree).
# ---------------------------------------------------------------------------
_TUBE_ID        = 0.020       # [m]  tube inner diameter (clean)
_N_TUBES        = 200         # [-]  number of parallel tubes
_TUBE_LENGTH    = 6.0         # [m]  tube length
_TUBE_ROUGHNESS = 1.5e-5      # [m]  drawn stainless steel
_MDOT_H         = 5.0         # [kg/s] nominal tube-side gas mass flow
_MU_H           = 1.2e-5      # [Pa·s] gas dynamic viscosity (PoC constant)
_P_TOTAL        = 40.0e5      # [Pa]  operating pressure (40 bar)
_M_H            = 0.016628    # [kg/mol] effective feed-gas molar mass (calibrated
                               #           so rho_h = 32.0 kg/m³ at 250 K, 40 bar,
                               #           matching FluidProperties simulation default)
_R_GAS          = 8.314       # [J/mol/K] universal gas constant


def _compute_dp_ideal_tube(T_h_in_K: float) -> float:
    """
    Clean-tube (frost-free) tube-side pressure drop using Darcy-Weisbach
    with Churchill (1977) friction factor.

    Gas density is computed via the ideal gas law so the baseline ΔP
    tracks changes in inlet temperature:

        ρ_h = P * M / (R * T_h_in)    [kg/m³]
        dP  = f_D * G² * L / (2 * ρ_h * D_h)   [Pa]

    At nominal T_h_in = 250 K this reproduces ≈ 618 Pa (matching the EDA
    precomputed reference value from the PDE simulation).
    """
    T = max(float(T_h_in_K), 100.0)   # guard against bad sensor reads

    # Ideal gas density
    rho_h = _P_TOTAL * _M_H / (_R_GAS * T)

    # Tube-side mass flux [kg/m²/s]
    A_flow = _N_TUBES * np.pi / 4.0 * _TUBE_ID ** 2
    G = _MDOT_H / A_flow

    # Reynolds number
    Re = max(G * _TUBE_ID / _MU_H, 1.0)

    # Churchill (1977) friction factor — valid for all Re and roughness
    eps_D = _TUBE_ROUGHNESS / _TUBE_ID
    inner = max((7.0 / Re) ** 0.9 + 0.27 * eps_D, 1.0e-300)
    A_ch  = (2.457 * np.log(1.0 / inner)) ** 16
    B_ch  = (37530.0 / Re) ** 16
    f_D   = 8.0 * ((8.0 / Re) ** 12 + (A_ch + B_ch) ** (-1.5)) ** (1.0 / 12.0)

    # Darcy-Weisbach total pressure drop over tube length
    dP_per_m = f_D * G ** 2 / (2.0 * rho_h * _TUBE_ID)
    return float(dP_per_m * _TUBE_LENGTH)


# First-principles ideal values (from EDA: clean-tube Dittus-Boelter/Kern/Churchill)
# dP_ideal is now computed dynamically via _compute_dp_ideal_tube(); the
# values below are kept only for U_ideal and outlet temperature ideals.
_IDEAL_VALUES = {
    "normal_operation": {"U_ideal": 255.6278, "dP_ideal": 618.4408, "T_h_out_ideal": 162.0454, "T_c_out_ideal": 183.3230},
    "gradual_freezing": {"U_ideal": 255.6278, "dP_ideal": 618.4408, "T_h_out_ideal": 162.0454, "T_c_out_ideal": 183.3230},
    "rapid_freezing":   {"U_ideal": 255.6278, "dP_ideal": 618.4408, "T_h_out_ideal": 162.0454, "T_c_out_ideal": 183.3230},
    "partial_blockage": {"U_ideal": 255.6278, "dP_ideal": 618.4408, "T_h_out_ideal": 162.0454, "T_c_out_ideal": 183.3230},
    "defrost_recovery": {"U_ideal": 208.1000, "dP_ideal": 404.6000, "T_h_out_ideal": 162.0454, "T_c_out_ideal": 183.3230},
}

# Observable columns present in combined_dataset.csv
_RAW_COLUMNS = [
    "scenario", "run_id", "t_s",
    "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K",
    "delta_P_Pa", "U_mean_W_m2K",
    "delta_f_mean_m", "delta_f_max_m",
    "freezing_alarm", "early_stop",
]


class DataService:
    """
    Loads combined_dataset.csv and provides scenario data for streaming.

    Computes derived features (error signals, lags, rates-of-change) that
    match the model's expected input signature.
    """

    def __init__(self, data_dir: str, eda_dir: str) -> None:
        self._data_dir = Path(data_dir)
        self._eda_dir = Path(eda_dir)
        self._df = self._load_combined_dataset()
        self._enriched: Optional[pd.DataFrame] = None
        self._scenarios = sorted(self._df["scenario"].unique().tolist())
        logger.info(
            "DataService: loaded %d rows, scenarios=%s",
            len(self._df), self._scenarios,
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    @property
    def scenarios(self) -> List[str]:
        return self._scenarios

    def get_scenario_data(self, scenario: str, run_id: int = 1) -> pd.DataFrame:
        """Return raw rows for a given scenario + run."""
        mask = (self._df["scenario"] == scenario) & (self._df["run_id"] == run_id)
        df = self._df.loc[mask].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No data found for scenario='{scenario}', run_id={run_id}. "
                f"Available scenarios: {self._scenarios}"
            )
        return df

    def build_model_input_row(
        self,
        scenario: str,
        current_row: Dict,
        history: List[Dict],
    ) -> Dict:
        """
        Build a single model input row matching the MLflow model signature.

        The model expects 25 features:
          t_s, T_h_in_K, T_h_out_K, T_c_in_K, T_c_out_K, delta_P_Pa,
          t_h, U_ideal, dP_ideal, T_h_out_ideal, T_c_out_ideal,
          U_error, T_h_out_error, T_c_out_error,
          dP_error(t-1), dP_error(t-2), dP_error(t-3),
          delta_P_Pa(t-1), delta_P_Pa(t-2), delta_P_Pa(t-3),
          U_error(t-1), U_error(t-2), U_error(t-3),
          d_dP_error_dt, d_delta_P_Pa_dt, t_norm
        """
        ideals = _IDEAL_VALUES.get(scenario, _IDEAL_VALUES["normal_operation"])

        t_s = float(current_row["t_s"])
        t_h = t_s / 3600.0

        T_h_in = float(current_row["T_h_in_K"])
        T_h_out = float(current_row["T_h_out_K"])
        T_c_out = float(current_row["T_c_out_K"])
        delta_P = float(current_row["delta_P_Pa"])

        # Dynamic ideal pressure drop from physics correlations (Darcy-Weisbach +
        # Churchill, with density from ideal gas law at current T_h_in)
        dp_ideal_dynamic = _compute_dp_ideal_tube(T_h_in)

        # Compute U_mean from raw data if available, else estimate from history
        U_mean = float(current_row.get("U_mean_W_m2K", ideals["U_ideal"]))

        # Error signals using dynamic baseline
        U_error = U_mean - ideals["U_ideal"]
        dP_error = delta_P - dp_ideal_dynamic
        T_h_out_error = T_h_out - ideals["T_h_out_ideal"]
        T_c_out_error = T_c_out - ideals["T_c_out_ideal"]

        # Lag features (from history)
        def _get_lag(field: str, lag_idx: int) -> float:
            """Get lagged value from history. Returns current if not enough history."""
            if lag_idx <= len(history):
                return float(history[-(lag_idx)][field])
            return float(current_row.get(field, 0.0))

        # Compute historical dP_error and U_error for lags
        # Use dynamic dP_ideal per historical point (based on that point's T_h_in)
        hist_dp_errors = []
        hist_delta_ps = []
        hist_u_errors = []
        for h in history:
            h_scenario = h.get("scenario", scenario)
            h_ideals = _IDEAL_VALUES.get(h_scenario, ideals)
            h_t_h_in = float(h.get("T_h_in_K", T_h_in))
            h_dp_ideal = _compute_dp_ideal_tube(h_t_h_in)
            h_dp_error = float(h["delta_P_Pa"]) - h_dp_ideal
            h_u_mean = float(h.get("U_mean_W_m2K", h_ideals["U_ideal"]))
            h_u_error = h_u_mean - h_ideals["U_ideal"]
            hist_dp_errors.append(h_dp_error)
            hist_delta_ps.append(float(h["delta_P_Pa"]))
            hist_u_errors.append(h_u_error)

        def _lag(values: List[float], lag: int, default: float) -> float:
            if lag <= len(values):
                return values[-(lag)]
            return default

        dp_error_lag1 = _lag(hist_dp_errors, 1, dP_error)
        dp_error_lag2 = _lag(hist_dp_errors, 2, dP_error)
        dp_error_lag3 = _lag(hist_dp_errors, 3, dP_error)

        delta_p_lag1 = _lag(hist_delta_ps, 1, delta_P)
        delta_p_lag2 = _lag(hist_delta_ps, 2, delta_P)
        delta_p_lag3 = _lag(hist_delta_ps, 3, delta_P)

        u_error_lag1 = _lag(hist_u_errors, 1, U_error)
        u_error_lag2 = _lag(hist_u_errors, 2, U_error)
        u_error_lag3 = _lag(hist_u_errors, 3, U_error)

        # Rate-of-change (smoothed finite difference over last ~5 points)
        window = 5
        all_dp_errors = hist_dp_errors + [dP_error]
        all_delta_ps = hist_delta_ps + [delta_P]

        if len(all_dp_errors) >= 2:
            recent_dp = all_dp_errors[-min(window, len(all_dp_errors)):]
            d_dp_error_dt = (recent_dp[-1] - recent_dp[0]) / max(len(recent_dp) - 1, 1)
        else:
            d_dp_error_dt = 0.0

        if len(all_delta_ps) >= 2:
            recent_p = all_delta_ps[-min(window, len(all_delta_ps)):]
            d_delta_p_dt = (recent_p[-1] - recent_p[0]) / max(len(recent_p) - 1, 1)
        else:
            d_delta_p_dt = 0.0

        # Time normalization (approximate — we don't know total duration a priori)
        # Use current t_h / estimated_duration (from scenario typical durations)
        _SCENARIO_DURATIONS = {
            "normal_operation": 6.0, "gradual_freezing": 24.0,
            "rapid_freezing": 4.0, "defrost_recovery": 2.0,
            "partial_blockage": 16.0,
        }
        est_duration = _SCENARIO_DURATIONS.get(scenario, 24.0)
        t_norm = min(t_h / est_duration, 1.0) if est_duration > 0 else 0.0

        return {
            "t_s": t_s,
            "T_h_in_K": float(current_row["T_h_in_K"]),
            "T_h_out_K": T_h_out,
            "T_c_in_K": float(current_row["T_c_in_K"]),
            "T_c_out_K": T_c_out,
            "delta_P_Pa": delta_P,
            "t_h": t_h,
            "U_ideal": ideals["U_ideal"],
            "dP_ideal": dp_ideal_dynamic,
            "T_h_out_ideal": ideals["T_h_out_ideal"],
            "T_c_out_ideal": ideals["T_c_out_ideal"],
            "U_error": U_error,
            "T_h_out_error": T_h_out_error,
            "T_c_out_error": T_c_out_error,
            "dP_error(t-1)": dp_error_lag1,
            "dP_error(t-2)": dp_error_lag2,
            "dP_error(t-3)": dp_error_lag3,
            "delta_P_Pa(t-1)": delta_p_lag1,
            "delta_P_Pa(t-2)": delta_p_lag2,
            "delta_P_Pa(t-3)": delta_p_lag3,
            "U_error(t-1)": u_error_lag1,
            "U_error(t-2)": u_error_lag2,
            "U_error(t-3)": u_error_lag3,
            "d_dP_error_dt": d_dp_error_dt,
            "d_delta_P_Pa_dt": d_delta_p_dt,
            "t_norm": t_norm,
        }

    def get_scenario_metadata(self) -> List[Dict]:
        """Return metadata for all scenarios (for the frontend dropdown)."""
        meta = []
        for sc in self._scenarios:
            sc_df = self._df[self._df["scenario"] == sc]
            runs = sorted(sc_df["run_id"].unique().tolist())
            duration_s = sc_df.groupby("run_id")["t_s"].max().mean()
            n_points = int(sc_df.groupby("run_id").size().mean())
            meta.append({
                "scenario": sc,
                "label": sc.replace("_", " ").title(),
                "runs": runs,
                "avg_duration_h": round(duration_s / 3600, 1),
                "avg_points": n_points,
            })
        return meta

    # ──────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────

    def _load_combined_dataset(self) -> pd.DataFrame:
        path = self._data_dir / "combined_dataset.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Combined dataset not found at {path}. "
                "Run scripts/generate_dataset.py first."
            )
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
