"""
Exploratory Data Analysis — LNG Heat Exchanger Freezing Monitoring
===================================================================

This script performs a thorough EDA on simulation data of CO₂ freeze-out
in a shell-and-tube heat exchanger.  It follows the hybrid-modelling
methodology from the PE capstone project:

    KPI_actual = KPI_ideal(first-principles) + Error(ML)

Workflow
--------
1. Load and validate the combined dataset
2. Compute **ideal** (clean-tube) baselines from first-principles
3. Compute error signals  error = actual − ideal
4. Descriptive statistics and data-quality summary
5. Time-series visualisation per scenario
6. Error distribution analysis (skewness, transformations)
7. Correlation analysis (Pearson, Spearman)
8. Temporal dependency analysis (ACF)
9. Feature-engineering assessment for ML
10. Export summary statistics as JSON for the report

All plots are saved to  eda/plots/  and a JSON summary is written to
eda/eda_summary.json  for automated report generation.

Re-runnability
--------------
Simply re-run this script after regenerating data (e.g. with more runs):

    python eda/eda_analysis.py [--data-path <csv>] [--threshold <Pa>]

The script discovers columns and scenarios dynamically; no hard-coded
row counts or scenario lists.

Usage
-----
    cd <repo_root>
    python eda/eda_analysis.py
    python eda/eda_analysis.py --data-path data/simulated/combined_dataset.csv
    python eda/eda_analysis.py --threshold 930  # custom alarm threshold in Pa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripted execution
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import src.*
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.correlations import (
    equivalent_diameter_shell,
    friction_factor_churchill,
    heat_transfer_coefficient,
    nusselt_dittus_boelter,
    nusselt_kern_shell,
    overall_heat_transfer_coefficient,
    prandtl_number,
    pressure_drop_per_unit_length,
    reynolds_number,
)
from src.heat_exchanger import FluidProperties, ShellAndTubeGeometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Nominal mass-flow rates per scenario (from src/scenarios.py definitions).
# For perturbed runs the actual flow may differ by ≈ 5 %; this is accounted
# for in measurement-uncertainty propagation (Section 9).
SCENARIO_FLOWS: dict[str, dict[str, float]] = {
    "normal_operation":  {"mdot_h": 5.0, "mdot_c": 8.0},
    "gradual_freezing":  {"mdot_h": 5.0, "mdot_c": 8.0},
    "rapid_freezing":    {"mdot_h": 5.0, "mdot_c": 8.0},
    "defrost_recovery":  {"mdot_h": 4.0, "mdot_c": 5.0},
    "partial_blockage":  {"mdot_h": 5.0, "mdot_c": 8.0},
}

# Measurement uncertainty model (1σ values, from src/scenarios.py NoiseParameters)
SENSOR_NOISE = {
    "T_K":     0.5,     # K  — RTD class A absolute
    "dP_frac": 0.002,   # 0.2 % of reading
    "mdot_frac": 0.05,  # 5 % perturbation used in dataset generation
}

# Consistent colour palette for scenarios
SCENARIO_COLOURS: dict[str, str] = {
    "normal_operation":  "#2196F3",   # blue
    "gradual_freezing":  "#FF9800",   # orange
    "rapid_freezing":    "#F44336",   # red
    "defrost_recovery":  "#4CAF50",   # green
    "partial_blockage":  "#9C27B0",   # purple
}

# Plot styling
PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
}
plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# Data classes for clarity
# ---------------------------------------------------------------------------

@dataclass
class IdealBaseline:
    """Clean-tube (δ_f = 0) first-principles predictions."""
    U_ideal: float           # [W/m²K] overall HTC, no frost
    dP_ideal: float          # [Pa]    tube-side pressure drop, clean
    T_h_out_ideal: float     # [K]     gas outlet temperature, clean
    T_c_out_ideal: float     # [K]     LNG outlet temperature, clean
    Q_ideal: float           # [W]     heat duty, clean
    effectiveness: float     # [-]     ε = Q / Q_max


# ===================================================================
# SECTION 1 — Data Loading & Validation
# ===================================================================

def load_and_validate(data_path: Path) -> pd.DataFrame:
    """Load the combined dataset and perform sanity checks."""
    logger.info("Loading data from %s", data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run `python scripts/generate_dataset.py` first."
        )

    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # ---- Required columns -------------------------------------------------
    required = {
        "scenario", "run_id", "t_s",
        "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K",
        "delta_P_Pa", "U_mean_W_m2K",
        "delta_f_mean_m", "delta_f_max_m",
        "freezing_alarm",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # ---- Null check -------------------------------------------------------
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Null values detected:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("No null values found — dataset is complete.")

    # ---- Physical sanity check (temperatures in Kelvin) -------------------
    for col in ["T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K"]:
        lo, hi = df[col].min(), df[col].max()
        if lo < 50 or hi > 500:
            logger.warning(
                "Column %s has values outside [50, 500] K: [%.1f, %.1f]",
                col, lo, hi,
            )

    # ---- Convert time to hours for readability ----------------------------
    df["t_h"] = df["t_s"] / 3600.0

    # ---- Create a unique group key for per-run analysis -------------------
    df["group"] = df["scenario"] + "_run" + df["run_id"].astype(str)

    logger.info(
        "Scenarios: %s | Runs per scenario: %s",
        df["scenario"].unique().tolist(),
        df.groupby("scenario")["run_id"].nunique().to_dict(),
    )
    return df


# ===================================================================
# SECTION 2 — Ideal Model Computation (Clean-Tube Baseline)
# ===================================================================

def compute_ideal_baseline(
    T_h_in: float,
    T_c_in: float,
    mdot_h: float,
    mdot_c: float,
    geo: ShellAndTubeGeometry,
    fld: FluidProperties,
) -> IdealBaseline:
    """
    Compute clean-tube steady-state predictions using the full
    correlation set from ``src/correlations``.

    This is the **simple first-principles model** in the hybrid
    methodology — it uses geometry + known fluid properties but
    does NOT need any frost-kinetics parameters.

    Parameters
    ----------
    T_h_in, T_c_in : float [K]   inlet temperatures
    mdot_h, mdot_c : float [kg/s] mass flow rates
    geo : ShellAndTubeGeometry    fixed exchanger geometry
    fld : FluidProperties         fluid physical properties

    Returns
    -------
    IdealBaseline   all clean-tube KPIs
    """
    # ---- Tube-side HTC (h_h) ----
    D_h = geo.tube_id               # clean hydraulic diameter
    A_flow = geo.total_clean_flow_area
    rel_rough = geo.tube_roughness / D_h

    Re_h = reynolds_number(mdot_h, D_h, fld.mu_h, A_flow)
    Pr_h = prandtl_number(fld.cp_h, fld.mu_h, fld.k_h)
    Nu_h = nusselt_dittus_boelter(Re_h, Pr_h, heating=False)
    h_h = heat_transfer_coefficient(Nu_h, fld.k_h, D_h)

    # ---- Shell-side HTC (h_c) ----
    D_e = equivalent_diameter_shell(geo.pitch, geo.tube_od, geo.tube_layout)
    A_shell = geo.shell_cross_flow_area
    G_s = mdot_c / A_shell
    Re_s = G_s * D_e / fld.mu_c
    Pr_c = prandtl_number(fld.cp_c, fld.mu_c, fld.k_c)
    Nu_c = nusselt_kern_shell(float(Re_s), Pr_c)
    h_c = heat_transfer_coefficient(Nu_c, fld.k_c, D_e)

    # ---- Overall U (clean — no frost) ----
    U_ideal = overall_heat_transfer_coefficient(
        h_tube=float(h_h),
        h_shell=float(h_c),
        frost_thickness=0.0,
        k_frost=fld.k_frost,
        wall_thickness=geo.wall_thickness,
        k_wall=geo.k_wall,
    )

    # ---- Effectiveness–NTU (counter-flow) ----
    A_ht = np.pi * D_h * geo.tube_length * geo.n_tubes
    C_h = mdot_h * fld.cp_h
    C_c = mdot_c * fld.cp_c
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max if C_max > 0 else 0.0

    NTU = float(U_ideal) * A_ht / C_min

    if abs(C_r - 1.0) < 1e-10:
        eps = NTU / (1.0 + NTU)
    else:
        exp_term = np.exp(-NTU * (1.0 - C_r))
        eps = (1.0 - exp_term) / (1.0 - C_r * exp_term)

    eps = float(np.clip(eps, 0.0, 1.0))
    Q_ideal = eps * C_min * (T_h_in - T_c_in)

    T_h_out_ideal = T_h_in - Q_ideal / C_h
    T_c_out_ideal = T_c_in + Q_ideal / C_c

    # ---- Tube-side ΔP (clean) ----
    G_h = mdot_h / A_flow
    f_D = friction_factor_churchill(float(Re_h), rel_rough)
    dP_per_m = pressure_drop_per_unit_length(f_D, G_h, fld.rho_h, D_h)
    dP_ideal = float(dP_per_m) * geo.tube_length

    return IdealBaseline(
        U_ideal=float(U_ideal),
        dP_ideal=dP_ideal,
        T_h_out_ideal=T_h_out_ideal,
        T_c_out_ideal=T_c_out_ideal,
        Q_ideal=Q_ideal,
        effectiveness=eps,
    )


def add_ideal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, compute clean-tube ideal baseline and add error columns.

    New columns added
    -----------------
    U_ideal, dP_ideal, T_h_out_ideal, T_c_out_ideal, Q_ideal, eps_ideal
    U_error, dP_error, T_h_out_error, T_c_out_error

    Measurement uncertainty columns (1σ propagated)
    ------------------------------------------------
    U_ideal_unc, dP_ideal_unc
    """
    geo = ShellAndTubeGeometry()
    fld = FluidProperties()

    # Pre-compute per (scenario, run_id) since inlet conditions are
    # approximately constant within a run (temps vary due to noise, but
    # flows are constant).  We use the time-averaged inlet temperatures
    # for the ideal model.
    results = {}
    for (scenario, run_id), grp in df.groupby(["scenario", "run_id"]):
        flows = SCENARIO_FLOWS.get(scenario, {"mdot_h": 5.0, "mdot_c": 8.0})
        mdot_h = flows["mdot_h"]
        mdot_c = flows["mdot_c"]

        # Use median inlet temps (robust to noise spikes)
        T_h_in = grp["T_h_in_K"].median()
        T_c_in = grp["T_c_in_K"].median()

        try:
            baseline = compute_ideal_baseline(T_h_in, T_c_in, mdot_h, mdot_c, geo, fld)
            results[(scenario, run_id)] = baseline
        except Exception as e:
            logger.error(
                "Failed to compute ideal baseline for %s run %d: %s",
                scenario, run_id, e,
            )
            # Fallback: use NaN so the row is flagged but not lost
            results[(scenario, run_id)] = IdealBaseline(
                U_ideal=np.nan, dP_ideal=np.nan,
                T_h_out_ideal=np.nan, T_c_out_ideal=np.nan,
                Q_ideal=np.nan, effectiveness=np.nan,
            )

    # Map baselines back to every row
    ideal_cols = {
        "U_ideal": [], "dP_ideal": [], "T_h_out_ideal": [], "T_c_out_ideal": [],
        "Q_ideal": [], "eps_ideal": [],
    }
    for _, row in df.iterrows():
        bl = results[(row["scenario"], row["run_id"])]
        ideal_cols["U_ideal"].append(bl.U_ideal)
        ideal_cols["dP_ideal"].append(bl.dP_ideal)
        ideal_cols["T_h_out_ideal"].append(bl.T_h_out_ideal)
        ideal_cols["T_c_out_ideal"].append(bl.T_c_out_ideal)
        ideal_cols["Q_ideal"].append(bl.Q_ideal)
        ideal_cols["eps_ideal"].append(bl.effectiveness)

    for col, vals in ideal_cols.items():
        df[col] = vals

    # ---- Error columns (actual − ideal) ----
    df["U_error"] = df["U_mean_W_m2K"] - df["U_ideal"]
    df["dP_error"] = df["delta_P_Pa"] - df["dP_ideal"]
    df["T_h_out_error"] = df["T_h_out_K"] - df["T_h_out_ideal"]
    df["T_c_out_error"] = df["T_c_out_K"] - df["T_c_out_ideal"]

    # ---- Measurement Uncertainty Propagation (1σ) ----
    # δU_ideal ≈ ∂U/∂T_in * σ_T  (first-order Taylor)
    # For ΔP: σ_dP = dP_frac * dP_reading + contribution from flow uncertainty
    df["dP_meas_unc"] = np.sqrt(
        (SENSOR_NOISE["dP_frac"] * df["delta_P_Pa"]) ** 2
    )
    # Error uncertainty combines measurement noise and ideal-model parameter
    # uncertainty (flow rate perturbation effect)
    # dP_ideal_unc ≈  (∂dP/∂mdot) * σ_mdot ≈ 2 * dP_ideal * σ_mdot/mdot
    # (since dP ∝ G² ∝ mdot², ∂dP/∂mdot = 2*dP/mdot)
    df["dP_ideal_unc"] = 2.0 * df["dP_ideal"] * SENSOR_NOISE["mdot_frac"]
    df["dP_error_unc"] = np.sqrt(df["dP_meas_unc"] ** 2 + df["dP_ideal_unc"] ** 2)

    logger.info("Added %d ideal + error columns.", len(ideal_cols) + 4)
    return df


# ===================================================================
# SECTION 3 — Descriptive Statistics
# ===================================================================

def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute and return summary statistics as a nested dict."""
    stats: dict = {}

    # ---- Per-scenario summary ----
    numeric_cols = [
        "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K",
        "delta_P_Pa", "U_mean_W_m2K",
        "delta_f_mean_m", "delta_f_max_m",
        "U_ideal", "dP_ideal", "U_error", "dP_error",
    ]
    scenario_stats = {}
    for scenario, grp in df.groupby("scenario"):
        scenario_stats[scenario] = {
            "n_rows": int(len(grp)),
            "n_runs": int(grp["run_id"].nunique()),
            "duration_h": round(float(grp["t_h"].max()), 2),
            "freezing_alarm_rate": round(float(grp["freezing_alarm"].mean()), 4),
        }
        for col in numeric_cols:
            if col in grp.columns:
                scenario_stats[scenario][col] = {
                    "mean": round(float(grp[col].mean()), 4),
                    "std": round(float(grp[col].std()), 4),
                    "min": round(float(grp[col].min()), 4),
                    "max": round(float(grp[col].max()), 4),
                }
    stats["per_scenario"] = scenario_stats

    # ---- Overall ----
    stats["overall"] = {
        "total_rows": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_runs_total": int(df.groupby(["scenario", "run_id"]).ngroups),
        "columns": list(df.columns),
    }

    # ---- Error signal statistics ----
    from scipy import stats as sp_stats

    error_stats = {}
    for col in ["U_error", "dP_error", "T_h_out_error", "T_c_out_error"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        error_stats[col] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "skewness": round(float(sp_stats.skew(vals)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(vals)), 4),
            "shapiro_p": round(float(sp_stats.shapiro(
                vals.sample(min(len(vals), 5000), random_state=42)
            ).pvalue), 6),
        }
    stats["error_signals"] = error_stats

    return stats


# ===================================================================
# SECTION 4 — Plotting Functions
# ===================================================================

def _save_fig(fig: plt.Figure, plot_dir: Path, name: str) -> None:
    """Save figure as PNG and close."""
    path = plot_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved plot: %s", path.name)


# ---- 4a. Time-series overview per scenario ----

def plot_timeseries_overview(df: pd.DataFrame, plot_dir: Path) -> None:
    """Plot all key variables over time, one subplot grid per scenario."""
    variables = [
        ("T_h_in_K", "T_h,in [K]"),
        ("T_h_out_K", "T_h,out [K]"),
        ("T_c_in_K", "T_c,in [K]"),
        ("T_c_out_K", "T_c,out [K]"),
        ("delta_P_Pa", "ΔP [Pa]"),
        ("U_mean_W_m2K", "U_mean [W/m²K]"),
        ("delta_f_max_m", "δ_f,max [m]"),
    ]

    for scenario, grp in df.groupby("scenario"):
        fig, axes = plt.subplots(len(variables), 1, figsize=(12, 2.5 * len(variables)),
                                 sharex=True)
        fig.suptitle(f"Time Series — {scenario}", fontsize=14, y=1.01)

        for ax, (col, ylabel) in zip(axes, variables):
            for run_id, run_grp in grp.groupby("run_id"):
                ax.plot(run_grp["t_h"], run_grp[col],
                        label=f"run {run_id}", alpha=0.8)
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", framealpha=0.7)

        axes[-1].set_xlabel("Time [h]")
        fig.tight_layout()
        _save_fig(fig, plot_dir, f"ts_overview_{scenario}")


# ---- 4b. Scenario comparison overlay ----

def plot_scenario_comparison(df: pd.DataFrame, plot_dir: Path) -> None:
    """Overlay all scenarios on a single multi-panel figure for key KPIs."""
    kpis = [
        ("delta_P_Pa", "ΔP [Pa]"),
        ("U_mean_W_m2K", "U_mean [W/m²K]"),
        ("delta_f_max_m", "δ_f,max [m]"),
    ]

    fig, axes = plt.subplots(len(kpis), 1, figsize=(12, 3.5 * len(kpis)))
    fig.suptitle("Scenario Comparison — Key KPIs", fontsize=14)

    for ax, (col, ylabel) in zip(axes, kpis):
        for scenario, grp in df.groupby("scenario"):
            # Use run 1 for clean comparison
            run1 = grp[grp["run_id"] == 1]
            colour = SCENARIO_COLOURS.get(scenario, "#888888")
            ax.plot(run1["t_h"], run1[col],
                    label=scenario, color=colour, alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", framealpha=0.7)

    axes[-1].set_xlabel("Time [h]")
    fig.tight_layout()
    _save_fig(fig, plot_dir, "scenario_comparison")


# ---- 4c. Ideal vs Actual overlay ----

def plot_ideal_vs_actual(df: pd.DataFrame, plot_dir: Path) -> None:
    """Overlay ideal baseline predictions vs actual measurements."""
    pairs = [
        ("U_mean_W_m2K", "U_ideal", "U [W/m²K]"),
        ("delta_P_Pa", "dP_ideal", "ΔP [Pa]"),
        ("T_h_out_K", "T_h_out_ideal", "T_h,out [K]"),
        ("T_c_out_K", "T_c_out_ideal", "T_c,out [K]"),
    ]

    for scenario, grp in df.groupby("scenario"):
        fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 3 * len(pairs)),
                                 sharex=True)
        fig.suptitle(f"Ideal (clean) vs Actual — {scenario}", fontsize=14, y=1.01)

        run1 = grp[grp["run_id"] == 1]

        for ax, (actual_col, ideal_col, ylabel) in zip(axes, pairs):
            ax.plot(run1["t_h"], run1[actual_col],
                    label="Actual", color="#F44336", alpha=0.8)
            ax.axhline(run1[ideal_col].iloc[0], color="#2196F3",
                       linestyle="--", linewidth=1.5, label="Ideal (clean)")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", framealpha=0.7)

        axes[-1].set_xlabel("Time [h]")
        fig.tight_layout()
        _save_fig(fig, plot_dir, f"ideal_vs_actual_{scenario}")


# ---- 4d. Error evolution ----

def plot_error_evolution(df: pd.DataFrame, plot_dir: Path) -> None:
    """Plot error signals (actual − ideal) over time."""
    error_cols = [
        ("U_error", "U error [W/m²K]"),
        ("dP_error", "ΔP error [Pa]"),
        ("T_h_out_error", "T_h,out error [K]"),
        ("T_c_out_error", "T_c,out error [K]"),
    ]

    fig, axes = plt.subplots(len(error_cols), 1, figsize=(14, 3 * len(error_cols)))
    fig.suptitle("Error Signals (Actual − Ideal) — All Scenarios", fontsize=14)

    for ax, (col, ylabel) in zip(axes, error_cols):
        for scenario, grp in df.groupby("scenario"):
            run1 = grp[grp["run_id"] == 1]
            colour = SCENARIO_COLOURS.get(scenario, "#888888")
            ax.plot(run1["t_h"], run1[col],
                    label=scenario, color=colour, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.legend(loc="best", framealpha=0.7)

    axes[-1].set_xlabel("Time [h]")
    fig.tight_layout()
    _save_fig(fig, plot_dir, "error_evolution")


# ---- 4e. Error distribution histograms ----

def plot_error_distributions(df: pd.DataFrame, plot_dir: Path) -> None:
    """Histogram + KDE of error signals, overall and per-scenario."""
    from scipy.stats import norm

    error_cols = ["U_error", "dP_error", "T_h_out_error", "T_c_out_error"]

    # Overall distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Error Signal Distributions (All Data)", fontsize=14)
    axes = axes.ravel()

    for ax, col in zip(axes, error_cols):
        vals = df[col].dropna()
        ax.hist(vals, bins=50, density=True, alpha=0.7,
                color="#2196F3", edgecolor="white")

        # Fit normal for reference
        mu, sigma = float(vals.mean()), float(vals.std())
        x = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(x, norm.pdf(x, mu, sigma), "r--", linewidth=1.5,
                label=f"N({mu:.1f}, {sigma:.1f})")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    _save_fig(fig, plot_dir, "error_distributions_overall")

    # Per-scenario dP_error — the most important signal
    fig, axes = plt.subplots(1, len(df["scenario"].unique()), figsize=(18, 4))
    fig.suptitle("ΔP Error Distribution by Scenario", fontsize=14)

    for ax, (scenario, grp) in zip(axes, df.groupby("scenario")):
        vals = grp["dP_error"].dropna()
        ax.hist(vals, bins=30, density=True, alpha=0.7,
                color=SCENARIO_COLOURS.get(scenario, "#888"), edgecolor="white")
        ax.set_title(scenario, fontsize=9)
        ax.set_xlabel("ΔP error [Pa]")

    fig.tight_layout()
    _save_fig(fig, plot_dir, "dP_error_by_scenario")


# ---- 4f. Correlation heatmaps ----

def plot_correlation_heatmaps(df: pd.DataFrame, plot_dir: Path) -> None:
    """Pearson and Spearman correlation heatmaps for error-relevant variables."""
    corr_cols = [
        "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K",
        "delta_P_Pa", "U_mean_W_m2K",
        "delta_f_mean_m", "delta_f_max_m",
        "U_ideal", "dP_ideal",
        "U_error", "dP_error",
        "T_h_out_error", "T_c_out_error",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]

    for method in ["pearson", "spearman"]:
        corr_matrix = df[corr_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_cols, fontsize=8)

        # Annotate cells
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_matrix.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label=f"{method.title()} Correlation")
        ax.set_title(f"{method.title()} Correlation Heatmap", fontsize=14)
        fig.tight_layout()
        _save_fig(fig, plot_dir, f"correlation_{method}")


# ---- 4g. Autocorrelation of error signals ----

def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function for lags 0..max_lag."""
    n = len(x)
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")
    acf = acf[n - 1:]  # positive lags only
    acf = acf / acf[0] if acf[0] != 0 else acf
    return acf[: max_lag + 1]


def plot_autocorrelation(df: pd.DataFrame, plot_dir: Path) -> None:
    """ACF of dP_error for freezing scenarios.

    Shows temporal dependency — critical for deciding lag features.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Autocorrelation — ΔP Error Signal", fontsize=14)
    axes = axes.ravel()

    freezing_scenarios = [
        s for s in df["scenario"].unique()
        if s != "defrost_recovery"
    ]

    for idx, scenario in enumerate(freezing_scenarios[:4]):
        ax = axes[idx]
        grp = df[(df["scenario"] == scenario) & (df["run_id"] == 1)]
        vals = grp["dP_error"].dropna().values

        if len(vals) < 20:
            ax.set_title(f"{scenario} — insufficient data")
            continue

        max_lag = min(len(vals) - 1, 100)
        acf = _autocorrelation(vals, max_lag)
        lags = np.arange(max_lag + 1)

        ax.bar(lags, acf, width=0.8, alpha=0.7, color="#2196F3")
        # 95% CI for white noise
        ci = 1.96 / np.sqrt(len(vals))
        ax.axhline(ci, color="red", linestyle="--", linewidth=0.8, label="95% CI")
        ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("Lag (time steps)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, plot_dir, "acf_dP_error")


# ---- 4h. ΔP alarm threshold analysis ----

def plot_threshold_analysis(
    df: pd.DataFrame,
    threshold_pa: float,
    plot_dir: Path,
) -> dict:
    """
    Analyse time-to-threshold for different scenarios.

    The threshold is user-configurable for the future application.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: ΔP evolution with threshold line
    ax = axes[0]
    times_to_threshold: dict[str, list[float | None]] = {}

    for scenario, grp in df.groupby("scenario"):
        colour = SCENARIO_COLOURS.get(scenario, "#888")
        for run_id, run_grp in grp.groupby("run_id"):
            run_grp = run_grp.sort_values("t_h")
            ax.plot(run_grp["t_h"], run_grp["delta_P_Pa"],
                    color=colour, alpha=0.5, linewidth=0.8)

            # Time to threshold
            above = run_grp[run_grp["delta_P_Pa"] >= threshold_pa]
            t_cross = float(above["t_h"].iloc[0]) if len(above) > 0 else None

            if scenario not in times_to_threshold:
                times_to_threshold[scenario] = []
            times_to_threshold[scenario].append(t_cross)

    # legend — one entry per scenario
    for scenario, colour in SCENARIO_COLOURS.items():
        if scenario in df["scenario"].unique():
            ax.plot([], [], color=colour, label=scenario)

    ax.axhline(threshold_pa, color="red", linewidth=2, linestyle="--",
               label=f"Threshold ({threshold_pa:.0f} Pa)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("ΔP [Pa]")
    ax.set_title("ΔP Evolution vs Alarm Threshold")
    ax.legend(fontsize=8)

    # Right panel: bar chart of time-to-threshold by scenario
    ax2 = axes[1]
    scenarios_sorted = sorted(times_to_threshold.keys())
    means = []
    labels = []
    colors = []
    for s in scenarios_sorted:
        vals = [v for v in times_to_threshold[s] if v is not None]
        if vals:
            means.append(np.mean(vals))
        else:
            means.append(0)
        labels.append(s.replace("_", "\n"))
        colors.append(SCENARIO_COLOURS.get(s, "#888"))

    bars = ax2.bar(labels, means, color=colors, alpha=0.8, edgecolor="white")
    ax2.set_ylabel("Time to Threshold [h]")
    ax2.set_title("Mean Time to Reach Alarm Threshold")
    for bar, val in zip(bars, means):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}h", ha="center", va="bottom", fontsize=9)
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, 0.1,
                     "N/A", ha="center", va="bottom", fontsize=9, color="gray")

    fig.tight_layout()
    _save_fig(fig, plot_dir, "threshold_analysis")

    return times_to_threshold


# ---- 4i. Measurement uncertainty visualisation ----

def plot_uncertainty_bands(df: pd.DataFrame, plot_dir: Path) -> None:
    """Show error signals with measurement uncertainty bands."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Error Signals with Measurement Uncertainty (±1σ)", fontsize=14)

    for scenario in ["gradual_freezing", "rapid_freezing"]:
        grp = df[(df["scenario"] == scenario) & (df["run_id"] == 1)]
        if grp.empty:
            continue

        colour = SCENARIO_COLOURS.get(scenario, "#888")

        for idx, (col, unc_col, ylabel) in enumerate([
            ("dP_error", "dP_error_unc", "ΔP error [Pa]"),
        ]):
            ax = axes[idx if scenario == "gradual_freezing" else idx + 1]
            ax.plot(grp["t_h"], grp[col], color=colour, label=scenario, alpha=0.9)
            ax.fill_between(
                grp["t_h"],
                grp[col] - grp[unc_col],
                grp[col] + grp[unc_col],
                alpha=0.2, color=colour,
            )
            ax.set_ylabel(ylabel)
            ax.set_title(f"ΔP Error — {scenario}")
            ax.legend()

    axes[-1].set_xlabel("Time [h]")
    fig.tight_layout()
    _save_fig(fig, plot_dir, "uncertainty_bands")


# ---- 4j. Feature correlation with frost thickness ----

def plot_feature_vs_frost(df: pd.DataFrame, plot_dir: Path) -> None:
    """Scatter plots of observable features vs frost thickness (ground truth)."""
    features = [
        ("delta_P_Pa", "ΔP [Pa]"),
        ("U_mean_W_m2K", "U [W/m²K]"),
        ("dP_error", "ΔP error [Pa]"),
        ("U_error", "U error [W/m²K]"),
        ("T_h_out_error", "T_h,out error [K]"),
        ("T_c_out_error", "T_c,out error [K]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Feature Correlation with Frost Thickness", fontsize=14)
    axes = axes.ravel()

    # Exclude defrost (frost is shrinking, different dynamics)
    df_fz = df[df["scenario"] != "defrost_recovery"]

    for ax, (col, label) in zip(axes, features):
        for scenario, grp in df_fz.groupby("scenario"):
            colour = SCENARIO_COLOURS.get(scenario, "#888")
            ax.scatter(grp["delta_f_max_m"] * 1000, grp[col],
                       s=4, alpha=0.4, color=colour, label=scenario)
        ax.set_xlabel("δ_f,max [mm]")
        ax.set_ylabel(label)

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save_fig(fig, plot_dir, "feature_vs_frost")


# ---- 4k. Rate of change analysis ----

def plot_rate_of_change(df: pd.DataFrame, plot_dir: Path) -> None:
    """Compute and plot rate of change of ΔP — key early-warning feature."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle("Rate of Change of ΔP — Early Warning Indicator", fontsize=14)

    for scenario, grp in df.groupby("scenario"):
        colour = SCENARIO_COLOURS.get(scenario, "#888")
        for run_id, run_grp in grp.groupby("run_id"):
            run_grp = run_grp.sort_values("t_s")
            dt = run_grp["t_s"].diff()
            dP_rate = run_grp["delta_P_Pa"].diff() / dt  # Pa/s

            ax = axes[0]
            ax.plot(run_grp["t_h"].iloc[1:], dP_rate.iloc[1:],
                    color=colour, alpha=0.5, linewidth=0.8)

            # Smoothed (rolling mean, window=10)
            ax = axes[1]
            dP_smooth = dP_rate.rolling(window=10, min_periods=1).mean()
            ax.plot(run_grp["t_h"].iloc[1:], dP_smooth.iloc[1:],
                    color=colour, alpha=0.7, linewidth=1.0)

    for scenario, colour in SCENARIO_COLOURS.items():
        if scenario in df["scenario"].unique():
            axes[0].plot([], [], color=colour, label=scenario)

    axes[0].set_ylabel("dΔP/dt [Pa/s]")
    axes[0].set_title("Raw")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("dΔP/dt [Pa/s] (smoothed)")
    axes[1].set_title("10-point Rolling Average")
    axes[1].set_xlabel("Time [h]")

    fig.tight_layout()
    _save_fig(fig, plot_dir, "dP_rate_of_change")


# ===================================================================
# SECTION 5 — Feature Engineering Assessment
# ===================================================================

def assess_features(df: pd.DataFrame, plot_dir: Path) -> dict:
    """
    Quantify feature relevance for ML error prediction.

    Returns a dict of feature importance metrics.
    """
    from scipy.stats import pearsonr, spearmanr

    # Target: dP_error (most operationally relevant)
    target = "dP_error"
    df_fz = df[df["scenario"] != "defrost_recovery"].copy()

    candidates = [
        "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K",
        "delta_P_Pa", "U_mean_W_m2K",
        "T_h_out_ideal", "T_c_out_ideal", "U_ideal", "dP_ideal",
        "T_h_out_error", "T_c_out_error", "U_error",
    ]
    candidates = [c for c in candidates if c in df_fz.columns]

    importance = {}
    for feat in candidates:
        vals = df_fz[[feat, target]].dropna()
        if len(vals) < 10:
            continue
        # Skip constant features (e.g. U_ideal, dP_ideal which are per-run constants)
        if vals[feat].std() < 1e-12 or vals[target].std() < 1e-12:
            continue
        r_p, p_p = pearsonr(vals[feat], vals[target])
        r_s, p_s = spearmanr(vals[feat], vals[target])
        importance[feat] = {
            "pearson_r": round(float(r_p), 4),
            "pearson_p": float(p_p),
            "spearman_r": round(float(r_s), 4),
            "spearman_p": float(p_s),
        }

    # ---- Lag-1 autocorrelation of target (for lagging decision) ----
    acf_vals = {}
    for scenario, grp in df_fz.groupby("scenario"):
        for run_id, run_grp in grp.groupby("run_id"):
            vals = run_grp[target].dropna().values
            if len(vals) > 5:
                lag1_acf = float(np.corrcoef(vals[:-1], vals[1:])[0, 1])
                acf_vals[f"{scenario}_run{run_id}"] = round(lag1_acf, 4)

    # ---- Visualise feature importances ----
    fig, ax = plt.subplots(figsize=(10, 6))

    feats_sorted = sorted(importance.items(),
                          key=lambda x: abs(x[1]["spearman_r"]), reverse=True)
    names = [f[0] for f in feats_sorted]
    spearman_vals = [f[1]["spearman_r"] for f in feats_sorted]

    colors = ["#F44336" if v < 0 else "#4CAF50" for v in spearman_vals]
    ax.barh(names, spearman_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Spearman Correlation with ΔP Error")
    ax.set_title("Feature Relevance for ΔP Error Prediction")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    _save_fig(fig, plot_dir, "feature_importance")

    return {
        "feature_correlations": importance,
        "lag1_acf_dP_error": acf_vals,
    }


# ===================================================================
# SECTION 6 — Transformation Assessment
# ===================================================================

def assess_transformations(df: pd.DataFrame, plot_dir: Path) -> dict:
    """Test transformations on dP_error for normality (required for linear models)."""
    from scipy.stats import shapiro, boxcox, yeojohnson

    target = df["dP_error"].dropna().values
    results = {}

    # Original
    stat_orig, p_orig = shapiro(target[:5000] if len(target) > 5000 else target)
    results["original"] = {"shapiro_stat": round(float(stat_orig), 6),
                           "shapiro_p": round(float(p_orig), 6)}

    # Log1p (if all positive) or Yeo-Johnson
    try:
        yj_transformed, yj_lambda = yeojohnson(target)
        stat_yj, p_yj = shapiro(
            yj_transformed[:5000] if len(yj_transformed) > 5000 else yj_transformed
        )
        results["yeo_johnson"] = {
            "lambda": round(float(yj_lambda), 4),
            "shapiro_stat": round(float(stat_yj), 6),
            "shapiro_p": round(float(p_yj), 6),
        }
    except Exception as e:
        results["yeo_johnson"] = {"error": str(e)}

    # Square root of absolute value (sign-preserving)
    sqrt_transformed = np.sign(target) * np.sqrt(np.abs(target))
    stat_sq, p_sq = shapiro(
        sqrt_transformed[:5000] if len(sqrt_transformed) > 5000 else sqrt_transformed
    )
    results["sqrt_abs"] = {"shapiro_stat": round(float(stat_sq), 6),
                           "shapiro_p": round(float(p_sq), 6)}

    # ---- Plot transformation comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("ΔP Error — Transformation Comparison", fontsize=14)

    axes[0].hist(target, bins=50, density=True, alpha=0.7, color="#2196F3")
    axes[0].set_title(f"Original (Shapiro p={p_orig:.4f})")

    if "lambda" in results.get("yeo_johnson", {}):
        axes[1].hist(yj_transformed, bins=50, density=True, alpha=0.7, color="#FF9800")
        axes[1].set_title(
            f"Yeo-Johnson λ={yj_lambda:.2f} (p={p_yj:.4f})"
        )
    else:
        axes[1].set_title("Yeo-Johnson — failed")

    axes[2].hist(sqrt_transformed, bins=50, density=True, alpha=0.7, color="#4CAF50")
    axes[2].set_title(f"Sign-preserving √|x| (p={p_sq:.4f})")

    for ax in axes:
        ax.set_xlabel("Transformed value")
        ax.set_ylabel("Density")

    fig.tight_layout()
    _save_fig(fig, plot_dir, "transformation_comparison")

    return results


# ===================================================================
# SECTION 7 — dP Error Trend Analysis (for forecasting feasibility)
# ===================================================================

def analyse_trend_for_forecasting(df: pd.DataFrame, plot_dir: Path) -> dict:
    """
    Analyse the dP_error trend to assess forecasting feasibility.
    Fits simple trend models (linear, quadratic) per scenario.
    """
    trend_results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ΔP Error Trend Model Fits (Forecasting Feasibility)", fontsize=14)
    axes = axes.ravel()

    freezing_scenarios = [
        s for s in df["scenario"].unique()
        if s not in ("defrost_recovery",)
    ]

    for idx, scenario in enumerate(freezing_scenarios[:4]):
        ax = axes[idx]
        grp = df[(df["scenario"] == scenario) & (df["run_id"] == 1)].sort_values("t_s")

        t = grp["t_h"].values
        y = grp["dP_error"].values

        # Linear fit
        try:
            coeffs_lin = np.polyfit(t, y, 1)
            y_lin = np.polyval(coeffs_lin, t)
            residuals_lin = y - y_lin
            r2_lin = 1 - np.sum(residuals_lin**2) / np.sum((y - y.mean())**2)
        except Exception:
            coeffs_lin = [0, 0]
            r2_lin = 0

        # Quadratic fit
        try:
            coeffs_quad = np.polyfit(t, y, 2)
            y_quad = np.polyval(coeffs_quad, t)
            residuals_quad = y - y_quad
            r2_quad = 1 - np.sum(residuals_quad**2) / np.sum((y - y.mean())**2)
        except Exception:
            coeffs_quad = [0, 0, 0]
            r2_quad = 0

        ax.scatter(t, y, s=4, alpha=0.5, color="#2196F3", label="Data")
        ax.plot(t, y_lin, "r--", linewidth=1.5, label=f"Linear R²={r2_lin:.3f}")
        ax.plot(t, y_quad, "g-", linewidth=1.5, label=f"Quad R²={r2_quad:.3f}")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("ΔP error [Pa]")
        ax.set_title(scenario, fontsize=10)
        ax.legend(fontsize=8)

        trend_results[scenario] = {
            "linear_r2": round(float(r2_lin), 4),
            "linear_slope_Pa_per_h": round(float(coeffs_lin[0]), 4),
            "quad_r2": round(float(r2_quad), 4),
        }

    fig.tight_layout()
    _save_fig(fig, plot_dir, "trend_analysis")

    return trend_results


# ===================================================================
# MAIN
# ===================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EDA for LNG Heat Exchanger Freezing Monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path", "-d",
        default="data/simulated/combined_dataset.csv",
        help="Path to the combined dataset CSV.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help=(
            "ΔP alarm threshold in Pa.  If not set, it is computed as 150%% "
            "of the normal-operation baseline ΔP (matching the simulation "
            "alarm logic)."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="eda",
        help="Root output directory for plots and reports.",
    )
    args = parser.parse_args(argv)

    # ---- Paths ----
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EDA — LNG Heat Exchanger Freezing Monitoring")
    logger.info("=" * 60)

    # ---- 1. Load data ----
    df = load_and_validate(data_path)

    # ---- 2. Compute ideal baselines and errors ----
    logger.info("Computing clean-tube ideal baselines...")
    df = add_ideal_columns(df)

    # ---- 3. Determine threshold ----
    if args.threshold is not None:
        threshold_pa = args.threshold
    else:
        # Auto-compute: 150% of normal-operation baseline ΔP
        normal_data = df[df["scenario"] == "normal_operation"]
        if not normal_data.empty:
            baseline_dp = float(normal_data["delta_P_Pa"].median())
            threshold_pa = 1.5 * baseline_dp
        else:
            threshold_pa = 930.0  # fallback
    logger.info("Using ΔP alarm threshold: %.1f Pa", threshold_pa)

    # ---- 4. Descriptive statistics ----
    logger.info("Computing descriptive statistics...")
    stats = compute_statistics(df)
    stats["threshold_pa"] = threshold_pa

    # ---- 5. Plots ----
    logger.info("Generating plots...")
    plot_timeseries_overview(df, plot_dir)
    plot_scenario_comparison(df, plot_dir)
    plot_ideal_vs_actual(df, plot_dir)
    plot_error_evolution(df, plot_dir)
    plot_error_distributions(df, plot_dir)
    plot_correlation_heatmaps(df, plot_dir)
    plot_autocorrelation(df, plot_dir)
    threshold_times = plot_threshold_analysis(df, threshold_pa, plot_dir)
    plot_uncertainty_bands(df, plot_dir)
    plot_feature_vs_frost(df, plot_dir)
    plot_rate_of_change(df, plot_dir)

    # ---- 6. Feature assessment ----
    logger.info("Assessing feature relevance and temporal dependencies...")
    feature_assessment = assess_features(df, plot_dir)
    stats["feature_assessment"] = feature_assessment

    # ---- 7. Transformation assessment ----
    logger.info("Testing error transformations...")
    transform_results = assess_transformations(df, plot_dir)
    stats["transformations"] = transform_results

    # ---- 8. Trend analysis ----
    logger.info("Analysing error trends for forecasting feasibility...")
    trend_results = analyse_trend_for_forecasting(df, plot_dir)
    stats["trend_analysis"] = trend_results

    # ---- 9. Times to threshold ----
    stats["times_to_threshold"] = {
        scenario: [
            round(t, 2) if t is not None else None
            for t in times
        ]
        for scenario, times in threshold_times.items()
    }

    # ---- 10. Export summary ----
    summary_path = output_dir / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Summary statistics saved to %s", summary_path)

    # ---- Export enriched dataset for future ML steps ----
    enriched_path = output_dir / "enriched_dataset.csv"
    df.to_csv(enriched_path, index=False)
    logger.info("Enriched dataset (with ideal + error columns) saved to %s", enriched_path)

    logger.info("=" * 60)
    logger.info("EDA complete. %d plots saved to %s/", len(list(plot_dir.glob("*.png"))), plot_dir)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
