"""
generate_dataset.py — batch simulation runner and dataset assembler
====================================================================

Runs all five operating scenarios N times each (default N=10), perturbing
operating conditions between runs to produce diverse ML training data.
All runs are labelled and assembled into a single dataset ready for training.

Each run:
  * run_id = 1 : nominal conditions (exact scenario definition)
  * run_id > 1 : perturbed conditions (Gaussian noise on T_in, mdot, y_CO2,
                 k_dep, k_rem; intensity = 5 % by default)

Output files (in --outdir, default: data/simulated/)
-----------------------------------------------------
  {scenario_name}.csv          All runs for that scenario (with noise)
  combined_dataset.csv         All scenarios × all runs merged, with labels
  combined_dataset.parquet     Parquet version for faster I/O in ML pipelines

Dataset columns
---------------
  scenario      str        Scenario name
  run_id        int        Run index (1 = nominal, 2+ = perturbed)
  t_s           [s]        Elapsed simulation time
  T_h_in_K      [K]        Gas inlet temperature  (noisy)
  T_h_out_K     [K]        Gas outlet temperature (noisy)
  T_c_in_K      [K]        LNG inlet temperature  (noisy)
  T_c_out_K     [K]        LNG outlet temperature (noisy)
  delta_P_Pa    [Pa]       Tube-side total pressure drop (noisy)
  U_mean_W_m2K  [W/m²K]   Spatial-mean overall HTC (ground truth, not noisy)
  delta_f_max_m [m]        Max frost thickness along z (ground truth label)
  delta_f_mean_m[m]        Spatial-mean frost thickness (ground truth label)
  freezing_alarm int       1 if ΔP exceeds 150 % of t=0 baseline (binary label)
  early_stop    int        1 if simulation terminated early by blockage event

Usage
-----
  python scripts/generate_dataset.py
  python scripts/generate_dataset.py --n-runs 20
  python scripts/generate_dataset.py --scenarios rapid_freezing normal_operation
  python scripts/generate_dataset.py --outdir data/simulated --skip-parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.heat_exchanger import ShellAndTubeGeometry, FluidProperties, run_simulation
from src.scenarios import (
    ALL_SCENARIOS,
    NoiseParameters,
    add_noise,
    results_to_dataframe,
    run_partial_blockage_simulation,
    perturb_operating_conditions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all scenarios and assemble a labelled ML dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--outdir", "-o",
        default="data/simulated",
        help="Output directory for CSV / Parquet files.",
    )
    p.add_argument(
        "--n-runs", "-n",
        type=int,
        default=10,
        help=(
            "Number of simulation runs per scenario. Run 1 uses nominal "
            "conditions; runs 2+ apply a 5 %% Gaussian perturbation to "
            "inlet temperatures, flow rates, CO2 content and kinetic constants."
        ),
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Subset of scenario names to run (default: all five).",
    )
    p.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip Parquet output (useful if pyarrow is not installed).",
    )
    return p


# ---------------------------------------------------------------------------
# Per-scenario / per-run runner
# ---------------------------------------------------------------------------

def _run_one(scenario_name: str, operating, geo: ShellAndTubeGeometry,
             fld: FluidProperties, run_id: int):
    """Run a single (scenario, operating conditions, run_id) tuple."""
    t0 = time.perf_counter()

    if scenario_name == "partial_blockage":
        # Reconstruct a minimal Scenario-like object with the (possibly
        # perturbed) operating conditions so the partial-blockage runner
        # picks them up via scenario.operating.
        from src.scenarios import SCENARIO_PARTIAL, Scenario, NoiseParameters
        from dataclasses import replace as _dc_replace
        tmp_scenario = _dc_replace(SCENARIO_PARTIAL, operating=operating)
        result = run_partial_blockage_simulation(tmp_scenario, geo, fld, run_id=run_id)
    else:
        result = run_simulation(geo, fld, operating, scenario_name, run_id=run_id)

    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geo = ShellAndTubeGeometry()
    fld = FluidProperties()

    # Filter scenarios if --scenarios provided
    scenarios_to_run = ALL_SCENARIOS
    if args.scenarios:
        name_set = set(args.scenarios)
        scenarios_to_run = [s for s in ALL_SCENARIOS if s.name in name_set]
        if not scenarios_to_run:
            logger.error(
                "No matching scenarios found for: %s. "
                "Valid names: %s",
                args.scenarios,
                [s.name for s in ALL_SCENARIOS],
            )
            return 1
        unknown = name_set - {s.name for s in scenarios_to_run}
        if unknown:
            logger.warning("Ignoring unknown scenario names: %s", unknown)

    if args.n_runs < 1:
        logger.error("--n-runs must be >= 1, got %d.", args.n_runs)
        return 1

    all_dfs: list[pd.DataFrame] = []
    total_t0 = time.perf_counter()
    total_runs_attempted = 0
    total_runs_succeeded = 0

    for scenario in scenarios_to_run:
        logger.info(
            "━━━ Scenario: %-22s  (%d runs)  — %s",
            scenario.name, args.n_runs, scenario.description[:55] + "...",
        )

        # Deterministic per-scenario RNG so results are reproducible
        base_rng = np.random.default_rng(
            scenario.noise.rng_seed if scenario.noise.rng_seed is not None else 0
        )

        scenario_dfs: list[pd.DataFrame] = []
        n_success = 0
        n_early_stop = 0

        for run_id in range(1, args.n_runs + 1):
            total_runs_attempted += 1

            # run_id = 1: nominal conditions; run_id > 1: perturbed
            if run_id == 1:
                run_ops = scenario.operating
            else:
                run_ops = perturb_operating_conditions(
                    scenario.operating, rng=base_rng, intensity=0.05
                )

            result, wall_time = _run_one(
                scenario.name, run_ops, geo, fld, run_id=run_id
            )

            if not result.success:
                logger.error(
                    "  ✗ run_id=%-3d FAILED: %s  — skipping.",
                    run_id, result.solver_message,
                )
                continue

            n_success += 1
            total_runs_succeeded += 1
            if result.early_termination:
                n_early_stop += 1

            # Independent noise seed per run so each run has different noise
            run_noise = NoiseParameters(
                sigma_T=scenario.noise.sigma_T,
                sigma_dP_frac=scenario.noise.sigma_dP_frac,
                sigma_mdot_frac=scenario.noise.sigma_mdot_frac,
                rng_seed=None,
            )
            noisy = add_noise(result, run_noise)
            df = results_to_dataframe(noisy)

            # run_id column for traceability
            df.insert(1, "run_id", run_id)

            # Binary alarm label: ΔP > 150 % of t=0 baseline for this run
            dP_ref = float(result.delta_P_total[0])
            df["freezing_alarm"] = (df["delta_P_Pa"] > 1.5 * dP_ref).astype(int)

            # Flag early-stopped runs
            df["early_stop"] = int(result.early_termination)

            scenario_dfs.append(df)

            dP_change_pct = (result.delta_P_total[-1] / dP_ref - 1.0) * 100.0
            logger.info(
                "  ✓ run_id=%-3d | ΔP: %+.1f %%  | δ_f_max: %.3f mm  "
                "| alarms: %d/%d%s  | wall: %.0f s",
                run_id,
                dP_change_pct,
                result.delta_f_max[-1] * 1e3,
                int(df["freezing_alarm"].sum()),
                len(df),
                "  [EARLY STOP]" if result.early_termination else "",
                wall_time,
            )

        if not scenario_dfs:
            logger.error(
                "All runs failed for scenario '%s' — skipping.", scenario.name
            )
            continue

        scenario_combined = pd.concat(scenario_dfs, ignore_index=True)
        csv_path = outdir / f"{scenario.name}.csv"
        scenario_combined.to_csv(csv_path, index=False, float_format="%.6g")
        logger.info(
            "  Saved %s: %d rows (%d/%d runs succeeded, %d early stops)",
            csv_path, len(scenario_combined), n_success, args.n_runs, n_early_stop,
        )
        all_dfs.append(scenario_combined)

    if not all_dfs:
        logger.error("No scenarios completed successfully. Dataset not written.")
        return 2

    # Assemble combined dataset
    combined = pd.concat(all_dfs, ignore_index=True)

    # Canonical column order
    col_order = [
        "scenario", "run_id", "t_s",
        "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K",
        "delta_P_Pa",
        "U_mean_W_m2K", "delta_f_mean_m", "delta_f_max_m",
        "freezing_alarm", "early_stop",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]

    csv_combined = outdir / "combined_dataset.csv"
    combined.to_csv(csv_combined, index=False, float_format="%.6g")
    logger.info("Combined CSV → %s  (%d rows)", csv_combined, len(combined))

    if not args.skip_parquet:
        try:
            parquet_path = outdir / "combined_dataset.parquet"
            combined.to_parquet(parquet_path, index=False)
            logger.info("Combined Parquet → %s", parquet_path)
        except ImportError:
            logger.warning(
                "pyarrow not installed — skipping Parquet export. "
                "Install with: pip install pyarrow"
            )

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'═'*65}")
    print(f"  Dataset generation complete in {total_elapsed:.0f} s")
    print(f"  Runs           : {total_runs_succeeded}/{total_runs_attempted} succeeded")
    print(f"  Rows           : {len(combined):,}")
    print(f"  Scenarios      : {combined['scenario'].nunique()}")
    print(f"  Runs per scen. : up to {args.n_runs}")
    print(f"  Columns        : {list(combined.columns)}")
    print(f"  Alarm ratio    : {combined['freezing_alarm'].mean()*100:.1f} %")
    print(f"  Early-stop rows: {int(combined['early_stop'].sum()):,} "
          f"({combined['early_stop'].mean()*100:.1f} %)")
    print(f"  δ_f_max range  : {combined['delta_f_max_m'].min()*1e3:.4f} – "
          f"{combined['delta_f_max_m'].max()*1e3:.3f} mm")
    print(f"  ΔP range       : {combined['delta_P_Pa'].min():.0f} – "
          f"{combined['delta_P_Pa'].max():.0f} Pa")
    print(f"  Output dir     : {outdir.resolve()}")
    print(f"{'═'*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
