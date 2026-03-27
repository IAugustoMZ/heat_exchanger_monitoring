"""
simulate.py — single-scenario transient simulation runner
==========================================================

Run one named scenario, save the raw time-series to CSV, and produce a
diagnostic figure showing temperature profiles, frost growth, and the
pressure drop signal over time.

Usage
-----
  python scripts/simulate.py --scenario gradual_freezing
  python scripts/simulate.py --scenario rapid_freezing --outdir data/simulated
  python scripts/simulate.py --list-scenarios

Available scenario names
------------------------
  normal_operation   gradual_freezing   rapid_freezing
  defrost_recovery   partial_blockage
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path (allows running from any working dir)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.heat_exchanger import (
    ShellAndTubeGeometry,
    FluidProperties,
    run_simulation,
)
from src.scenarios import (
    ALL_SCENARIOS,
    SCENARIO_PARTIAL,
    NoiseParameters,
    add_noise,
    results_to_dataframe,
    run_partial_blockage_simulation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCENARIO_MAP = {s.name: s for s in ALL_SCENARIOS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a single LNG heat exchanger freezing scenario.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario", "-s",
        default="gradual_freezing",
        help="Scenario name to simulate.",
    )
    p.add_argument(
        "--outdir", "-o",
        default="data/simulated",
        help="Directory where CSV and PNG outputs are written.",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenario names and exit.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation (useful in CI).",
    )
    return p


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_figure(result, outpath: Path) -> None:
    """
    4-panel diagnostic figure:
      top-left  : T_h and T_c axial profiles at t=0, 25%, 50%, 75%, 100%
      top-right : delta_f axial profile at the same time slices
      bot-left  : ΔP total (tube-side) vs time
      bot-right : U_mean and delta_f_max vs time
    """
    t = result.t
    z = result.z
    n_t = len(t)
    snap_indices = np.unique(
        np.round(np.linspace(0, n_t - 1, 5)).astype(int)
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(snap_indices)))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Scenario: {result.scenario_name.replace('_', ' ').title()}\n"
        f"Wall-clock solve time: {result.elapsed_wall_time_s:.1f} s",
        fontsize=13,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # --- panel 1: temperature profiles ---
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, c in zip(snap_indices, colors):
        label = f"t = {t[idx]/3600:.1f} h"
        ax1.plot(z, result.T_h[idx], color=c, lw=1.8, label=label)
        ax1.plot(z, result.T_c[idx], color=c, lw=1.8, ls="--")
    ax1.set_xlabel("Axial position  z  [m]")
    ax1.set_ylabel("Temperature  [K]")
    ax1.set_title("Temperature profiles (solid=gas, dashed=LNG)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # --- panel 2: frost thickness profiles ---
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, c in zip(snap_indices, colors):
        label = f"t = {t[idx]/3600:.1f} h"
        ax2.plot(z, result.delta_f[idx] * 1e3, color=c, lw=1.8, label=label)
    ax2.set_xlabel("Axial position  z  [m]")
    ax2.set_ylabel("Frost thickness  δ_f  [mm]")
    ax2.set_title("CO₂ frost layer profile")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # --- panel 3: ΔP time series ---
    ax3 = fig.add_subplot(gs[1, 0])
    t_h = t / 3600
    ax3.plot(t_h, result.delta_P_total / 1e3, color="tab:red", lw=2)
    ax3.set_xlabel("Time  [h]")
    ax3.set_ylabel("Tube-side ΔP  [kPa]")
    ax3.set_title("Pressure drop signal (ML target)")
    ax3.grid(alpha=0.3)

    # Mark 20 % and 50 % increase thresholds
    dP0 = result.delta_P_total[0]
    for pct, style in [(0.2, "--"), (0.5, ":")]:
        ax3.axhline(dP0 * (1 + pct) / 1e3, color="gray", ls=style,
                    label=f"+{int(pct*100)} % alarm")
    ax3.legend(fontsize=8)

    # --- panel 4: U_mean and delta_f_max ---
    ax4 = fig.add_subplot(gs[1, 1])
    color_u = "tab:blue"
    color_f = "tab:orange"
    ax4.plot(t_h, result.U_mean, color=color_u, lw=2, label="U_mean")
    ax4.set_xlabel("Time  [h]")
    ax4.set_ylabel("U_mean  [W/m²K]", color=color_u)
    ax4.tick_params(axis="y", labelcolor=color_u)
    ax4.grid(alpha=0.3)

    ax4b = ax4.twinx()
    ax4b.plot(t_h, result.delta_f_max * 1e3, color=color_f, lw=2,
              ls="--", label="δ_f_max")
    ax4b.set_ylabel("Max frost  δ_f_max  [mm]", color=color_f)
    ax4b.tick_params(axis="y", labelcolor=color_f)
    ax4.set_title("Heat transfer degradation")

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", outpath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_scenarios:
        print("Available scenarios:")
        for s in ALL_SCENARIOS:
            print(f"  {s.name:<22}  {s.description[:70]}...")
        return 0

    scenario_name = args.scenario
    if scenario_name not in SCENARIO_MAP:
        logger.error(
            "Unknown scenario '%s'. Use --list-scenarios to see options.",
            scenario_name,
        )
        return 1

    scenario = SCENARIO_MAP[scenario_name]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    geo = ShellAndTubeGeometry()
    fld = FluidProperties()

    # --- Run simulation ---
    if scenario_name == "partial_blockage":
        result = run_partial_blockage_simulation(scenario, geo, fld)
    else:
        result = run_simulation(geo, fld, scenario.operating, scenario_name)

    if not result.success:
        logger.error(
            "Simulation failed for scenario '%s': %s",
            scenario_name, result.solver_message,
        )
        return 2

    # --- Apply sensor noise ---
    noise = scenario.noise
    noisy = add_noise(result, noise)
    df = results_to_dataframe(noisy)

    # Add alarm label: ΔP > 50 % above initial is a freezing event
    dP_ref = float(result.delta_P_total[0])
    df["freezing_alarm"] = (df["delta_P_Pa"] > 1.5 * dP_ref).astype(int)

    # Save CSV
    csv_path = outdir / f"{scenario_name}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    logger.info(
        "CSV saved → %s  (%d rows × %d cols)",
        csv_path, len(df), len(df.columns),
    )

    # Print summary
    print(f"\n{'─'*55}")
    print(f"  Scenario       : {scenario_name}")
    print(f"  Duration       : {result.t[-1]/3600:.1f} h")
    print(f"  ΔP initial     : {result.delta_P_total[0]:.0f} Pa")
    print(f"  ΔP final       : {result.delta_P_total[-1]:.0f} Pa  "
          f"({(result.delta_P_total[-1]/result.delta_P_total[0]-1)*100:.1f} % increase)")
    print(f"  δ_f max        : {result.delta_f_max[-1]*1e3:.3f} mm")
    print(f"  U_mean final   : {result.U_mean[-1]:.1f} W/m²K")
    print(f"  Freezing alarms: {int(df['freezing_alarm'].sum())} / {len(df)} time steps")
    print(f"{'─'*55}\n")

    # --- Plot ---
    if not args.no_plot:
        png_path = outdir / f"{scenario_name}.png"
        _make_figure(result, png_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
