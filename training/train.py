"""
train.py
========
Main training entry point for the LNG Heat Exchanger frost-monitoring ML model.

Methodology (PE Hybrid Approach)
---------------------------------
    ΔP_predicted(t) = ΔP_ideal  +  dP_error_ML(t)

where dP_error_ML is trained to predict the deviation of actual ΔP from the
clean-tube first-principles baseline.

Business Questions Answered
---------------------------
A. Unexpected correlations between input data and ΔP increase
   → Feature importance / coefficients logged per model
B. Proxy for heavy components in feed gas
   → Observable-only features ranked by the model
C. Expected runtime / defrost date
   → Autoregressive forecast plot logged to MLflow

Usage
-----
    cd <repo_root>
    python training/train.py
    python training/train.py --config-dir training/config
    python training/train.py --models lasso ridge elastic_net
    python training/train.py --no-mlflow  # local-only run

MLflow Tracking
---------------
- Parent run per model type
- Each Optuna trial → nested child run
- Best model registered to Model Registry as "hx_frost_dP_error_<model_name>"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.modules.data.loader import DataLoader
from training.modules.data.preprocessor import DataPreprocessor
from training.modules.evaluation.interpretability import InterpretabilityReporter
from training.modules.evaluation.metrics import MetricsCalculator
from training.modules.models.registry import ModelRegistry
from training.modules.transformers.target_transformer import YeoJohnsonTargetTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_configs(config_dir: Path) -> Dict:
    """Load and merge all YAML configs from config_dir."""
    configs = {}
    for name in ("experiment_config", "features_config", "model_config"):
        path = config_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}. "
                "Ensure training/config/ contains all three YAML files."
            )
        with open(path, "r", encoding="utf-8") as f:
            configs[name] = yaml.safe_load(f)
        logger.info("Loaded config: %s", path.name)
    return configs


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
def setup_mlflow(configs: Dict, use_mlflow: bool) -> Optional[object]:
    """Initialise MLflow tracking; return the mlflow module or None."""
    if not use_mlflow:
        logger.info("MLflow disabled — running in local-only mode.")
        return None
    try:
        import mlflow
        tracking_uri = configs["experiment_config"]["mlflow"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)
        exp_name = configs["experiment_config"]["mlflow"]["experiment_name"]
        mlflow.set_experiment(exp_name)

        # Tag the experiment
        tags = configs["experiment_config"]["mlflow"].get("run_tags", {})
        logger.info("MLflow tracking URI: %s | Experiment: %s", tracking_uri, exp_name)
        return mlflow
    except ImportError:
        logger.warning("mlflow not installed — running without tracking.")
        return None
    except Exception as exc:
        logger.warning(
            "MLflow setup failed (%s) — running without tracking. "
            "Start the MLflow server with: docker compose up -d", exc
        )
        return None


# ---------------------------------------------------------------------------
# Core training function for one model
# ---------------------------------------------------------------------------
def train_one_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    target_transformer: YeoJohnsonTargetTransformer,
    registry: ModelRegistry,
    metrics_calc: MetricsCalculator,
    configs: Dict,
    mlflow_module,
    feature_names: List[str],
    time_h_test: np.ndarray,
) -> Dict:
    """
    Full training loop for one model: Optuna tuning → best pipeline fit
    → metrics → interpretability artefacts → MLflow logging.

    Returns
    -------
    dict  Summary of best metrics for this model.
    """
    logger.info("=" * 60)
    logger.info("Training model: %s", model_name)
    logger.info("=" * 60)

    optuna_cfg = configs["experiment_config"].get("optuna", {})
    n_trials = optuna_cfg.get("n_trials", 80)
    timeout = optuna_cfg.get("timeout_seconds", 300)
    direction = optuna_cfg.get("direction", "minimize")
    n_splits = configs["experiment_config"]["cross_validation"].get("n_splits", 5)

    # Transform target
    y_train_tf = target_transformer.transform(y_train_raw)

    parent_run_id: Optional[str] = None
    result_summary: Dict = {"model_name": model_name}

    # ----------------------------------------------------------------
    # Optuna study
    # ----------------------------------------------------------------
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error(
            "optuna not installed. Run: pip install optuna  "
            "or use requirements-training.txt"
        )
        return result_summary

    sampler_name = optuna_cfg.get("sampler", "TPESampler")
    pruner_name = optuna_cfg.get("pruner", "MedianPruner")
    sampler = getattr(optuna.samplers, sampler_name)(seed=configs["experiment_config"].get("random_seed", 42))
    pruner = getattr(optuna.pruners, pruner_name)()

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # ----------------------------------------------------------------
    # MLflow parent run
    # ----------------------------------------------------------------
    ctx_manager = (
        mlflow_module.start_run(run_name=f"{model_name}_training")
        if mlflow_module else _null_context()
    )

    with ctx_manager as active_run:
        parent_run_id = active_run.info.run_id if active_run else None

        objective = registry.make_optuna_objective(
            model_name=model_name,
            X_train=X_train,
            y_train=pd.Series(y_train_tf),
            n_splits=n_splits,
            scoring="neg_mean_absolute_error",
            mlflow_run_id=parent_run_id,
        )

        logger.info(
            "Starting Optuna optimisation: %d trials, timeout=%ds",
            n_trials, timeout
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        best_trial = study.best_trial
        logger.info(
            "Best trial #%d: score=%.4f  params=%s",
            best_trial.number, best_trial.value, best_trial.params,
        )

        # ----------------------------------------------------------------
        # Refit best pipeline on full training set
        # ----------------------------------------------------------------
        best_params = best_trial.params
        k_best = best_params.pop("k_best", 8)
        rfe_n = best_params.pop("rfe_n_features", 4)
        best_pipeline = registry.build_pipeline(
            model_name=model_name,
            k_best=min(k_best, X_train.shape[1]),
            rfe_n_features=min(rfe_n, k_best),
            estimator_params=best_params,
        )
        best_pipeline.fit(X_train, y_train_tf)
        logger.info("Best pipeline fitted on full training set.")

        # ----------------------------------------------------------------
        # Predictions (back-transform to Pa scale)
        # ----------------------------------------------------------------
        y_train_pred_tf = best_pipeline.predict(X_train)
        y_test_pred_tf = best_pipeline.predict(X_test)

        y_train_pred = target_transformer.inverse_transform(y_train_pred_tf)
        y_test_pred = target_transformer.inverse_transform(y_test_pred_tf)

        # Clip to physical range (dP_error ≥ 0)
        y_train_pred = np.clip(y_train_pred, 0, None)
        y_test_pred = np.clip(y_test_pred, 0, None)

        # ----------------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------------
        all_metrics = metrics_calc.compute_all(
            y_train_raw, y_train_pred,
            y_test_raw, y_test_pred,
        )

        threshold_metrics = metrics_calc.time_to_threshold_error(
            time_h=time_h_test,
            dP_error_true=y_test_raw,
            dP_error_pred=y_test_pred,
        )
        all_metrics.update(threshold_metrics)
        result_summary.update(all_metrics)

        # ----------------------------------------------------------------
        # MLflow logging
        # ----------------------------------------------------------------
        if mlflow_module:
            mlflow_module.log_metrics(all_metrics)
            mlflow_module.log_params({
                "model_name": model_name,
                "n_optuna_trials": len(study.trials),
                "best_trial_number": best_trial.number,
                "best_cv_score": best_trial.value,
                "k_best": k_best,
                "rfe_n_features": rfe_n,
                "target_transform_lambda": target_transformer.lambda_fitted_,
                **{f"best_{k}": v for k, v in best_params.items()},
            })
            target_transformer.log_to_mlflow()

        # ----------------------------------------------------------------
        # Interpretability artefacts
        # ----------------------------------------------------------------
        reporter = InterpretabilityReporter(
            model_name=model_name,
            alarm_threshold_pa=metrics_calc.alarm_threshold_pa,
            dP_ideal=metrics_calc.dP_ideal,
        )

        # Feature importance (Business Questions A + B)
        fig_importance, importance_dict = reporter.feature_importance_plot(
            pipeline=best_pipeline,
            feature_names=feature_names,
        )
        result_summary["feature_importance"] = importance_dict

        # Residuals plot
        fig_residuals = reporter.residuals_plot(
            y_train_true=y_train_raw,
            y_train_pred=y_train_pred,
            y_test_true=y_test_raw,
            y_test_pred=y_test_pred,
        )

        # Runtime forecast (Business Question C)
        fig_forecast, forecast_summary = reporter.runtime_forecast_plot(
            pipeline=best_pipeline,
            X_history=X_test,
            y_history=y_test_raw,
            time_h_history=time_h_test,
            reference_datetime=datetime.now(),
        )
        result_summary["forecast"] = forecast_summary

        # Log figures to MLflow
        if mlflow_module:
            _log_figures_to_mlflow(
                mlflow_module,
                {
                    f"{model_name}_feature_importance": fig_importance,
                    f"{model_name}_residuals": fig_residuals,
                    f"{model_name}_runtime_forecast": fig_forecast,
                },
            )
            # Log best pipeline as MLflow model
            try:
                from mlflow.models.signature import infer_signature
                signature = infer_signature(
                    X_train.head(5),
                    best_pipeline.predict(X_train.head(5)),
                )
                mlflow_module.sklearn.log_model(
                    best_pipeline,
                    artifact_path="model",
                    input_example=X_train.head(5),
                    signature=signature,
                    registered_model_name=f"hx_frost_dP_error_{model_name}",
                )
                logger.info("Model registered to MLflow Model Registry as 'hx_frost_dP_error_%s'", model_name)
            except Exception as reg_exc:
                logger.warning("MLflow model registration failed: %s", reg_exc)

            # Log feature importance as JSON artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix=f"{model_name}_importance_"
            ) as tmp:
                json.dump(importance_dict, tmp, indent=2)
                tmp_path = tmp.name
            mlflow_module.log_artifact(tmp_path, artifact_path="interpretability")

            # Log forecast summary as JSON artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix=f"{model_name}_forecast_"
            ) as tmp:
                json.dump(
                    {k: str(v) if v is not None else None for k, v in forecast_summary.items()},
                    tmp, indent=2,
                )
                tmp_path = tmp.name
            mlflow_module.log_artifact(tmp_path, artifact_path="interpretability")

        plt.close("all")

        logger.info(
            "Model '%s' done — test R²=%.4f  MAE=%.2f Pa",
            model_name,
            all_metrics.get("test_r2", float("nan")),
            all_metrics.get("test_mae_pa", float("nan")),
        )

    return result_summary


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    configs = load_configs(config_dir)

    exp_cfg = configs["experiment_config"]
    feat_cfg = configs["features_config"]
    model_cfg = configs["model_config"]

    random_seed = exp_cfg.get("random_seed", 42)
    np.random.seed(random_seed)

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    data_path = _resolve_path(
        exp_cfg["data"]["enriched_dataset_path"], _REPO_ROOT
    )
    loader = DataLoader(
        data_path=data_path,
        drop_columns=feat_cfg.get("drop_columns", []),
    )
    df = loader.load()

    # ----------------------------------------------------------------
    # 2. Feature engineering + train/test split
    # ----------------------------------------------------------------
    train_runs = exp_cfg["cross_validation"].get("train_runs", [1, 2])
    test_runs = exp_cfg["cross_validation"].get("test_runs", [3])

    preprocessor = DataPreprocessor(
        target_col=exp_cfg["data"]["target_column"],
        group_col=exp_cfg["data"]["group_column"],
        scenario_col=exp_cfg["data"]["scenario_column"],
        time_col=exp_cfg["data"]["time_column"],
        train_runs=train_runs,
        test_runs=test_runs,
        lag_columns=feat_cfg["lag_features"]["columns"],
        n_lags=feat_cfg["lag_features"]["n_lags"],
        roc_columns=feat_cfg["rate_of_change"]["columns"],
        roc_window=feat_cfg["rate_of_change"]["window"],
    )

    X_train, X_test, y_train_raw, y_test_raw = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names()

    logger.info("Feature names (%d): %s", len(feature_names), feature_names)

    # Time series for test set (needed for runtime forecast + threshold metrics)
    test_df = df[df["run_id"].isin(test_runs)].reset_index(drop=True)
    time_h_test = test_df[exp_cfg["data"]["time_column"]].values

    # Align test time with X_test (they should match row-for-row)
    if len(time_h_test) != len(X_test):
        logger.warning(
            "time_h_test length (%d) != X_test length (%d) — trimming to min.",
            len(time_h_test), len(X_test),
        )
        min_len = min(len(time_h_test), len(X_test))
        time_h_test = time_h_test[:min_len]
        X_test = X_test.iloc[:min_len]
        y_test_raw = y_test_raw.iloc[:min_len] if hasattr(y_test_raw, "iloc") else y_test_raw[:min_len]

    # ----------------------------------------------------------------
    # 3. Target transformer
    # ----------------------------------------------------------------
    tf_cfg = exp_cfg.get("target_transform", {})
    target_transformer = YeoJohnsonTargetTransformer(
        lambda_=tf_cfg.get("lambda_", 0.61),
        refit=tf_cfg.get("refit", False),
        standardize=False,
    )
    target_transformer.fit(y_train_raw.values)

    # ----------------------------------------------------------------
    # 4. Setup MLflow
    # ----------------------------------------------------------------
    mlflow_module = setup_mlflow(configs, use_mlflow=not args.no_mlflow)

    # ----------------------------------------------------------------
    # 5. Setup model registry + metrics calculator
    # ----------------------------------------------------------------
    registry = ModelRegistry(
        model_config=model_cfg,
        experiment_config=exp_cfg,
        random_seed=random_seed,
    )

    alarm_threshold = exp_cfg["data"]["alarm_threshold_pa"]
    # dP_ideal varies by scenario; use the mean from training data
    dP_ideal_col = exp_cfg["data"].get("dP_ideal_column", "dP_ideal")
    if dP_ideal_col in df.columns:
        dP_ideal_value = float(df.loc[df["run_id"].isin(train_runs), dP_ideal_col].mean())
    else:
        dP_ideal_value = 618.4  # EDA-computed fallback
    logger.info("Using dP_ideal = %.2f Pa (computed from training set)", dP_ideal_value)

    metrics_calc = MetricsCalculator(
        alarm_threshold_pa=alarm_threshold,
        dP_ideal=dP_ideal_value,
    )

    # ----------------------------------------------------------------
    # 6. Determine models to train
    # ----------------------------------------------------------------
    active_models = args.models or registry.get_active_models()
    logger.info("Models to train: %s", active_models)

    # ----------------------------------------------------------------
    # 7. Training loop
    # ----------------------------------------------------------------
    all_results: List[Dict] = []
    for model_name in active_models:
        try:
            result = train_one_model(
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train_raw=y_train_raw.values,
                y_test_raw=y_test_raw.values,
                target_transformer=target_transformer,
                registry=registry,
                metrics_calc=metrics_calc,
                configs=configs,
                mlflow_module=mlflow_module,
                feature_names=feature_names,
                time_h_test=time_h_test,
            )
            all_results.append(result)
        except Exception as exc:
            logger.error(
                "Training failed for model '%s': %s", model_name, exc,
                exc_info=True,
            )
            all_results.append({"model_name": model_name, "error": str(exc)})

    # ----------------------------------------------------------------
    # 8. Summary leaderboard
    # ----------------------------------------------------------------
    _print_leaderboard(all_results)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_path(relative_path: str, repo_root: Path) -> Path:
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return repo_root / p


def _log_figures_to_mlflow(mlflow_module, figures: Dict[str, plt.Figure]) -> None:
    """Save figures to temp files and log as MLflow artifacts."""
    for name, fig in figures.items():
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix=f"{name}_"
            ) as tmp:
                fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
                mlflow_module.log_artifact(tmp.name, artifact_path="figures")
        except Exception as exc:
            logger.warning("Could not log figure '%s' to MLflow: %s", name, exc)


def _print_leaderboard(results: List[Dict]) -> None:
    """Print a concise leaderboard summarising all trained models."""
    rows = []
    for r in results:
        rows.append({
            "model": r.get("model_name", "?"),
            "test_r2": f"{r.get('test_r2', float('nan')):.4f}",
            "test_mae_pa": f"{r.get('test_mae_pa', float('nan')):.2f}",
            "test_rmse_pa": f"{r.get('test_rmse_pa', float('nan')):.2f}",
            "remaining_h": r.get("forecast", {}).get("remaining_runtime_h", "—"),
            "error": r.get("error", ""),
        })
    leaderboard = pd.DataFrame(rows)
    sep = "=" * 90
    logger.info("\n%s\nLEADERBOARD\n%s\n%s\n%s", sep, sep, leaderboard.to_string(index=False), sep)


class _null_context:
    """No-op context manager used when MLflow is unavailable."""
    def __enter__(self):
        class _DummyRun:
            class info:
                run_id = None
        return _DummyRun()

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LNG heat exchanger dP_error forecast model (hybrid PE methodology)."
    )
    parser.add_argument(
        "--config-dir",
        default=str(_REPO_ROOT / "training" / "config"),
        help="Directory containing experiment_config.yaml, features_config.yaml, model_config.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to train (overrides active_models in model_config.yaml). "
             "E.g.: --models lasso ridge random_forest",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow tracking (local-only run).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
