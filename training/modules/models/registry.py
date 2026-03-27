"""
registry.py
===========
ModelRegistry — builds sklearn Pipelines from config for each algorithm.

Responsibilities
----------------
- Parse model_config.yaml to instantiate the correct estimator
- Wrap estimator in a standard Pipeline:
    ColumnTransformer (StandardScaler) → SelectKBest → RFE → Estimator
- Provide an Optuna objective factory that suggests hyperparameters from
  the config, trains, and returns the CV score
- Log each Optuna trial as a child MLflow run

Design notes
------------
- RFE is only applied to linear models (interpretability path).
  Non-linear models use SelectKBest only (RFE is slow on RF/GBM).
- The Pipeline exposes named steps so coefficients/importances can be
  extracted by InterpretabilityReporter after fitting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

# Map config estimator names → sklearn classes
_ESTIMATOR_MAP: Dict[str, Any] = {
    "Lasso": Lasso,
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
}

_LINEAR_ESTIMATORS = {"Lasso", "Ridge", "ElasticNet", "LinearRegression"}

# RFE needs a base estimator that supports coef_
_RFE_BASE = Lasso


class ModelRegistry:
    """
    Builds and manages sklearn Pipelines from YAML config.

    Parameters
    ----------
    model_config : dict
        Loaded model_config.yaml.
    experiment_config : dict
        Loaded experiment_config.yaml.
    random_seed : int
        Global random seed.
    """

    def __init__(
        self,
        model_config: Dict,
        experiment_config: Dict,
        random_seed: int = 42,
    ) -> None:
        self.model_config = model_config
        self.experiment_config = experiment_config
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_pipeline(
        self,
        model_name: str,
        k_best: int = 8,
        rfe_n_features: int = 5,
        estimator_params: Optional[Dict] = None,
    ) -> Pipeline:
        """
        Construct the full sklearn Pipeline for a named model.

        Parameters
        ----------
        model_name : str
            Key from model_config.yaml (e.g. "lasso", "random_forest").
        k_best : int
            Number of features to keep with SelectKBest.
        rfe_n_features : int
            Number of features after RFE (linear models only).
        estimator_params : dict, optional
            Hyperparameters to pass directly to the estimator constructor.

        Returns
        -------
        Pipeline
        """
        cfg = self._get_model_cfg(model_name)
        estimator_cls = _ESTIMATOR_MAP.get(cfg["estimator"])
        if estimator_cls is None:
            raise ValueError(
                f"ModelRegistry: unknown estimator '{cfg['estimator']}'. "
                f"Available: {list(_ESTIMATOR_MAP.keys())}"
            )

        params = estimator_params or {}
        is_linear = cfg["estimator"] in _LINEAR_ESTIMATORS

        # Add random state where supported (all linear + RF/GBM)
        if cfg["estimator"] not in {"LinearRegression", "SVR"}:
            params.setdefault("random_state", self.random_seed)

        estimator = estimator_cls(**params)

        steps: List[Tuple[str, Any]] = [
            ("scaler", StandardScaler()),
            ("feature_selection_kbest", SelectKBest(f_regression, k=k_best)),
        ]

        if is_linear and cfg["estimator"] != "LinearRegression":
            # RFE wraps a fast Lasso for feature ranking
            steps.append((
                "feature_selection_rfe",
                RFE(
                    estimator=_RFE_BASE(random_state=self.random_seed),
                    n_features_to_select=rfe_n_features,
                )
            ))

        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        logger.debug(
            "ModelRegistry: built pipeline for '%s' with %d steps",
            model_name,
            len(steps),
        )
        return pipeline

    def make_optuna_objective(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 5,
        scoring: str = "neg_mean_absolute_error",
        mlflow_run_id: Optional[str] = None,
    ):
        """
        Return a callable Optuna objective for the given model.

        Each call to the objective:
        1. Samples hyperparameters from the config search space
        2. Builds a pipeline
        3. Fits with TimeSeriesSplit CV
        4. Logs the trial as a child MLflow run

        Parameters
        ----------
        model_name : str
        X_train : pd.DataFrame
        y_train : pd.Series
        n_splits : int
            Inner CV folds.
        scoring : str
            sklearn scoring string.
        mlflow_run_id : str, optional
            Parent run ID to nest child runs under.

        Returns
        -------
        Callable  (trial → float)
        """
        cfg = self._get_model_cfg(model_name)
        feature_sel_cfg = self.model_config.get("feature_selection", {})
        ts_cv = TimeSeriesSplit(n_splits=n_splits)

        def objective(trial) -> float:
            import mlflow  # optional dependency

            # -- Sample feature selection hyperparameters ------------------
            k_cfg = feature_sel_cfg.get("univariate_k", {"low": 3, "high": 10})
            k_best = trial.suggest_int(
                "k_best",
                k_cfg.get("low", 3),
                min(k_cfg.get("high", 10), X_train.shape[1]),
            )
            is_linear = cfg["estimator"] in _LINEAR_ESTIMATORS
            rfe_n_features = k_best  # default for non-linear
            if is_linear and cfg["estimator"] != "LinearRegression":
                rfe_cfg = feature_sel_cfg.get("rfe_n_features", {"low": 2, "high": 5})
                rfe_n_features = trial.suggest_int(
                    "rfe_n_features",
                    rfe_cfg.get("low", 2),
                    min(rfe_cfg.get("high", 5), k_best),
                )

            # -- Sample estimator hyperparameters --------------------------
            estimator_params = self._suggest_params(trial, cfg.get("params", {}))

            # -- Build pipeline & CV score ---------------------------------
            pipeline = self.build_pipeline(
                model_name=model_name,
                k_best=k_best,
                rfe_n_features=rfe_n_features,
                estimator_params=estimator_params,
            )

            try:
                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=ts_cv,
                    scoring=scoring,
                    n_jobs=1,    # keep deterministic; parallelize at Optuna study level
                )
                cv_score = float(np.mean(scores))
            except Exception as exc:
                logger.warning(
                    "ModelRegistry: trial %d failed with error: %s — pruning.",
                    trial.number, exc
                )
                raise optuna.exceptions.TrialPruned() from exc

            # -- Log to MLflow as child run --------------------------------
            log_all = (
                self.experiment_config.get("optuna", {}).get("log_all_trials", True)
            )
            if log_all and mlflow_run_id:
                try:
                    with mlflow.start_run(
                        run_name=f"{model_name}_trial_{trial.number:04d}",
                        nested=True,
                    ):
                        mlflow.log_params({
                            "model_name": model_name,
                            "trial_number": trial.number,
                            "k_best": k_best,
                            "rfe_n_features": rfe_n_features,
                            **{f"param_{k}": v for k, v in estimator_params.items()},
                        })
                        mlflow.log_metric("cv_neg_mae", cv_score)
                except Exception as log_exc:
                    logger.debug("MLflow child run logging failed: %s", log_exc)

            return cv_score  # Optuna maximises neg_MAE (direction=maximize → closest to 0 = lowest MAE)

        import optuna  # noqa — ensure available at objective-call time
        return objective

    def get_active_models(self) -> List[str]:
        """Return list of model names enabled in config."""
        return self.model_config.get("active_models", list(_ESTIMATOR_MAP.keys()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_model_cfg(self, model_name: str) -> Dict:
        cfg = self.model_config.get(model_name)
        if cfg is None:
            raise ValueError(
                f"ModelRegistry: model '{model_name}' not found in model_config.yaml. "
                f"Available: {[k for k in self.model_config if isinstance(self.model_config[k], dict)]}"
            )
        return cfg

    @staticmethod
    def _suggest_params(trial, param_space: Dict) -> Dict:
        """Translate YAML param spec into an Optuna suggest call."""
        import optuna  # noqa

        params: Dict[str, Any] = {}
        for param_name, spec in param_space.items():
            suggest_type = spec.get("suggest")
            if suggest_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif suggest_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, spec["low"], spec["high"]
                )
            elif suggest_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, spec["choices"]
                )
            # spec with no suggest key → fixed value (skip)
        return params
