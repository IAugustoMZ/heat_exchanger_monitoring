"""
metrics.py
==========
MetricsCalculator — centralised computation and reporting of model metrics.

Computes R², MAE, RMSE on train/test sets and the physically meaningful
"time-to-threshold error" (difference between predicted and actual
critical-condition time).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Compute, store, and log model performance metrics.

    Parameters
    ----------
    alarm_threshold_pa : float
        ΔP alarm threshold in Pa (e.g. 943.3 Pa — 150% of baseline).
    dP_ideal : float
        Constant ideal ΔP baseline; used to reconstruct absolute ΔP from error.
    """

    def __init__(
        self,
        alarm_threshold_pa: float = 943.3,
        dP_ideal: float = 618.4,
    ) -> None:
        self.alarm_threshold_pa = alarm_threshold_pa
        self.dP_ideal = dP_ideal

    # ------------------------------------------------------------------
    # Core regression metrics
    # ------------------------------------------------------------------
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split: str = "test",
    ) -> Dict[str, float]:
        """
        Compute R², MAE, RMSE for one split.

        Parameters
        ----------
        y_true : array-like
            Observed values (original Pa scale, NOT transformed).
        y_pred : array-like
            Predicted values (original Pa scale).
        split : str
            Label for the metric keys (``"train"`` or ``"test"``).

        Returns
        -------
        dict  with keys ``{split}_r2``, ``{split}_mae``, ``{split}_rmse``.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) == 0:
            logger.warning("MetricsCalculator.compute: empty arrays for split '%s'.", split)
            return {}

        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        metrics = {
            f"{split}_r2": r2,
            f"{split}_mae_pa": mae,
            f"{split}_rmse_pa": rmse,
        }
        logger.info(
            "MetricsCalculator [%s]: R²=%.4f  MAE=%.2f Pa  RMSE=%.2f Pa",
            split, r2, mae, rmse,
        )
        return metrics

    def compute_all(
        self,
        y_train_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics for both train and test splits and merge."""
        train_metrics = self.compute(y_train_true, y_train_pred, split="train")
        test_metrics = self.compute(y_test_true, y_test_pred, split="test")
        all_metrics = {**train_metrics, **test_metrics}

        # Generalisation gap
        if "train_r2" in all_metrics and "test_r2" in all_metrics:
            all_metrics["r2_generalisation_gap"] = float(
                all_metrics["train_r2"] - all_metrics["test_r2"]
            )

        return all_metrics

    # ------------------------------------------------------------------
    # Business KPI: time-to-threshold error
    # ------------------------------------------------------------------
    def time_to_threshold_error(
        self,
        time_h: np.ndarray,
        dP_error_true: np.ndarray,
        dP_error_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute the error in predicted vs actual time-to-critical-condition.

        The "critical condition" is when ΔP_total = dP_ideal + dP_error ≥ threshold.

        Parameters
        ----------
        time_h : array-like
            Elapsed time in hours (aligned with y arrays).
        dP_error_true : array-like
            Actual dP_error values.
        dP_error_pred : array-like
            Predicted dP_error values.

        Returns
        -------
        dict
            ``t_threshold_true_h``, ``t_threshold_pred_h``,
            ``threshold_error_h`` (pred − true; positive = pessimistic).
        """
        time_h = np.asarray(time_h, dtype=float)
        dP_error_true = np.asarray(dP_error_true, dtype=float)
        dP_error_pred = np.asarray(dP_error_pred, dtype=float)

        error_threshold = self.alarm_threshold_pa - self.dP_ideal

        t_true = self._first_crossing(time_h, dP_error_true, error_threshold)
        t_pred = self._first_crossing(time_h, dP_error_pred, error_threshold)

        result: Dict[str, float] = {}
        if t_true is not None:
            result["t_threshold_true_h"] = t_true
        if t_pred is not None:
            result["t_threshold_pred_h"] = t_pred
        if t_true is not None and t_pred is not None:
            result["threshold_error_h"] = t_pred - t_true
            logger.info(
                "MetricsCalculator: threshold at %.2f h (actual) vs %.2f h (predicted) → error = %.2f h",
                t_true, t_pred, result["threshold_error_h"],
            )
        elif t_true is None and t_pred is not None:
            logger.warning(
                "MetricsCalculator: model predicts threshold crossing at %.2f h "
                "but actual series did not cross threshold — false positive.",
                t_pred,
            )
            result["false_positive_threshold"] = 1.0
        elif t_true is not None and t_pred is None:
            logger.warning(
                "MetricsCalculator: actual threshold crossing at %.2f h "
                "but model did not predict it — false negative.",
                t_true,
            )
            result["false_negative_threshold"] = 1.0
        else:
            result["no_threshold_crossing"] = 1.0

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _first_crossing(
        time_h: np.ndarray,
        values: np.ndarray,
        threshold: float,
    ) -> Optional[float]:
        """Return time of first crossing above threshold, or None."""
        mask = values >= threshold
        if not mask.any():
            return None
        idx = int(np.argmax(mask))  # first True
        return float(time_h[idx])
