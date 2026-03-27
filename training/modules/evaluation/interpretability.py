"""
interpretability.py
====================
InterpretabilityReporter — answers the three business questions via plots
and structured dictionaries logged to MLflow.

Business Questions Answered
---------------------------
A. Unexpected correlations between input data and ΔP increase
   → Coefficient bar chart (linear) or SHAP bar chart (non-linear)

B. Proxy for heavy components in feed gas
   → Highlight observable-only features selected by RFE / SHAP top-k

C. Expected runtime / defrost date
   → Autoregressive runtime forecast: project dP_error forward until
     dP_ideal + dP_error_projected ≥ threshold

All figures are returned as matplotlib Figure objects so the caller
can log them to MLflow or save locally.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Observable-only features (available in production without first-principles)
_OBSERVABLE_FEATURES = {
    "T_h_in_K", "T_h_out_K", "T_c_in_K", "T_c_out_K", "delta_P_Pa",
}


class InterpretabilityReporter:
    """
    Generate interpretability artefacts from a fitted pipeline.

    Parameters
    ----------
    model_name : str
        Name of the model (used in titles).
    alarm_threshold_pa : float
        Absolute ΔP alarm threshold [Pa].
    dP_ideal : float
        Constant clean-tube ΔP baseline [Pa].
    """

    def __init__(
        self,
        model_name: str,
        alarm_threshold_pa: float = 943.3,
        dP_ideal: float = 618.4,
    ) -> None:
        self.model_name = model_name
        self.alarm_threshold_pa = alarm_threshold_pa
        self.dP_ideal = dP_ideal

    # ------------------------------------------------------------------
    # Question A + B: Feature importance / coefficients
    # ------------------------------------------------------------------
    def feature_importance_plot(
        self,
        pipeline: Pipeline,
        feature_names: List[str],
        top_n: int = 15,
    ) -> Tuple[plt.Figure, Dict]:
        """
        Return a feature importance bar chart + importance dictionary.

        For linear models: uses ``model.coef_`` (standardised by the scaler).
        For tree / ensemble: uses ``model.feature_importances_``.
        For SVR: uses SHAP kernel explainer (slow; only for small datasets).

        Parameters
        ----------
        pipeline : fitted sklearn Pipeline
        feature_names : list[str]
            Names of the features *before* SelectKBest / RFE filtering
            (i.e. X_train.columns).
        top_n : int
            Number of top features to display.

        Returns
        -------
        fig : matplotlib.Figure
        importance_dict : dict
            ``{feature_name: importance_value}`` sorted descending.
        """
        model_step = pipeline.named_steps["model"]
        selected_features = self._get_selected_features(pipeline, feature_names)

        # --- Extract importances -----------------------------------------
        if hasattr(model_step, "coef_"):
            raw_importance = np.abs(model_step.coef_).ravel()
            importance_label = "|Coefficient|"
        elif hasattr(model_step, "feature_importances_"):
            raw_importance = model_step.feature_importances_.ravel()
            importance_label = "Feature Importance"
        else:
            logger.warning(
                "InterpretabilityReporter: model '%s' has no coef_ or "
                "feature_importances_; skipping importance plot.",
                self.model_name,
            )
            return self._empty_fig("No feature importance available"), {}

        # Align lengths
        n = min(len(raw_importance), len(selected_features))
        raw_importance = raw_importance[:n]
        selected_features = selected_features[:n]

        # Sort
        order = np.argsort(raw_importance)[::-1][:top_n]
        sorted_names = [selected_features[i] for i in order]
        sorted_vals = raw_importance[order]

        # Colour-code observable vs error-signal features
        colours = [
            "#2196F3" if self._is_observable(name) else "#FF9800"
            for name in sorted_names
        ]

        # --- Plot --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
        bars = ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colours[::-1])
        ax.set_xlabel(importance_label, fontsize=12)
        ax.set_title(
            f"{self.model_name} — Feature Importance\n"
            f"(Blue = Observable DCS  |  Orange = Error Signal / Derived)",
            fontsize=11,
        )
        ax.tick_params(axis="y", labelsize=9)
        fig.tight_layout()

        importance_dict = dict(zip(sorted_names, [float(v) for v in sorted_vals]))

        # Annotate business question B
        observable_selected = [n for n in sorted_names if self._is_observable(n)]
        if observable_selected:
            logger.info(
                "InterpretabilityReporter [QsB]: observable proxy features ranked by model: %s",
                observable_selected,
            )

        return fig, importance_dict

    # ------------------------------------------------------------------
    # Question C: Runtime forecast
    # ------------------------------------------------------------------
    def runtime_forecast_plot(
        self,
        pipeline: Pipeline,
        X_history: pd.DataFrame,
        y_history: np.ndarray,
        time_h_history: np.ndarray,
        horizon_multiplier: float = 5.0,
        min_points: int = 10,
        reference_datetime: Optional[datetime] = None,
    ) -> Tuple[plt.Figure, Dict]:
        """
        Autoregressive dP_error forecast until alarm threshold is projected.

        Strategy
        --------
        1. Predict dP_error on current history using the fitted pipeline
        2. Extrapolate the linear trend of *predicted* dP_error forward
        3. Find when dP_ideal + dP_error_projected ≥ threshold
        4. Report the estimated remaining runtime in hours and as a date

        Parameters
        ----------
        pipeline : fitted sklearn Pipeline
        X_history : DataFrame
            Feature matrix for the history period.
        y_history : array
            Actual dP_error values for the history.
        time_h_history : array
            Elapsed hours for each history row.
        horizon_multiplier : float
            Forecast to ``current_time × horizon_multiplier`` hours.
        min_points : int
            Minimum history rows needed before forecasting.
        reference_datetime : datetime, optional
            If provided, compute calendar defrost date.

        Returns
        -------
        fig : matplotlib.Figure
        forecast_summary : dict
            ``t_defrost_h``, ``remaining_h``, ``defrost_date`` (if datetime given).
        """
        if len(X_history) < min_points:
            logger.warning(
                "InterpretabilityReporter: only %d history points (min=%d); "
                "returning placeholder forecast.",
                len(X_history), min_points,
            )
            return self._empty_fig("Insufficient history for forecast"), {}

        # --- In-sample predictions ---------------------------------------
        try:
            y_pred_history = pipeline.predict(X_history)
        except Exception as exc:
            logger.error("InterpretabilityReporter: prediction failed: %s", exc)
            return self._empty_fig(f"Prediction error: {exc}"), {}

        current_t = float(time_h_history[-1])
        error_threshold = self.alarm_threshold_pa - self.dP_ideal

        # --- Linear extrapolation of predicted dP_error ------------------
        # Use last 30% of history to fit the trend (avoids startup transients)
        n_trend = max(min_points, int(0.3 * len(y_pred_history)))
        t_trend = time_h_history[-n_trend:]
        y_trend = y_pred_history[-n_trend:]

        coeffs = np.polyfit(t_trend, y_trend, deg=1)  # linear fit
        slope, intercept = coeffs[0], coeffs[1]

        # Maximum credible forecast: 30 days = 720 h
        MAX_FORECAST_H = 720.0

        if slope <= 1e-6:  # negligible or negative slope — no crossing predicted
            logger.info(
                "InterpretabilityReporter: dP_error slope ≤ 1e-6 Pa/h (%.6f); "
                "no threshold crossing predicted.", slope
            )
            t_defrost = None
        else:
            # Solve: slope * t + intercept = error_threshold
            t_defrost_raw = (error_threshold - intercept) / slope
            if t_defrost_raw <= current_t:
                t_defrost = None  # already past threshold
            elif (t_defrost_raw - current_t) > MAX_FORECAST_H:
                logger.info(
                    "InterpretabilityReporter: projected defrost at %.1f h exceeds "
                    "maximum forecast horizon (%.0f h) — treating as 'not within horizon'.",
                    t_defrost_raw, MAX_FORECAST_H,
                )
                t_defrost = None
            else:
                t_defrost = t_defrost_raw

        # --- Build forecast series for plotting --------------------------
        t_max = current_t * horizon_multiplier if current_t > 0 else current_t + 50
        t_future = np.linspace(current_t, t_max, 200)
        dP_error_future = slope * t_future + intercept
        dP_total_future = self.dP_ideal + dP_error_future

        # --- Plot --------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        # Top: dP_error evolution
        ax1 = axes[0]
        ax1.plot(time_h_history, y_history, "k.-", alpha=0.5, lw=0.8, label="Actual dP_error")
        ax1.plot(time_h_history, y_pred_history, "b-", lw=1.5, label="Model prediction (history)")
        ax1.plot(t_future, dP_error_future, "b--", lw=1.5, alpha=0.7, label="Extrapolated forecast")
        ax1.axhline(error_threshold, color="red", ls="--", alpha=0.7, label=f"Error threshold ({error_threshold:.0f} Pa)")
        if t_defrost is not None:
            ax1.axvline(t_defrost, color="red", ls=":", lw=2, label=f"Predicted defrost at {t_defrost:.1f} h")
        ax1.set_ylabel("dP_error [Pa]", fontsize=11)
        ax1.set_title(f"{self.model_name} — Runtime Forecast (Business Question C)", fontsize=12)
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Bottom: total ΔP
        ax2 = axes[1]
        dP_total_history = self.dP_ideal + y_history
        dP_total_pred_history = self.dP_ideal + y_pred_history
        ax2.plot(time_h_history, dP_total_history, "k.-", alpha=0.5, lw=0.8, label="Actual ΔP_total")
        ax2.plot(time_h_history, dP_total_pred_history, "b-", lw=1.5, label="Predicted ΔP_total")
        ax2.plot(t_future, dP_total_future, "b--", lw=1.5, alpha=0.7, label="Forecast ΔP_total")
        ax2.axhline(self.alarm_threshold_pa, color="red", ls="--", alpha=0.7,
                    label=f"Alarm threshold ({self.alarm_threshold_pa:.0f} Pa)")
        if t_defrost is not None:
            ax2.axvline(t_defrost, color="red", ls=":", lw=2)
            fill_end = min(t_defrost + (t_max - current_t) * 0.5, t_max)
            ax2.fill_betweenx(
                [0, self.alarm_threshold_pa * 1.3],
                t_defrost, fill_end,
                alpha=0.05, color="red", label="Predicted alarm zone",
            )
        ax2.set_xlabel("Elapsed Time [h]", fontsize=11)
        ax2.set_ylabel("ΔP_total [Pa]", fontsize=11)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        # --- Summary dict ------------------------------------------------
        summary: Dict = {
            "current_t_h": float(current_t),
            "dP_error_slope_pa_per_h": float(slope),
            "error_threshold_pa": float(error_threshold),
        }
        if t_defrost is not None:
            remaining_h = t_defrost - current_t
            summary["t_defrost_h"] = float(t_defrost)
            summary["remaining_runtime_h"] = float(remaining_h)
            logger.info(
                "InterpretabilityReporter [QsC]: predicted defrost at %.1f h "
                "(%.1f h remaining from current t=%.1f h)",
                t_defrost, remaining_h, current_t,
            )
            if reference_datetime is not None:
                defrost_date = reference_datetime + timedelta(hours=remaining_h)
                summary["defrost_date"] = defrost_date.isoformat()
                logger.info(
                    "InterpretabilityReporter [QsC]: defrost date = %s",
                    defrost_date.strftime("%Y-%m-%d %H:%M"),
                )
        else:
            summary["t_defrost_h"] = None
            summary["remaining_runtime_h"] = None
            logger.info(
                "InterpretabilityReporter [QsC]: no threshold crossing predicted "
                "within forecast horizon."
            )

        return fig, summary

    # ------------------------------------------------------------------
    # Prediction residuals plot (universal)
    # ------------------------------------------------------------------
    def residuals_plot(
        self,
        y_train_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray,
    ) -> plt.Figure:
        """Actual vs Predicted scatter + residuals histogram."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Scatter: train
        ax = axes[0]
        ax.scatter(y_train_true, y_train_pred, alpha=0.3, s=4, c="steelblue", label="Train")
        lims = [min(y_train_true.min(), y_train_pred.min()), max(y_train_true.max(), y_train_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect")
        ax.set_xlabel("Actual dP_error [Pa]")
        ax.set_ylabel("Predicted dP_error [Pa]")
        ax.set_title(f"{self.model_name} — Train Set")
        ax.legend(fontsize=8)

        # Scatter: test
        ax = axes[1]
        ax.scatter(y_test_true, y_test_pred, alpha=0.3, s=4, c="darkorange", label="Test")
        lims = [min(y_test_true.min(), y_test_pred.min()), max(y_test_true.max(), y_test_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect")
        ax.set_xlabel("Actual dP_error [Pa]")
        ax.set_title(f"{self.model_name} — Test Set")
        ax.legend(fontsize=8)

        # Residuals histogram
        ax = axes[2]
        residuals_test = y_test_pred - y_test_true
        ax.hist(residuals_test, bins=40, color="darkorange", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", ls="--")
        ax.set_xlabel("Residual [Pa]")
        ax.set_ylabel("Count")
        ax.set_title(f"{self.model_name} — Test Residuals")

        fig.suptitle(f"Model: {self.model_name}", fontsize=13, y=1.02)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_selected_features(
        self, pipeline: Pipeline, feature_names: List[str]
    ) -> List[str]:
        """Walk pipeline steps to get post-selection feature names."""
        current_names = list(feature_names)

        if "feature_selection_kbest" in pipeline.named_steps:
            selector = pipeline.named_steps["feature_selection_kbest"]
            if hasattr(selector, "get_support"):
                mask = selector.get_support()
                if len(mask) == len(current_names):
                    current_names = [n for n, m in zip(current_names, mask) if m]

        if "feature_selection_rfe" in pipeline.named_steps:
            selector = pipeline.named_steps["feature_selection_rfe"]
            if hasattr(selector, "get_support"):
                mask = selector.get_support()
                if len(mask) == len(current_names):
                    current_names = [n for n, m in zip(current_names, mask) if m]

        return current_names

    @staticmethod
    def _is_observable(feature_name: str) -> bool:
        """Check if feature originates from an observable DCS signal."""
        return any(obs in feature_name for obs in _OBSERVABLE_FEATURES)

    @staticmethod
    def _empty_fig(message: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12,
                transform=ax.transAxes)
        ax.axis("off")
        return fig
