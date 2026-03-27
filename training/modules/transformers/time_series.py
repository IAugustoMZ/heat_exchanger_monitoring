"""
time_series.py
==============
Scikit-learn compatible transformers for time-series feature engineering.

LaggedFeaturesTransformer  — creates t-1 … t-n lag columns per group.
RateOfChangeTransformer    — rolling finite-difference (smoothed dX/dt).

Both implement TransformerMixin + BaseEstimator so they plug into
any sklearn Pipeline or ColumnTransformer.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class LaggedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Create lagged copies of specified columns, respecting group boundaries.

    Parameters
    ----------
    columns : list[str]
        Column names to lag.
    n_lags : int
        Number of lag steps (creates t-1, t-2, …, t-n_lags).
    group_col : str, optional
        Column that identifies independent time series (e.g. "group").
        Rows are shifted *within* each group, preventing leakage across
        different scenarios/runs.  If None, the whole DataFrame is shifted.
    drop_original : bool
        If True, drop the original columns after adding lags.
        Default False (keep both).

    Notes
    -----
    - NaN rows introduced by shifting are forward-filled with the first
      valid observation within the group, so the transformer never reduces
      the number of rows.  This is intentional — during inference on a
      streaming window, there may be insufficient history.
    - The transformer is fit()-less (no learned parameters).  fit() is
      provided only for pipeline compatibility.
    """

    def __init__(
        self,
        columns: List[str],
        n_lags: int = 1,
        group_col: Optional[str] = None,
        drop_original: bool = False,
    ) -> None:
        self.columns = columns
        self.n_lags = n_lags
        self.group_col = group_col
        self.drop_original = drop_original

    # ------------------------------------------------------------------
    # fit — stateless; just validates column presence
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "LaggedFeaturesTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"LaggedFeaturesTransformer expects a DataFrame, got {type(X)}"
            )
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"LaggedFeaturesTransformer: columns not found in X: {missing}"
            )
        if self.group_col and self.group_col not in X.columns:
            raise ValueError(
                f"LaggedFeaturesTransformer: group_col '{self.group_col}' not in X"
            )
        self.feature_names_in_ = list(X.columns)
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"LaggedFeaturesTransformer expects a DataFrame, got {type(X)}"
            )
        X_out = X.copy()

        for col in self.columns:
            if col not in X_out.columns:
                logger.warning(
                    "LaggedFeaturesTransformer: column '%s' not found "
                    "during transform, skipping.", col
                )
                continue
            for lag in range(1, self.n_lags + 1):
                lag_col_name = f"{col}(t-{lag})"
                if self.group_col and self.group_col in X_out.columns:
                    X_out[lag_col_name] = (
                        X_out.groupby(self.group_col, sort=False)[col]
                        .shift(lag)
                    )
                else:
                    X_out[lag_col_name] = X_out[col].shift(lag)

                # Forward-fill NaN at start of each group
                if self.group_col and self.group_col in X_out.columns:
                    X_out[lag_col_name] = (
                        X_out.groupby(self.group_col, sort=False)[lag_col_name]
                        .transform(lambda s: s.bfill())
                    )
                else:
                    X_out[lag_col_name] = X_out[lag_col_name].bfill()

        if self.drop_original:
            X_out.drop(columns=self.columns, inplace=True, errors="ignore")

        logger.debug(
            "LaggedFeaturesTransformer: added %d lag columns (n_lags=%d).",
            len(self.columns) * self.n_lags,
            self.n_lags,
        )
        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return output feature names for Pipeline compatibility."""
        base = list(self.feature_names_in_)
        lag_names: List[str] = []
        for col in self.columns:
            for lag in range(1, self.n_lags + 1):
                lag_names.append(f"{col}(t-{lag})")
        if self.drop_original:
            base = [c for c in base if c not in self.columns]
        return base + lag_names


class RateOfChangeTransformer(BaseEstimator, TransformerMixin):
    """
    Compute smoothed rate-of-change (first discrete derivative) for columns.

    For each target column *c* computes:

        dX/dt = (X_smooth[t] - X_smooth[t-1]) / dt

    where X_smooth is a rolling mean over `window` samples and
    dt = time difference between consecutive rows (assumed uniform = 1 sample
    unless a `time_col` is provided).

    Parameters
    ----------
    columns : list[str]
        Columns for which to compute the rate of change.
    window : int
        Rolling-mean window (samples) applied before differencing.
        Reduces noise.  Must be >= 1.
    group_col : str, optional
        Column that identifies independent time series.  Differencing
        is performed within each group.
    time_col : str, optional
        Column with real-valued time (e.g., seconds or hours).  If
        provided, dX/dt is divided by dt to give physical units.
        If None, dX/dt is expressed per sample.
    fill_value : float
        Value to use for the first row of each group (no previous sample).
        Default 0.0 (safe for dP_error which starts near 0).
    """

    def __init__(
        self,
        columns: List[str],
        window: int = 5,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        fill_value: float = 0.0,
    ) -> None:
        self.columns = columns
        self.window = window
        self.group_col = group_col
        self.time_col = time_col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None) -> "RateOfChangeTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"RateOfChangeTransformer expects a DataFrame, got {type(X)}"
            )
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"RateOfChangeTransformer: columns not found in X: {missing}"
            )
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"RateOfChangeTransformer expects a DataFrame, got {type(X)}"
            )
        X_out = X.copy()

        def _roc_series(s: pd.Series, time_s: Optional[pd.Series]) -> pd.Series:
            """Compute roc for one series (already within group)."""
            smoothed = s.rolling(window=self.window, min_periods=1).mean()
            delta_x = smoothed.diff()
            if time_s is not None:
                delta_t = time_s.diff().replace(0, np.nan)
                roc = delta_x / delta_t
            else:
                roc = delta_x
            return roc.fillna(self.fill_value)

        for col in self.columns:
            if col not in X_out.columns:
                logger.warning(
                    "RateOfChangeTransformer: column '%s' missing, skipping.", col
                )
                continue
            roc_col_name = f"d_{col}_dt"
            if self.group_col and self.group_col in X_out.columns:
                # per-group transform keeping index aligned
                time_series_map = (
                    X_out.groupby(self.group_col, sort=False)[self.time_col]
                    if self.time_col and self.time_col in X_out.columns
                    else None
                )
                results = []
                for grp_key, grp_df in X_out.groupby(self.group_col, sort=False):
                    time_s = grp_df[self.time_col] if (
                        self.time_col and self.time_col in grp_df.columns
                    ) else None
                    roc = _roc_series(grp_df[col], time_s)
                    results.append(roc)
                X_out[roc_col_name] = pd.concat(results).sort_index()
            else:
                time_s = (
                    X_out[self.time_col]
                    if self.time_col and self.time_col in X_out.columns
                    else None
                )
                X_out[roc_col_name] = _roc_series(X_out[col], time_s)

        logger.debug(
            "RateOfChangeTransformer: added %d rate-of-change columns.",
            len(self.columns),
        )
        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        base = list(self.feature_names_in_)
        roc_names = [f"d_{col}_dt" for col in self.columns]
        return base + roc_names
