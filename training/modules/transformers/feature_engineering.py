"""
feature_engineering.py
=======================
Additional scikit-learn compatible transformers for domain-specific
feature engineering on LNG heat exchanger data.

ElapsedTimeNormalizerTransformer  — normalises elapsed time within each group.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class ElapsedTimeNormalizerTransformer(BaseEstimator, TransformerMixin):
    """
    Normalise elapsed time `t_h` relative to each group's start.

    In production streaming, each new window starts with t_h > 0 because
    it's the total elapsed time since the last defrost.  Normalising to
    [0, 1] within each training group lets the model learn a
    stage-of-campaign signal that generalises across runs of different
    total lengths.

    The normalisation is:

        t_norm = (t_h - t_min) / (t_max - t_min + eps)

    where t_min and t_max are computed *per group* during fit.

    Parameters
    ----------
    time_col : str
        Column name for elapsed time (e.g. ``"t_h"``).
    group_col : str, optional
        Column identifying independent groups.  If None, normalise globally.
    eps : float
        Small constant to prevent division by zero for single-sample groups.
    add_normalised_col : bool
        If True, add a new ``t_norm`` column *and* keep the original.
        If False, replace ``time_col`` with the normalised version.

    Attributes
    ----------
    group_stats_ : dict
        ``{group_key: (t_min, t_max)}`` fitted during training.
    global_stats_ : tuple
        ``(t_min, t_max)`` for the whole training set (fallback for unseen groups).
    """

    def __init__(
        self,
        time_col: str = "t_h",
        group_col: Optional[str] = None,
        eps: float = 1e-8,
        add_normalised_col: bool = True,
    ) -> None:
        self.time_col = time_col
        self.group_col = group_col
        self.eps = eps
        self.add_normalised_col = add_normalised_col

    def fit(self, X: pd.DataFrame, y=None) -> "ElapsedTimeNormalizerTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ElapsedTimeNormalizerTransformer expects a DataFrame, got {type(X)}"
            )
        if self.time_col not in X.columns:
            raise ValueError(
                f"ElapsedTimeNormalizerTransformer: '{self.time_col}' not in X"
            )

        self.feature_names_in_ = list(X.columns)

        # Global stats as fallback
        self.global_stats_ = (
            float(X[self.time_col].min()),
            float(X[self.time_col].max()),
        )

        # Per-group stats
        self.group_stats_: dict = {}
        if self.group_col and self.group_col in X.columns:
            for grp, sub in X.groupby(self.group_col):
                self.group_stats_[grp] = (
                    float(sub[self.time_col].min()),
                    float(sub[self.time_col].max()),
                )

        logger.info(
            "ElapsedTimeNormalizerTransformer fitted: %d groups, global range [%.2f, %.2f] h",
            len(self.group_stats_),
            self.global_stats_[0],
            self.global_stats_[1],
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "global_stats_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ElapsedTimeNormalizerTransformer expects a DataFrame, got {type(X)}"
            )
        if self.time_col not in X.columns:
            raise ValueError(
                f"ElapsedTimeNormalizerTransformer: '{self.time_col}' not in X"
            )
        X_out = X.copy()
        output_col = "t_norm" if self.add_normalised_col else self.time_col

        if self.group_col and self.group_col in X_out.columns:
            for grp, sub_idx in X_out.groupby(self.group_col).groups.items():
                t_vals = X_out.loc[sub_idx, self.time_col]
                if grp in self.group_stats_:
                    t_min, t_max = self.group_stats_[grp]
                else:
                    # Unseen group at inference time — use global stats
                    logger.warning(
                        "ElapsedTimeNormalizerTransformer: unseen group '%s', "
                        "using global stats.", grp
                    )
                    t_min, t_max = self.global_stats_
                X_out.loc[sub_idx, output_col] = (
                    (t_vals - t_min) / (t_max - t_min + self.eps)
                )
        else:
            t_min, t_max = self.global_stats_
            X_out[output_col] = (
                (X_out[self.time_col] - t_min) / (t_max - t_min + self.eps)
            )

        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        base = list(self.feature_names_in_)
        if self.add_normalised_col:
            return base + ["t_norm"]
        return base
