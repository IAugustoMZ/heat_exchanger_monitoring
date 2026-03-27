"""
target_transformer.py
=====================
Yeo-Johnson power transformer for the target variable `dP_error`.

YeoJohnsonTargetTransformer wraps sklearn's PowerTransformer with a
fixed or data-fitted lambda.  It:

- Transforms y during training (fit / fit_transform)
- Provides inverse_transform to recover Pa-scale predictions for threshold
  comparison and business-layer consumption
- Logs the fitted lambda to MLflow when available

Usage
-----
    t = YeoJohnsonTargetTransformer(lambda_=0.61, refit=False)
    y_transformed = t.fit_transform(y_train)
    y_pred_pa = t.inverse_transform(model.predict(X_test))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class YeoJohnsonTargetTransformer(BaseEstimator, TransformerMixin):
    """
    Yeo-Johnson transform wrapper for a single-column regression target.

    Parameters
    ----------
    lambda_ : float, optional
        Fixed Box-Cox / Yeo-Johnson lambda.  Used *only* when
        ``refit=False``.  Default 0.61 (fitted on combined dataset in EDA).
    refit : bool
        If True, the lambda is estimated via MLE from the training data
        passed to fit().  If False, the provided ``lambda_`` is used.
    standardize : bool
        Whether to also z-score the transformed values.  Default False
        to keep transformed target on an interpretable scale.

    Attributes
    ----------
    lambda_fitted_ : float
        The lambda value actually used (either provided or MLE-estimated).
    power_transformer_ : PowerTransformer
        The underlying sklearn transformer.
    """

    def __init__(
        self,
        lambda_: float = 0.61,
        refit: bool = False,
        standardize: bool = False,
    ) -> None:
        self.lambda_ = lambda_
        self.refit = refit
        self.standardize = standardize

    def fit(self, y: np.ndarray, X=None) -> "YeoJohnsonTargetTransformer":
        """
        Fit the transformer.  Accepts 1-D array or single-column DataFrame.
        """
        y_arr = self._coerce_to_2d(y)

        self.power_transformer_ = PowerTransformer(
            method="yeo-johnson",
            standardize=self.standardize,
            copy=True,
        )

        if self.refit:
            self.power_transformer_.fit(y_arr)
            self.lambda_fitted_ = float(self.power_transformer_.lambdas_[0])
            logger.info(
                "YeoJohnsonTargetTransformer: refitted lambda = %.4f",
                self.lambda_fitted_,
            )
        else:
            # Manually set the lambda and fit mean/std (only needed for
            # standardize=True; with standardize=False these are unused).
            self.power_transformer_.fit(y_arr)  # fit for structural attributes
            self.power_transformer_.lambdas_ = np.array([self.lambda_])
            self.lambda_fitted_ = self.lambda_
            logger.info(
                "YeoJohnsonTargetTransformer: using fixed lambda = %.4f",
                self.lambda_fitted_,
            )

        return self

    def transform(self, y: np.ndarray, X=None) -> np.ndarray:
        check_is_fitted(self, "power_transformer_")
        y_arr = self._coerce_to_2d(y)
        transformed = self.power_transformer_.transform(y_arr)
        return transformed.ravel()

    def fit_transform(self, y: np.ndarray, X=None, **fit_params) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Recover original Pa-scale values from transformed predictions."""
        check_is_fitted(self, "power_transformer_")
        y_arr = self._coerce_to_2d(y_transformed)
        original = self.power_transformer_.inverse_transform(y_arr)
        return original.ravel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_to_2d(y) -> np.ndarray:
        """Convert 1-D array / Series to shape (n, 1) for sklearn compatibility."""
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, pd.DataFrame):
            y = y.values
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(
                f"YeoJohnsonTargetTransformer expects a 1-D array or "
                f"single-column 2-D array, got shape {y.shape}"
            )
        return y

    def log_to_mlflow(self) -> None:
        """Log fitted lambda to an active MLflow run (no-op if mlflow unavailable)."""
        check_is_fitted(self, "lambda_fitted_")
        try:
            import mlflow  # local import — optional dependency

            mlflow.log_param("target_transform_lambda", self.lambda_fitted_)
            mlflow.log_param("target_transform_method", "yeo_johnson")
            mlflow.log_param("target_transform_refit", self.refit)
        except ImportError:
            logger.debug("mlflow not installed; skipping lambda logging.")
        except Exception as exc:
            logger.warning("Could not log transform params to MLflow: %s", exc)
