"""
model_service.py
================
Handles communication with the MLflow model serving endpoint and
retrieval of interpretability artifacts from the MLflow tracking server.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Proxy for the MLflow model serving endpoint (port 5001).

    Responsibilities:
      - Send feature vectors to the served model and return predictions
      - Load interpretability artifacts (feature importance, forecasts, figures)
        from the MLflow artifact store
    """

    def __init__(
        self,
        serve_uri: str,
        tracking_uri: str,
        artifacts_dir: str,
        best_run_id: str,
    ) -> None:
        self._serve_uri = serve_uri.rstrip("/")
        self._tracking_uri = tracking_uri.rstrip("/")
        self._artifacts_dir = Path(artifacts_dir)
        self._best_run_id = best_run_id
        self._invocations_url = f"{self._serve_uri}/invocations"

    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    async def predict(self, features: Dict[str, float]) -> float:
        """
        Send a single row of features to MLflow model serving and
        return the predicted dP_error (in transformed space — the caller
        should apply inverse Yeo-Johnson if needed for Pa-scale).

        The MLflow serving endpoint accepts dataframe_split format.
        """
        columns = list(features.keys())
        values = [list(features.values())]

        payload = {
            "dataframe_split": {
                "columns": columns,
                "data": values,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self._invocations_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()

                # MLflow returns {"predictions": [value]} for sklearn models
                predictions = result.get("predictions", [])
                if not predictions:
                    raise ValueError(
                        f"Empty predictions returned from model. Response: {result}"
                    )
                return float(predictions[0])

        except httpx.ConnectError as exc:
            logger.error(
                "Cannot connect to MLflow model serving at %s: %s",
                self._invocations_url, exc,
            )
            raise ConnectionError(
                f"MLflow model serving unavailable at {self._serve_uri}. "
                "Ensure the mlflow container is running and healthy."
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "MLflow model serving returned HTTP %d: %s",
                exc.response.status_code, exc.response.text,
            )
            raise ValueError(
                f"Model prediction failed (HTTP {exc.response.status_code}): "
                f"{exc.response.text[:500]}"
            ) from exc

    # ──────────────────────────────────────────────────────────
    # Interpretability artifacts
    # ──────────────────────────────────────────────────────────

    def get_interpretability_data(self) -> Dict[str, Any]:
        """
        Load all interpretability artifacts for the best model from disk.

        Returns dict with keys: feature_importance, forecast, figures
        """
        run_artifacts = self._artifacts_dir / "1" / self._best_run_id / "artifacts"

        result: Dict[str, Any] = {
            "model_name": "lasso",
            "run_id": self._best_run_id,
            "feature_importance": {},
            "forecast": {},
            "figures": [],
        }

        # Load JSON artifacts from interpretability/
        interp_dir = run_artifacts / "interpretability"
        if interp_dir.exists():
            for json_file in sorted(interp_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "importance" in json_file.name:
                        result["feature_importance"] = data
                    elif "forecast" in json_file.name:
                        result["forecast"] = data
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to load %s: %s", json_file, exc)

        # List available figure paths
        figures_dir = run_artifacts / "figures"
        if figures_dir.exists():
            for png in sorted(figures_dir.glob("*.png")):
                result["figures"].append({
                    "name": png.stem,
                    "filename": png.name,
                })

        return result

    def get_figure_path(self, filename: str) -> Optional[Path]:
        """Return the absolute path to a figure PNG in the best model's artifacts."""
        # Sanitize filename to prevent path traversal
        safe_name = Path(filename).name
        fig_path = (
            self._artifacts_dir / "1" / self._best_run_id
            / "artifacts" / "figures" / safe_name
        )
        if fig_path.exists() and fig_path.suffix == ".png":
            return fig_path
        return None

    def get_all_models_metrics(self) -> List[Dict[str, Any]]:
        """
        Load metrics for all trained models to show comparison.
        Reads from the MLflow artifact directories.
        """
        models_info = []
        experiment_dir = self._artifacts_dir / "1"

        if not experiment_dir.exists():
            return models_info

        # Map of run_id to model names (from known training runs)
        _RUN_MODELS = {
            "90a44c352c3d4fe3a45df3e63a3e01b9": "Lasso",
            "ef217c36493e49529d0f747968279c2e": "Ridge",
            "39f4f9b498b74f6480a09964e28db2b4": "Linear Regression",
            "6f1bc9e92e70410f82e9152275910f37": "Random Forest",
            "44b5e9a3c1c2461fb073f1a7fa3c7cb2": "Gradient Boosting",
            "a15af313e4a948c99fdbd15a117651e7": "Elastic Net",
            "01a605bd406c4344840bdb4136b63d27": "SVR",
        }

        for run_id, model_name in _RUN_MODELS.items():
            run_dir = experiment_dir / run_id / "artifacts"
            if not run_dir.exists():
                continue

            info: Dict[str, Any] = {
                "model_name": model_name,
                "run_id": run_id,
                "is_best": run_id == self._best_run_id,
                "importance": {},
                "forecast": {},
            }

            interp_dir = run_dir / "interpretability"
            if interp_dir.exists():
                for jf in interp_dir.glob("*.json"):
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if "importance" in jf.name:
                            info["importance"] = data
                        elif "forecast" in jf.name:
                            info["forecast"] = data
                    except (json.JSONDecodeError, OSError):
                        pass

            models_info.append(info)

        return models_info

    def inverse_yeo_johnson(self, y: float, lam: float = 0.61) -> float:
        """
        Inverse Yeo-Johnson transform to convert model output back to Pa scale.

        For y >= 0 and λ ≠ 0:
            x = (y * λ + 1)^(1/λ) - 1
        """
        if y >= 0:
            if abs(lam) < 1e-10:
                return np.expm1(y)
            return (y * lam + 1) ** (1.0 / lam) - 1
        else:
            if abs(lam - 2) < 1e-10:
                return -np.expm1(-y)
            return 1 - (-(2 - lam) * y + 1) ** (1.0 / (2 - lam))
