"""
interpretability.py
===================
API routes for model interpretability (Business Questions A, B, C)
and MLflow artifact retrieval.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["interpretability"])


@router.get("/interpretability")
async def get_interpretability(request: Request):
    """
    Return interpretability data for the best model.

    Includes:
      - Feature importance (Question A: correlations with ΔP)
      - Observable proxy features (Question B: proxy for heavy components)
      - Runtime forecast (Question C: expected defrost date)
    """
    try:
        model_svc = request.app.state.model_service
        data = model_svc.get_interpretability_data()
        return data
    except Exception as exc:
        logger.error("Failed to get interpretability data: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/interpretability/figures/{filename}")
async def get_figure(request: Request, filename: str):
    """Serve a PNG figure from the MLflow artifacts."""
    model_svc = request.app.state.model_service
    fig_path = model_svc.get_figure_path(filename)
    if fig_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Figure '{filename}' not found in model artifacts.",
        )
    return FileResponse(fig_path, media_type="image/png")


@router.get("/models")
async def get_all_models(request: Request):
    """Return metrics and interpretability data for all trained models."""
    try:
        model_svc = request.app.state.model_service
        return {"models": model_svc.get_all_models_metrics()}
    except Exception as exc:
        logger.error("Failed to get models data: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
