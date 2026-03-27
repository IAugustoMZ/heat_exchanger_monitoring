"""
scenarios.py
============
API routes for scenario listing and real-time data streaming with ML inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["scenarios"])


@router.get("/scenarios")
async def list_scenarios(request: Request):
    """Return available simulation scenarios with metadata."""
    try:
        data_svc = request.app.state.data_service
        return {
            "scenarios": data_svc.get_scenario_metadata(),
            "alarm_threshold_pa": settings.alarm_threshold_pa,
            "dp_ideal_pa": settings.dp_ideal_pa,
        }
    except Exception as exc:
        logger.error("Failed to list scenarios: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stream/{scenario}")
async def stream_scenario(
    request: Request,
    scenario: str,
    run_id: int = Query(default=1, ge=1, le=3),
    speed: float = Query(default=1.0, ge=0.1, le=10.0, description="Playback speed multiplier. 1.0 = 1 pt/s, 2.0 = 2 pts/s, 0.5 = 1 pt/2s"),
):
    """
    Server-Sent Events (SSE) endpoint that streams data points for a scenario.

    Each event contains:
      - The raw sensor readings (T_h_in, T_h_out, T_c_in, T_c_out, ΔP)
      - ML model prediction (dP_error_predicted)
      - Alert status and runtime forecast
      - Ground truth (for demo visualization)

    Points are emitted at 1-second intervals (configurable via STREAM_INTERVAL_SEC).
    """
    data_svc = request.app.state.data_service
    model_svc = request.app.state.model_service

    if scenario not in data_svc.scenarios:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{scenario}'. Available: {data_svc.scenarios}",
        )

    try:
        df = data_svc.get_scenario_data(scenario, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    async def event_generator() -> AsyncGenerator[dict, None]:
        history = []
        total_points = len(df)

        for idx in range(total_points):
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected from stream %s", scenario)
                break

            row = df.iloc[idx].to_dict()

            # Build model input features
            model_input = data_svc.build_model_input_row(
                scenario=scenario,
                current_row=row,
                history=history,
            )

            # Get ML prediction from MLflow serving
            try:
                prediction_transformed = await model_svc.predict(model_input)
                dp_error_predicted = model_svc.inverse_yeo_johnson(
                    prediction_transformed, lam=0.61
                )
                dp_error_predicted = max(0.0, dp_error_predicted)  # Physical constraint
                model_available = True
            except (ConnectionError, ValueError) as exc:
                logger.warning("Model prediction failed at point %d: %s", idx, exc)
                dp_error_predicted = None
                model_available = False

            # Compute actual dP_error for comparison
            dp_ideal = model_input["dP_ideal"]
            dp_actual = float(row["delta_P_Pa"])
            dp_error_actual = dp_actual - dp_ideal

            # Alert logic
            dp_total_predicted = (dp_ideal + dp_error_predicted) if dp_error_predicted is not None else None
            alarm_predicted = dp_total_predicted is not None and dp_total_predicted >= settings.alarm_threshold_pa
            alarm_actual = int(row.get("freezing_alarm", 0)) == 1

            # Runtime forecast with error propagation
            remaining_runtime_h = None
            rul_lower = None
            rul_upper = None
            dp_error_slope_pa_per_h = None
            defrost_date = None
            if dp_error_predicted is not None and len(history) >= 10:
                recent_preds = [h.get("_dp_error_pred", 0.0) for h in history[-20:] if "_dp_error_pred" in h]
                n = len(recent_preds)
                if n >= 5:
                    import numpy as np
                    from datetime import datetime, timedelta
                    xs = np.arange(n, dtype=float)
                    ys = np.array(recent_preds, dtype=float)
                    # Ordinary least-squares slope + intercept
                    coeffs = np.polyfit(xs, ys, 1)
                    slope = float(coeffs[0])
                    steps_per_hour = 3600 / 60  # ~60 s per data step
                    if slope > 1e-3:
                        error_threshold = settings.alarm_threshold_pa - dp_ideal
                        t_remaining_steps = (error_threshold - dp_error_predicted) / slope
                        if t_remaining_steps > 0:
                            remaining_runtime_h = round(t_remaining_steps / steps_per_hour, 2)
                            dp_error_slope_pa_per_h = round(slope * steps_per_hour, 4)
                            defrost_date = (datetime.utcnow() + timedelta(hours=remaining_runtime_h)).isoformat()

                            # Error propagation: σ_RUL via partial derivatives
                            y_fit = np.polyval(coeffs, xs)
                            residuals = ys - y_fit
                            s2 = float(np.var(residuals, ddof=2)) if n > 2 else 0.0
                            Sxx = float(np.sum((xs - xs.mean()) ** 2))
                            sigma_slope = float(np.sqrt(s2 / Sxx)) if Sxx > 0 else 0.0
                            sigma_pred = float(np.std(residuals)) if n > 1 else 0.0
                            sigma_rul_steps = float(np.sqrt(
                                (sigma_pred / slope) ** 2
                                + (t_remaining_steps / slope * sigma_slope) ** 2
                            ))
                            sigma_rul_h = sigma_rul_steps / steps_per_hour
                            rul_lower = round(max(0.0, remaining_runtime_h - sigma_rul_h), 2)
                            rul_upper = round(remaining_runtime_h + sigma_rul_h, 2)

            event_data = {
                "index": idx,
                "total_points": total_points,
                "progress": round((idx + 1) / total_points * 100, 1),

                # Raw sensor readings
                "t_s": float(row["t_s"]),
                "t_h": round(float(row["t_s"]) / 3600, 4),
                "T_h_in_K": round(float(row["T_h_in_K"]), 2),
                "T_h_out_K": round(float(row["T_h_out_K"]), 2),
                "T_c_in_K": round(float(row["T_c_in_K"]), 2),
                "T_c_out_K": round(float(row["T_c_out_K"]), 2),
                "delta_P_Pa": round(dp_actual, 2),

                # ML prediction
                "dP_error_predicted": round(dp_error_predicted, 2) if dp_error_predicted is not None else None,
                "dP_total_predicted": round(dp_total_predicted, 2) if dp_total_predicted is not None else None,
                "model_available": model_available,

                # Alerts
                "alarm_predicted": alarm_predicted,
                "alarm_actual": alarm_actual,
                "alarm_threshold_pa": settings.alarm_threshold_pa,

                # Runtime forecast
                "remaining_runtime_h": remaining_runtime_h,
                "rul_lower": rul_lower,
                "rul_upper": rul_upper,
                "dp_error_slope_pa_per_h": dp_error_slope_pa_per_h,
                "defrost_date": defrost_date,

                # Ground truth (for demo only)
                "dP_error_actual": round(dp_error_actual, 2),
                "delta_f_mean_m": float(row.get("delta_f_mean_m", 0)),
                "delta_f_max_m": float(row.get("delta_f_max_m", 0)),
                "U_mean_W_m2K": round(float(row.get("U_mean_W_m2K", 0)), 2),
            }

            yield {"event": "datapoint", "data": json.dumps(event_data)}

            # Store in history for lag computation
            row_with_pred = dict(row)
            row_with_pred["_dp_error_pred"] = dp_error_predicted if dp_error_predicted is not None else 0.0
            history.append(row_with_pred)

            # Throttle to configured interval divided by speed multiplier
            await asyncio.sleep(settings.stream_interval_sec / speed)

        # Final event
        yield {
            "event": "complete",
            "data": json.dumps({"message": f"Scenario '{scenario}' run {run_id} complete.", "total_points": total_points}),
        }

    return EventSourceResponse(event_generator())
