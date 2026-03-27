"""
main.py
=======
FastAPI application entry point for the Heat Exchanger Monitoring backend.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.scenarios import router as scenarios_router
from app.routes.interpretability import router as interpretability_router
from app.services.data_service import DataService
from app.services.model_service import ModelService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hx_backend")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: load data and verify model connectivity."""
    logger.info("Loading scenario data from %s ...", settings.data_dir)
    application.state.data_service = DataService(
        data_dir=settings.data_dir,
        eda_dir=settings.eda_dir,
    )
    application.state.model_service = ModelService(
        serve_uri=settings.mlflow_model_serve_uri,
        tracking_uri=settings.mlflow_tracking_uri,
        artifacts_dir=settings.artifacts_dir,
        best_run_id=settings.best_model_run_id,
    )
    logger.info("Backend ready. Model serve URI: %s", settings.mlflow_model_serve_uri)
    yield
    logger.info("Shutting down backend.")


app = FastAPI(
    title="HX Frost Monitor API",
    description="Real-time LNG heat exchanger frost monitoring and ML inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scenarios_router, prefix="/api")
app.include_router(interpretability_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "hx-backend"}
