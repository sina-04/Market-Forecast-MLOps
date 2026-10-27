"""Authenticated historical forecast HTTP API."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, field_validator

from marketforecast.io import read_json
from . import LIMITATION, MODE
from .audit import AuditStore
from .monitoring import Monitor
from .runtime import Predictor, local_timestamp

logger = logging.getLogger("marketforecast.service")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: str

    @field_validator("as_of")
    @classmethod
    def timestamp(cls, value):
        local_timestamp(value)
        return value


class PredictionResponse(BaseModel):
    prediction_id: str
    release_id: str
    registry_version: str
    mode: Literal["historical_replay"]
    unit: str
    limitations: str
    input_timestamp: str
    target_timestamp: str
    predicted_adjusted_price: float
    persistence_forecast: float


def create_app(bundle=None, state=None, api_key=None, registration=None):
    @asynccontextmanager
    async def lifespan(app):
        key = api_key or os.getenv("MARKETFORECAST_API_KEY")
        if not key or len(key) < 24:
            raise ValueError("MARKETFORECAST_API_KEY must contain at least 24 characters")
        app.state.key = key
        predictor = Predictor(bundle or os.environ.get("MARKETFORECAST_BUNDLE", "/bundle"))
        store = AuditStore(state or os.environ.get("MARKETFORECAST_STATE", "/state/audit.sqlite"))
        store.ready()
        app.state.predictor, app.state.store = predictor, store
        app.state.monitor = Monitor(predictor, store)
        path = Path(registration or os.environ.get("MARKETFORECAST_REGISTRATION", "/registry/registration.json"))
        if not path.is_file():
            raise ValueError("Approved registry registration is required before serving")
        registered = read_json(path)
        if registered["release_id"] != predictor.release["release_id"] or not registered.get("verified"):
            raise ValueError("Registry registration differs from verified release")
        app.state.version = str(registered["registry_version"])
        app.state.monitor.evaluate()

        async def evaluate_loop():
            while True:
                await asyncio.sleep(60)
                try:
                    await asyncio.to_thread(app.state.monitor.evaluate)
                except Exception:
                    app.state.monitor.failures.inc()
                    logger.exception("Historical diagnostics failed")

        task = asyncio.create_task(evaluate_loop())
        try:
            yield
        finally:
            app.state.monitor.ready.set(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="Foolad Hourly LSTM — Historical Replay", version="1.0.0",
                  description=LIMITATION, lifespan=lifespan)

    def authenticated(request: Request, key=Security(api_key_header)):
        if not key or not hmac.compare_digest(key.encode(), request.app.state.key.encode()):
            raise HTTPException(401, "A valid X-API-Key is required")

    @app.middleware("http")
    async def observe(request, call_next):
        start = time.perf_counter()
        request.state.prediction_id = str(uuid.uuid4())
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Request failed", extra={"request_id": request.state.prediction_id})
            response = JSONResponse({"detail": "Internal service error"}, status_code=500)
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        route_name = route.path if route else "unmatched"
        monitor = getattr(request.app.state, "monitor", None)
        if request.url.path == "/v1/predict" and response.status_code != 200:
            try:
                request.app.state.store.record(request.state.prediction_id,
                    request.app.state.predictor.release["release_id"], response.status_code, elapsed,
                    registry_version=request.app.state.version)
            except Exception:
                logger.exception("Failure audit could not be recorded")
                response = JSONResponse({"detail": "Audit storage unavailable"}, status_code=503)
        if monitor and route_name != "/metrics":
            monitor.requests.labels(route_name, str(response.status_code)).inc()
            monitor.latency.labels(route_name).observe(elapsed)
        response.headers["X-Request-ID"] = request.state.prediction_id
        logger.info("request route=%s status=%s seconds=%.6f request_id=%s", route_name,
                    response.status_code, elapsed, request.state.prediction_id)
        return response

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(request: Request):
        try:
            request.app.state.store.ready()
        except (sqlite3.Error, OSError):
            request.app.state.monitor.ready.set(0)
            raise HTTPException(503, "Audit storage unavailable")
        request.app.state.monitor.ready.set(1)
        return {"status": "ready", "release_id": request.app.state.predictor.release["release_id"]}

    @app.get("/v1/model", dependencies=[Depends(authenticated)])
    def model(request: Request):
        predictor = request.app.state.predictor
        return {"release_id": predictor.release["release_id"], "registry_version": request.app.state.version,
                "mode": MODE, "target": predictor.metadata["target"], "unit": predictor.metadata["unit"],
                "sequence_length": 60, "features": 49, "supported_windows": len(predictor.index),
                "snapshot": read_json(predictor.root / "snapshot/manifest.json"), "limitations": LIMITATION}

    @app.get("/v1/replay/timestamps", dependencies=[Depends(authenticated)])
    def timestamps(request: Request):
        return {"mode": MODE, "timestamps": request.app.state.predictor.timestamps()}

    @app.post("/v1/predict", dependencies=[Depends(authenticated)], response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request):
        start = time.perf_counter()
        predictor = request.app.state.predictor
        try:
            result = predictor.predict(body.as_of)
        except KeyError:
            raise HTTPException(404, "Timestamp is not an eligible historical test window")
        except ValueError:
            logger.exception("Model inference failed")
            raise HTTPException(500, "Model produced an invalid prediction")
        try:
            request.app.state.store.record(request.state.prediction_id, predictor.release["release_id"],
                200, time.perf_counter() - start, result["input_timestamp"], result, request.app.state.version)
        except (sqlite3.Error, OSError):
            request.app.state.monitor.ready.set(0)
            raise HTTPException(503, "Prediction could not be durably recorded")
        return {"prediction_id": request.state.prediction_id, "release_id": predictor.release["release_id"],
                "registry_version": request.app.state.version, "mode": MODE,
                "unit": predictor.metadata["unit"], "limitations": LIMITATION, **result}

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request):
        return Response(generate_latest(request.app.state.monitor.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return app
