"""FastAPI application entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from config.config import config
from routers import auth_router, chat_router, interview_router, user_router
from utils.logger import get_logger
from utils.monitoring import runtime_metrics
from utils.resume_task_queue import start_resume_task_worker, stop_resume_task_worker
from utils.redis_client import redis_is_ready


logger = get_logger("main")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESUME_DIR = STATIC_DIR / "resumes"


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("application starting")
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(RESUME_DIR, exist_ok=True)
    start_resume_task_worker()
    yield
    stop_resume_task_worker()
    logger.info("application shutting down")


app = FastAPI(
    title="JobAgent API",
    description="Job assistant and interview service backend",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET_KEY,
    max_age=86400 * 7,
    https_only=config.SESSION_HTTPS_ONLY,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth_router.router, prefix="/api/auth", tags=["auth"])
app.include_router(user_router.router, prefix="/api/user", tags=["user"])
app.include_router(chat_router.router, prefix="/api/chat", tags=["chat"])
app.include_router(interview_router.router, prefix="/api/interview", tags=["interview"])


@app.middleware("http")
async def metrics_middleware(request, call_next):
    started = perf_counter()
    route = request.scope.get("route")
    path_template = getattr(route, "path", request.url.path)
    method = request.method
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (perf_counter() - started) * 1000.0
        runtime_metrics.record_request(
            method=method,
            path=path_template,
            status_code=500,
            latency_ms=elapsed_ms,
        )
        runtime_metrics.maybe_emit_alert()
        raise

    elapsed_ms = (perf_counter() - started) * 1000.0
    runtime_metrics.record_request(
        method=method,
        path=path_template,
        status_code=response.status_code,
        latency_ms=elapsed_ms,
    )
    runtime_metrics.maybe_emit_alert()
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "redis": "ok" if redis_is_ready() else "degraded",
        "env": config.ENV,
    }


@app.get("/api/metrics")
async def metrics():
    return runtime_metrics.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
