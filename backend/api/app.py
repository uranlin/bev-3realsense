"""ASGI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.services.runtime import NavigationRuntime

from .routes import create_router


def create_app():
    runtime = NavigationRuntime()
    app = FastAPI(title="BEV Navigation", lifespan=runtime.lifespan)
    app.state.runtime = runtime
    app.include_router(create_router(runtime))
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
