"""Control-plane HTTP server that runs alongside the smg gateway.

Exposes engine control endpoints (pause/continue generation, weight updates,
cache flush, etc.) on a dedicated port separate from the main serving port.

This server does NOT start its own Engine — it proxies control calls to the
smg gateway that was started by ``tokenspeed serve``.

Architecture::

    Client (generation)  ──►  smg gateway  :8080  ──►  gRPC engine
    Client (control)     ──►  http_server   :8081  ──►  smg gateway  :8080

Usage via ``tokenspeed serve``::

    tokenspeed serve --model <path> --port 8080 --control-port 8081

The ``--control-port`` flag is the only addition; all other ``tokenspeed serve``
flags are unchanged.
"""

from __future__ import annotations

import asyncio

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

app = FastAPI()

# URL of the smg gateway — set before uvicorn.run() is called.
_gateway_url: str = ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/readiness")
async def readiness():
    return JSONResponse({"status": "ready"})


# ---------------------------------------------------------------------------
# Generic proxy helper
# ---------------------------------------------------------------------------


async def _proxy(method: str, path: str, body: dict | None = None) -> JSONResponse:
    url = f"{_gateway_url.rstrip('/')}/{path.lstrip('/')}"
    async with aiohttp.ClientSession() as session:
        request_fn = getattr(session, method.lower())
        async with request_fn(
            url, json=body, timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            data = await resp.json(content_type=None)
            return JSONResponse(data, status_code=resp.status)


# ---------------------------------------------------------------------------
# Cache / profiling
# ---------------------------------------------------------------------------


@app.post("/flush_cache")
async def flush_cache():
    return await _proxy("POST", "/flush_cache")


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile():
    return await _proxy("POST", "/start_profile")


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile():
    return await _proxy("POST", "/stop_profile")


@app.get("/get_server_info")
async def get_server_info():
    return await _proxy("GET", "/get_server_info")


# ---------------------------------------------------------------------------
# RL training — pause / continue generation (requires PR #270 on smg side)
# ---------------------------------------------------------------------------


@app.post("/pause_generation")
async def pause_generation(request: Request):
    """Pause the scheduler via smg gateway.

    Request body (JSON):
        mode: "abort" | "in_place" | "retract"  (default: "abort")

    Proxied to ``POST {gateway}/pause_generation``.
    Requires the smg gateway to expose this route (see PR #270).
    """
    body = await request.json()
    return await _proxy("POST", "/pause_generation", body)


@app.post("/continue_generation")
async def continue_generation():
    """Resume the scheduler via smg gateway.

    Proxied to ``POST {gateway}/continue_generation``.
    Requires the smg gateway to expose this route (see PR #270).
    """
    return await _proxy("POST", "/continue_generation")


# ---------------------------------------------------------------------------
# Weight update (RL training)
# ---------------------------------------------------------------------------


@app.post("/init_weights_update_group")
async def init_weights_update_group(request: Request):
    return await _proxy("POST", "/init_weights_update_group", await request.json())


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request):
    return await _proxy(
        "POST", "/update_weights_from_distributed", await request.json()
    )


@app.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    return await _proxy("POST", "/update_weights_from_disk", await request.json())


@app.post("/release_memory_occupation")
async def release_memory_occupation(request: Request):
    return await _proxy("POST", "/release_memory_occupation", await request.json())


@app.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request):
    return await _proxy("POST", "/resume_memory_occupation", await request.json())


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start(
    *,
    gateway_url: str,
    host: str = "127.0.0.1",
    port: int = 8081,
) -> None:
    """Start the control HTTP server (blocking).

    Args:
        gateway_url: Base URL of the smg gateway, e.g. ``http://127.0.0.1:8080``.
        host: Bind address for the control server.
        port: Bind port for the control server.
    """
    global _gateway_url
    _gateway_url = gateway_url
    logger.info(
        "Starting TokenSpeed control HTTP server on %s:%d (gateway: %s)",
        host,
        port,
        gateway_url,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")
