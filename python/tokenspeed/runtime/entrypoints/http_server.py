"""Control-plane HTTP server that runs alongside the smg gateway.

Exposes engine control endpoints on a dedicated port separate from the main
serving port. This server does NOT start its own Engine — it proxies calls
to the smg gateway that was started by ``tokenspeed serve``.

Architecture::

    Client (generation)  ──►  smg gateway  :8000  ──►  gRPC engine
    Client (control)     ──►  http_server   :8001  ──►  smg gateway  :8000

Started automatically by ``tokenspeed serve`` on ``main_port + 1``.
Override with ``--control-port PORT``.
"""

from __future__ import annotations

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
