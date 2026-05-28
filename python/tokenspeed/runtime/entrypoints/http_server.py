"""Lightweight HTTP server that wraps Engine directly (no smg/gRPC dependency).

Useful for RL training (pause/continue generation, weight updates),
benchmarking, and any use-case that needs direct HTTP access to the engine
without the smg gateway overhead.

Usage::

    python -m tokenspeed.entrypoints.http_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8080 \
        --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tokenspeed.runtime.entrypoints.engine import Engine
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

app = FastAPI()
_engine: Engine | None = None


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/readiness")
async def readiness():
    if _engine is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/get_server_info")
async def get_server_info():
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(None, _engine.get_server_info)
    return JSONResponse(info)


@app.get("/v1/models")
async def get_models():
    info = await asyncio.get_event_loop().run_in_executor(None, _engine.get_server_info)
    model_id = info.get("model_path", "unknown")
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "tokenspeed"}],
        }
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.pop("prompt", "")
    sampling_params = body.pop("sampling_params", body)
    stream = sampling_params.pop("stream", False)

    if stream:

        async def _gen():
            async for chunk in _engine.async_generate(
                prompt, sampling_params=sampling_params
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    result = await _engine.async_generate(prompt, sampling_params=sampling_params)
    return JSONResponse(result)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model", "")
    max_tokens = body.get("max_tokens", 16)
    temperature = body.get("temperature", 1.0)
    stream = body.get("stream", False)

    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        **{
            k: v
            for k, v in body.items()
            if k not in ("prompt", "model", "max_tokens", "temperature", "stream")
        },
    }

    if stream:

        async def _gen():
            async for chunk in _engine.async_generate(
                prompt, sampling_params=sampling_params
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    result = await _engine.async_generate(prompt, sampling_params=sampling_params)
    text = result.get("text", "")
    completion_tokens = result.get("meta_info", {}).get("completion_tokens", 0)
    prompt_tokens = result.get("meta_info", {}).get("prompt_tokens", 0)
    return JSONResponse(
        {
            "id": result.get("meta_info", {}).get("id", ""),
            "object": "text_completion",
            "model": model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": result.get("meta_info", {}).get(
                        "finish_reason", "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    # Flatten messages to a prompt string; callers should send pre-tokenized prompts
    # or use the /generate endpoint for more control.
    prompt = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
    )
    max_tokens = body.get("max_tokens", 16)
    temperature = body.get("temperature", 1.0)
    model = body.get("model", "")

    sampling_params = {"max_new_tokens": max_tokens, "temperature": temperature}
    result = await _engine.async_generate(prompt, sampling_params=sampling_params)
    text = result.get("text", "")
    completion_tokens = result.get("meta_info", {}).get("completion_tokens", 0)
    prompt_tokens = result.get("meta_info", {}).get("prompt_tokens", 0)
    return JSONResponse(
        {
            "id": result.get("meta_info", {}).get("id", ""),
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": result.get("meta_info", {}).get(
                        "finish_reason", "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# Cache / profiling
# ---------------------------------------------------------------------------


@app.post("/flush_cache")
async def flush_cache():
    await asyncio.get_event_loop().run_in_executor(None, _engine.flush_cache)
    return JSONResponse({"status": "ok"})


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile():
    await asyncio.get_event_loop().run_in_executor(None, _engine.start_profile)
    return JSONResponse({"status": "ok"})


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile():
    await asyncio.get_event_loop().run_in_executor(None, _engine.stop_profile)
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# RL training — pause / continue generation (mirrors PR #270 / SGLang API)
# ---------------------------------------------------------------------------


@app.post("/pause_generation")
async def pause_generation(request: Request):
    """Pause the scheduler.

    Request body (JSON):
        mode: "abort" | "in_place" | "retract"  (default: "abort")

    "abort"    — aborts all in-flight requests then halts scheduling.
    "in_place" — halts new batches without touching running requests
                 (KV cache stays valid; intended for in-place weight updates).
    "retract"  — halts scheduling; running requests are retracted back to
                 the waiting queue with their KV pages retained.

    Requires Engine.pause_generation() (see PR #270).
    """
    if not hasattr(_engine, "pause_generation"):
        return JSONResponse(
            {"error": "pause_generation not available; requires PR #270"},
            status_code=501,
        )
    body = await request.json()
    mode = body.get("mode", "abort")
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _engine.pause_generation(mode)
    )
    return JSONResponse({"success": result.success, "message": result.message})


@app.post("/continue_generation")
async def continue_generation():
    """Resume the scheduler after a pause.

    Requires Engine.continue_generation() (see PR #270).
    """
    if not hasattr(_engine, "continue_generation"):
        return JSONResponse(
            {"error": "continue_generation not available; requires PR #270"},
            status_code=501,
        )
    result = await asyncio.get_event_loop().run_in_executor(
        None, _engine.continue_generation
    )
    return JSONResponse({"success": result.success, "message": result.message})


# ---------------------------------------------------------------------------
# Weight update (RL training)
# ---------------------------------------------------------------------------


@app.post("/init_weights_update_group")
async def init_weights_update_group(request: Request):
    body = await request.json()
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _engine.init_weights_update_group(
            master_address=body["master_address"],
            master_port=body["master_port"],
            rank_offset=body.get("rank_offset", 0),
            world_size=body["world_size"],
            group_name=body.get("group_name", ""),
            backend=body.get("backend", "nccl"),
        ),
    )
    return JSONResponse({"status": "ok"})


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request):
    body = await request.json()
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _engine.update_weights_from_distributed(
            name=body["name"],
            dtype=body["dtype"],
            shape=body["shape"],
        ),
    )
    return JSONResponse({"success": result[0], "message": result[1]})


@app.post("/release_memory_occupation")
async def release_memory_occupation(request: Request):
    body = await request.json()
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: _engine.release_memory_occupation(body.get("tags"))
    )
    return JSONResponse({"status": "ok"})


@app.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request):
    body = await request.json()
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: _engine.resume_memory_occupation(body.get("tags"))
    )
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="TokenSpeed direct HTTP server (no smg gateway)",
        add_help=False,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-h", "--help", action="store_true")
    args, engine_argv = parser.parse_known_args(argv)
    return args, engine_argv


def _build_engine(engine_argv: list[str]) -> Engine:
    from tokenspeed.runtime.utils.server_args import ServerArgs

    server_args = ServerArgs.from_cli_args(engine_argv)
    return Engine(**vars(server_args))


def run(
    host: str = "127.0.0.1",
    port: int = 8080,
    engine_kwargs: dict[str, Any] | None = None,
):
    """Start the HTTP server programmatically.

    Args:
        host: Bind address.
        port: Bind port.
        engine_kwargs: Keyword arguments forwarded to ``Engine()``.
    """
    global _engine
    _engine = Engine(**(engine_kwargs or {}))
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv=None):
    global _engine

    args, engine_argv = _parse_args(argv)
    if args.help or not engine_argv:
        print(__doc__)
        print(
            "Engine args are passed through to ServerArgs (see tokenspeed serve --help)."
        )
        return

    logger.info("Starting TokenSpeed HTTP server on %s:%d", args.host, args.port)
    _engine = _build_engine(engine_argv)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
