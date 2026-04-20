import asyncio
import importlib
import inspect
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, AsyncGenerator, List, Optional, Set

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm
import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              EmbeddingRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()

# ---- Per-request timing output file ----
RESULTS_DIR = os.environ.get("VLLM_REQUEST_TIMING_RESULTS_DIR", "results")
RESULTS_PATH = os.environ.get(
    "VLLM_REQUEST_TIMING_RESULTS_PATH",
    os.path.join(RESULTS_DIR, "result_times.json"),
)

_results_write_lock = asyncio.Lock()


def _now_perf() -> float:
    return time.perf_counter()


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _get_request_id(raw_request: Request) -> str:
    rid = raw_request.headers.get("x-request-id")
    return rid or uuid.uuid4().hex


def _ensure_results_dir_exists() -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)


def _empty_tbt_obj() -> dict:
    return {
        "count": 0,
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
        "max_ms": 0.0,
    }


def _build_tbt_obj(tbt_sorted: List[float]) -> dict:
    if not tbt_sorted:
        return _empty_tbt_obj()
    mean = sum(tbt_sorted) / len(tbt_sorted)
    return {
        "count": len(tbt_sorted),
        "mean_ms": round(mean * 1000.0, 3),
        "p50_ms": round((_percentile(tbt_sorted, 50) or 0.0) * 1000.0, 3),
        "p95_ms": round((_percentile(tbt_sorted, 95) or 0.0) * 1000.0, 3),
        "p99_ms": round((_percentile(tbt_sorted, 99) or 0.0) * 1000.0, 3),
        "max_ms": round(tbt_sorted[-1] * 1000.0, 3),
    }


async def _append_result_jsonl(obj: dict) -> None:
    _ensure_results_dir_exists()
    line = json.dumps(obj, sort_keys=True)
    async with _results_write_lock:
        loop = asyncio.get_running_loop()

        def _write():
            with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        await loop.run_in_executor(None, _write)


# ---------------------------------------------------------------------------
# Timing context: collects token timestamps from any async generator
# ---------------------------------------------------------------------------

class RequestTimingContext:
    """
    Wraps an async generator (streaming or internally-forced-streaming) and
    records per-token timestamps so TTFT and TBT can be computed accurately.

    Usage:
        ctx = RequestTimingContext(t0=_now_perf())
        async for chunk in ctx.wrap(generator):
            ...   # process or yield chunk
        result = ctx.build_result(...)
    """

    def __init__(self, t0: float):
        self.t0 = t0
        self._first_token_t: Optional[float] = None
        self._prev_token_t: Optional[float] = None
        self._tbt: List[float] = []
        self._chunk_count = 0
        self.error: Optional[str] = None

    async def wrap(self, gen: Any) -> AsyncGenerator[Any, None]:
        try:
            async for item in gen:
                now = _now_perf()
                self._chunk_count += 1
                if self._first_token_t is None:
                    self._first_token_t = now
                    self._prev_token_t = now
                else:
                    assert self._prev_token_t is not None
                    self._tbt.append(now - self._prev_token_t)
                    self._prev_token_t = now
                yield item
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    def build_result(
        self,
        *,
        request_id: str,
        route: str,
        method: str,
        model: Optional[str],
        stream: bool,
        t_end: Optional[float] = None,
    ) -> dict:
        if t_end is None:
            t_end = _now_perf()

        e2e_s = t_end - self.t0

        if self._first_token_t is not None:
            ttft_s = self._first_token_t - self.t0
        else:
            # No token was observed — use e2e as a fallback (error path)
            ttft_s = e2e_s

        tbt_sorted = sorted(self._tbt)

        return {
            "event": "openai_request_timing",
            "request_id": request_id,
            "method": method,
            "route": route,
            "model": model,
            "stream": stream,
            "e2e_ms": round(e2e_s * 1000.0, 3),
            "ttft_ms": round(ttft_s * 1000.0, 3),
            "tbt": _build_tbt_obj(tbt_sorted),
            "chunks": self._chunk_count,
            "error": self.error,
            "results_path": os.path.abspath(RESULTS_PATH),
        }


# ---------------------------------------------------------------------------
# Streaming path: wrap generator, log at end, yield SSE chunks to client
# ---------------------------------------------------------------------------

async def _instrument_streaming_response(
    *,
    gen: Any,
    request_id: str,
    route: str,
    method: str,
    model: Optional[str],
    t0: float,
) -> AsyncGenerator[Any, None]:
    ctx = RequestTimingContext(t0=t0)
    try:
        async for chunk in ctx.wrap(gen):
            yield chunk
    finally:
        t_end = _now_perf()
        result_obj = ctx.build_result(
            request_id=request_id,
            route=route,
            method=method,
            model=model,
            stream=True,
            t_end=t_end,
        )
        try:
            await _append_result_jsonl(result_obj)
        except Exception as exc:
            logger.exception("Failed writing request timing: %s", exc)
        logger.info("%s", json.dumps(result_obj, sort_keys=True))


# ---------------------------------------------------------------------------
# Non-streaming path: force the engine to stream internally so we can measure
# TTFT and TBT, then return the fully-aggregated JSON response to the client.
#
# Strategy: call the serving layer with stream=True (monkey-patched on a copy
# of the request), drain all SSE chunks while timing, then call again with the
# original non-streaming request to get the properly formatted response object.
#
# Problem with that approach: two engine calls, so latency is doubled.
#
# Better strategy (used here): intercept *after* the serving layer returns a
# generator but *before* the generator is iterated.  For non-streaming vLLM
# already iterates the engine internally and returns a single
# ChatCompletionResponse / CompletionResponse.  We cannot easily hook inside
# that without patching vLLM internals.
#
# Practical solution: re-issue as stream=True, drain the stream to capture
# timing, reconstruct the non-streaming response from the streamed chunks.
# This adds only the overhead of one extra Python loop (no extra GPU work —
# vLLM deduplicates by request_id if we reuse it, but since we can't reuse
# ids across two separate calls we just accept the small serialization cost).
#
# SIMPLEST correct solution that avoids double-inference:
#   Override request.stream = True, get the generator, drain it for timing,
#   and return the *streamed* chunks collapsed into one JSON body via the
#   serving layer's own non-streaming call — but only time the internal drain.
#
# ACTUAL implementation below:
#   We patch the request object to stream=True, get a streaming generator from
#   the serving layer, drain it entirely while recording timestamps, collect
#   all SSE text, then parse the chunks to reconstruct a response dict.
#   This gives accurate TTFT + TBT + E2E with a single inference pass.
# ---------------------------------------------------------------------------

async def _drain_stream_for_timing(
    gen: Any,
) -> tuple[List[str], RequestTimingContext]:
    """Drain an SSE generator, collecting raw lines and timing info."""
    ctx = RequestTimingContext(t0=_now_perf())  # t0 reset by caller
    lines: List[str] = []
    async for chunk in ctx.wrap(gen):
        if isinstance(chunk, bytes):
            lines.append(chunk.decode("utf-8", errors="replace"))
        else:
            lines.append(str(chunk))
    return lines, ctx


def _parse_sse_to_completion_response(lines: List[str]) -> Optional[dict]:
    """
    Parse SSE lines from /v1/completions stream into a collapsed non-streaming
    response dict (best-effort; falls back to None on parse error).
    """
    import json as _json

    # SSE format: "data: <json>\n\n" or "data: [DONE]\n\n"
    chunks = []
    for line in lines:
        for part in line.split("\n"):
            part = part.strip()
            if part.startswith("data:"):
                payload = part[len("data:"):].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunks.append(_json.loads(payload))
                except _json.JSONDecodeError:
                    pass

    if not chunks:
        return None

    # Merge text from all choices
    first = chunks[0]
    merged_choices: dict = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in merged_choices:
                merged_choices[idx] = {
                    "index": idx,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": None,
                }
            merged_choices[idx]["text"] += choice.get("text", "")
            if choice.get("finish_reason"):
                merged_choices[idx]["finish_reason"] = choice["finish_reason"]

    last = chunks[-1]
    return {
        "id": first.get("id"),
        "object": "text_completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": list(merged_choices.values()),
        "usage": last.get("usage"),
    }


def _parse_sse_to_chat_response(lines: List[str]) -> Optional[dict]:
    """
    Parse SSE lines from /v1/chat/completions stream into a collapsed
    non-streaming response dict (best-effort).
    """
    import json as _json

    chunks = []
    for line in lines:
        for part in line.split("\n"):
            part = part.strip()
            if part.startswith("data:"):
                payload = part[len("data:"):].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunks.append(_json.loads(payload))
                except _json.JSONDecodeError:
                    pass

    if not chunks:
        return None

    first = chunks[0]
    merged_choices: dict = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in merged_choices:
                merged_choices[idx] = {
                    "index": idx,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            delta = choice.get("delta", {})
            merged_choices[idx]["message"]["content"] += delta.get("content", "")
            if choice.get("finish_reason"):
                merged_choices[idx]["finish_reason"] = choice["finish_reason"]

    last = chunks[-1]
    return {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": list(merged_choices.values()),
        "usage": last.get("usage"),
    }


# ---------------------------------------------------------------------------
# Helpers to get a streaming generator for a non-streaming request
# ---------------------------------------------------------------------------

def _force_stream(request):
    """Return a shallow copy of the request with stream=True."""
    # Pydantic v1 / v2 compatible copy
    try:
        d = request.model_dump()          # pydantic v2
    except AttributeError:
        d = request.dict()                # pydantic v1
    d["stream"] = True
    return type(request)(**d)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


route = Mount("/metrics", make_asgi_app())
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    t0 = _now_perf()
    request_id = _get_request_id(raw_request)
    route = "/v1/chat/completions"
    method = raw_request.method
    model = getattr(request, "model", None)

    if request.stream:
        # --- True streaming: instrument generator directly ---
        generator = await openai_serving_chat.create_chat_completion(
            request, raw_request)

        if isinstance(generator, ErrorResponse):
            t_end = _now_perf()
            result_obj = {
                "event": "openai_request_timing",
                "request_id": request_id, "method": method, "route": route,
                "model": model, "stream": True,
                "e2e_ms": round((t_end - t0) * 1000.0, 3),
                "ttft_ms": round((t_end - t0) * 1000.0, 3),
                "tbt": _empty_tbt_obj(), "chunks": 0,
                "error": f"ErrorResponse code={generator.code}",
                "results_path": os.path.abspath(RESULTS_PATH),
            }
            await _append_result_jsonl(result_obj)
            logger.info("%s", json.dumps(result_obj, sort_keys=True))
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        wrapped = _instrument_streaming_response(
            gen=generator,
            request_id=request_id,
            route=route,
            method=method,
            model=model,
            t0=t0,
        )
        return StreamingResponse(content=wrapped, media_type="text/event-stream")

    else:
        # --- Non-streaming: force internal stream to get real TTFT/TBT ---
        stream_request = _force_stream(request)
        generator = await openai_serving_chat.create_chat_completion(
            stream_request, raw_request)

        if isinstance(generator, ErrorResponse):
            t_end = _now_perf()
            result_obj = {
                "event": "openai_request_timing",
                "request_id": request_id, "method": method, "route": route,
                "model": model, "stream": False,
                "e2e_ms": round((t_end - t0) * 1000.0, 3),
                "ttft_ms": round((t_end - t0) * 1000.0, 3),
                "tbt": _empty_tbt_obj(), "chunks": 0,
                "error": f"ErrorResponse code={generator.code}",
                "results_path": os.path.abspath(RESULTS_PATH),
            }
            await _append_result_jsonl(result_obj)
            logger.info("%s", json.dumps(result_obj, sort_keys=True))
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        # Drain the stream while timing
        ctx = RequestTimingContext(t0=t0)
        lines: List[str] = []
        async for chunk in ctx.wrap(generator):
            if isinstance(chunk, bytes):
                lines.append(chunk.decode("utf-8", errors="replace"))
            else:
                lines.append(str(chunk))

        t_end = _now_perf()
        result_obj = ctx.build_result(
            request_id=request_id, route=route, method=method,
            model=model, stream=False, t_end=t_end,
        )
        await _append_result_jsonl(result_obj)
        logger.info("%s", json.dumps(result_obj, sort_keys=True))

        # Reconstruct non-streaming response from SSE chunks
        resp_dict = _parse_sse_to_chat_response(lines)
        if resp_dict is not None:
            return JSONResponse(content=resp_dict)

        # Fallback: re-issue as non-streaming (rare error path)
        fallback = await openai_serving_chat.create_chat_completion(
            request, raw_request)
        if isinstance(fallback, ErrorResponse):
            return JSONResponse(content=fallback.model_dump(),
                                status_code=fallback.code)
        assert isinstance(fallback, ChatCompletionResponse)
        return JSONResponse(content=fallback.model_dump())


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    t0 = _now_perf()
    request_id = _get_request_id(raw_request)
    route = "/v1/completions"
    method = raw_request.method
    model = getattr(request, "model", None)

    if request.stream:
        # --- True streaming ---
        generator = await openai_serving_completion.create_completion(
            request, raw_request)

        if isinstance(generator, ErrorResponse):
            t_end = _now_perf()
            result_obj = {
                "event": "openai_request_timing",
                "request_id": request_id, "method": method, "route": route,
                "model": model, "stream": True,
                "e2e_ms": round((t_end - t0) * 1000.0, 3),
                "ttft_ms": round((t_end - t0) * 1000.0, 3),
                "tbt": _empty_tbt_obj(), "chunks": 0,
                "error": f"ErrorResponse code={generator.code}",
                "results_path": os.path.abspath(RESULTS_PATH),
            }
            await _append_result_jsonl(result_obj)
            logger.info("%s", json.dumps(result_obj, sort_keys=True))
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        wrapped = _instrument_streaming_response(
            gen=generator,
            request_id=request_id,
            route=route,
            method=method,
            model=model,
            t0=t0,
        )
        return StreamingResponse(content=wrapped, media_type="text/event-stream")

    else:
        # --- Non-streaming: force internal stream for accurate timing ---
        stream_request = _force_stream(request)
        generator = await openai_serving_completion.create_completion(
            stream_request, raw_request)

        if isinstance(generator, ErrorResponse):
            t_end = _now_perf()
            result_obj = {
                "event": "openai_request_timing",
                "request_id": request_id, "method": method, "route": route,
                "model": model, "stream": False,
                "e2e_ms": round((t_end - t0) * 1000.0, 3),
                "ttft_ms": round((t_end - t0) * 1000.0, 3),
                "tbt": _empty_tbt_obj(), "chunks": 0,
                "error": f"ErrorResponse code={generator.code}",
                "results_path": os.path.abspath(RESULTS_PATH),
            }
            await _append_result_jsonl(result_obj)
            logger.info("%s", json.dumps(result_obj, sort_keys=True))
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)

        # Drain stream while recording per-token timestamps
        ctx = RequestTimingContext(t0=t0)
        lines: List[str] = []
        async for chunk in ctx.wrap(generator):
            if isinstance(chunk, bytes):
                lines.append(chunk.decode("utf-8", errors="replace"))
            else:
                lines.append(str(chunk))

        t_end = _now_perf()
        result_obj = ctx.build_result(
            request_id=request_id, route=route, method=method,
            model=model, stream=False, t_end=t_end,
        )
        await _append_result_jsonl(result_obj)
        logger.info("%s", json.dumps(result_obj, sort_keys=True))

        # Reconstruct non-streaming JSON from SSE chunks
        resp_dict = _parse_sse_to_completion_response(lines)
        if resp_dict is not None:
            return JSONResponse(content=resp_dict)

        # Fallback (shouldn't happen in normal operation)
        fallback = await openai_serving_completion.create_completion(
            request, raw_request)
        if isinstance(fallback, ErrorResponse):
            return JSONResponse(content=fallback.model_dump(),
                                status_code=fallback.code)
        return JSONResponse(content=fallback.model_dump())


# ---------------------------------------------------------------------------
# /v1/embeddings  (no streaming — timing is purely E2E)
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    t0 = _now_perf()
    request_id = _get_request_id(raw_request)
    route = "/v1/embeddings"
    method = raw_request.method
    model = getattr(request, "model", None)

    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)

    t_end = _now_perf()
    e2e_ms = (t_end - t0) * 1000.0

    if isinstance(generator, ErrorResponse):
        result_obj = {
            "event": "openai_request_timing",
            "request_id": request_id, "method": method, "route": route,
            "model": model, "stream": False,
            "e2e_ms": round(e2e_ms, 3),
            # Embeddings have no token-level timing; e2e is the only signal.
            "ttft_ms": round(e2e_ms, 3),
            "tbt": _empty_tbt_obj(), "chunks": 0,
            "error": f"ErrorResponse code={generator.code}",
            "results_path": os.path.abspath(RESULTS_PATH),
        }
        await _append_result_jsonl(result_obj)
        logger.info("%s", json.dumps(result_obj, sort_keys=True))
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    result_obj = {
        "event": "openai_request_timing",
        "request_id": request_id, "method": method, "route": route,
        "model": model, "stream": False,
        "e2e_ms": round(e2e_ms, 3),
        "ttft_ms": round(e2e_ms, 3),
        "tbt": _empty_tbt_obj(), "chunks": 1,
        "error": None,
        "results_path": os.path.abspath(RESULTS_PATH),
    }
    await _append_result_jsonl(result_obj)
    logger.info("%s", json.dumps(result_obj, sort_keys=True))
    return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info("vLLM API server version %s", vllm.__version__)
    logger.info("args: %s", args)
    logger.info("Per-request timing output (JSONL) => %s",
                os.path.abspath(RESULTS_PATH))

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)

    if engine_args.image_input_type is not None and \
        engine_args.image_input_type.upper() != "PIXEL_VALUES":
        raise ValueError(
            f"Invalid image_input_type: {engine_args.image_input_type}. "
            "Only --image-input-type 'pixel_values' is supported for serving "
            "vision language models with the vLLM API server.")

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())

    openai_serving_chat = OpenAIServingChat(engine, model_config,
                                            served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names, args.lora_modules)
    openai_serving_embedding = OpenAIServingEmbedding(engine, model_config,
                                                      served_model_names)
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
