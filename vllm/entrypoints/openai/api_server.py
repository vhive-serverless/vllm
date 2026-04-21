import asyncio
import importlib
import inspect
import json
import os
import re
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional, Set

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

# ---------------------------------------------------------------------------
# ADDED: results logging
# ---------------------------------------------------------------------------
_RESULTS_PATH = os.path.abspath(os.environ.get("VLLM_RESULTS_PATH", "vllm_results.jsonl"))
_results_queue: Optional[asyncio.Queue] = None

async def _background_writer() -> None:
    os.makedirs(os.path.dirname(_RESULTS_PATH) or ".", exist_ok=True)
    with open(_RESULTS_PATH, "a", encoding="utf-8") as f:
        while True:
            line = await _results_queue.get()
            if line is None:
                f.flush()
                _results_queue.task_done()
                return
            f.write(line + "\n")
            f.flush()
            _results_queue.task_done()

def _log(obj: dict) -> None:
    if _results_queue:
        _results_queue.put_nowait(json.dumps(obj))

# ---------------------------------------------------------------------------
# ADDED: internal forced-stream helpers
# ---------------------------------------------------------------------------

def _force_stream(request):
    """Return a copy of the request with stream=True."""
    try:
        d = request.model_dump()   # pydantic v2
    except AttributeError:
        d = request.dict()         # pydantic v1
    d["stream"] = True
    return type(request)(**d)


def _parse_sse(raw: List[str]) -> List[dict]:
    """Parse SSE lines into a list of chunk dicts."""
    chunks = []
    for line in raw:
        for part in line.split("\n"):
            part = part.strip()
            if not part.startswith("data:"):
                continue
            payload = part[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                chunks.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return chunks


def _reconstruct_completion(chunks: List[dict]) -> dict:
    """Collapse streamed chunks into a non-streaming /v1/completions response."""
    merged: dict = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in merged:
                merged[idx] = {"index": idx, "text": "",
                               "logprobs": None, "finish_reason": None}
            merged[idx]["text"] += choice.get("text", "")
            if choice.get("finish_reason"):
                merged[idx]["finish_reason"] = choice["finish_reason"]
    first, last = chunks[0], chunks[-1]
    return {"id": first.get("id"), "object": "text_completion",
            "created": first.get("created"), "model": first.get("model"),
            "choices": list(merged.values()), "usage": last.get("usage")}


def _reconstruct_chat(chunks: List[dict]) -> dict:
    """Collapse streamed chunks into a non-streaming /v1/chat/completions response."""
    merged: dict = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in merged:
                merged[idx] = {"index": idx,
                               "message": {"role": "assistant", "content": ""},
                               "finish_reason": None}
            merged[idx]["message"]["content"] += \
                choice.get("delta", {}).get("content", "")
            if choice.get("finish_reason"):
                merged[idx]["finish_reason"] = choice["finish_reason"]
    first, last = chunks[0], chunks[-1]
    return {"id": first.get("id"), "object": "chat.completion",
            "created": first.get("created"), "model": first.get("model"),
            "choices": list(merged.values()), "usage": last.get("usage")}


async def _drain_and_time(generator: AsyncGenerator, endpoint: str) -> tuple[dict, dict]:
    """
    Drain a streaming generator, record TTFT/TBT/E2E, return (timing, chunks).
    Single inference pass — no double GPU work.
    """
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_prev:  Optional[float] = None
    tbt_samples: List[float] = []
    raw_lines: List[str] = []

    async for chunk in generator:
        t_now = time.perf_counter()
        raw_lines.append(
            chunk.decode("utf-8", errors="replace")
            if isinstance(chunk, bytes) else str(chunk)
        )
        if t_first is None:
            t_first = t_now
        else:
            tbt_samples.append(t_now - t_prev)
        t_prev = t_now

    t_end  = time.perf_counter()
    e2e    = t_end - t_start
    ttft   = (t_first - t_start) if t_first is not None else e2e
    mean_tbt = (sum(tbt_samples) / len(tbt_samples)) if tbt_samples else 0.0

    timing = {
        "endpoint":   endpoint,
        "e2e_s":      round(e2e,      6),
        "ttft_s":     round(ttft,     6),
        "mean_tbt_s": round(mean_tbt, 6),
        "tbt_count":  len(tbt_samples),
    }
    return timing, _parse_sse(raw_lines)

# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    global _results_queue                               # ADDED
    _results_queue = asyncio.Queue()                    # ADDED
    writer = asyncio.create_task(_background_writer())  # ADDED
    _running_tasks.add(writer)                          # ADDED
    writer.add_done_callback(_running_tasks.discard)    # ADDED

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield

    await _results_queue.join()                         # ADDED
    _results_queue.put_nowait(None)                     # ADDED
    await writer                                        # ADDED


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


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        # CHANGED: force internal stream for real TTFT/TBT
        stream_req = _force_stream(request)
        stream_gen = await openai_serving_chat.create_chat_completion(
            stream_req, raw_request)
        if isinstance(stream_gen, ErrorResponse):
            return JSONResponse(content=stream_gen.model_dump(),
                                status_code=stream_gen.code)
        timing, chunks = await _drain_and_time(stream_gen, "/v1/chat/completions")
        _log(timing)
        if chunks:
            return JSONResponse(content=_reconstruct_chat(chunks))
        # fallback: use already-computed non-stream response
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        # CHANGED: force internal stream for real TTFT/TBT
        stream_req = _force_stream(request)
        stream_gen = await openai_serving_completion.create_completion(
            stream_req, raw_request)
        if isinstance(stream_gen, ErrorResponse):
            return JSONResponse(content=stream_gen.model_dump(),
                                status_code=stream_gen.code)
        timing, chunks = await _drain_and_time(stream_gen, "/v1/completions")
        _log(timing)
        if chunks:
            return JSONResponse(content=_reconstruct_completion(chunks))
        # fallback: use already-computed non-stream response
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
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
    logger.info("Results => %s", _RESULTS_PATH)

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
