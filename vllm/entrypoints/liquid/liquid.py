from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from vllm.executor.proxy_manager import ProxyManager
from vllm.entrypoints.openai.protocol import CreateInstanceRequest, DeleteInstanceRequest
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm import EngineArgs, LLMEngine
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.utils import (FlexibleArgumentParser, get_open_zmq_ipc_path,
                        is_valid_ipv6_address, set_ulimit)
logger = init_logger('vllm.entrypoints.liquid.liquid')


# This would be your real builder
def build_proxy_manager(world_size: int, vllm_config: VllmConfig) -> ProxyManager:
    logger.info("Building Proxy Manager")
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.world_size = world_size
    logger.info(f"{vllm_config}")
    return ProxyManager(vllm_config)


# Server setup
app = FastAPI()

# @app.on_event("startup")
# def startup_event():
#     global proxy_manager
#     proxy_manager = build_proxy_manager()

@app.post("/create_instance")
def create_instance(req: CreateInstanceRequest):
    global proxy_manager
    proxy_manager.create_instance(req.uuid, req.gpu_ids)
    return {"status": "created", "uuid": req.uuid}

@app.post("/delete_instance")
def delete_instance(req: DeleteInstanceRequest):
    global proxy_manager
    proxy_manager.delete_instance(req.uuid)
    return {"status": "deleted", "uuid": req.uuid}

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Liquid OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    proxy_manager = build_proxy_manager(world_size=2,vllm_config=vllm_config)

    uvicorn.run(app=app, host=args.host, port=args.port)