from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from queue import Queue
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

mp_manager = None
task_queue_dict = None
result_queue_dict = None

# This would be your real builder
def build_proxy_manager(world_size: int, vllm_config: VllmConfig, mp_manager: SyncManager) -> ProxyManager:
    logger.info("Building Proxy Manager")
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.world_size = world_size
    logger.info(f"{vllm_config}")
    global task_queue_dict 
    for i in range(world_size):
        task_queue_dict[i] = mp_manager.Queue()
    
    global result_queue_dict
    for i in range(world_size):
        result_queue_dict[i] = mp_manager.Queue()

    return ProxyManager(task_queue_dict, result_queue_dict, vllm_config)


# Server setup
app = FastAPI()

# @app.on_event("startup")
# def startup_event():
#     global proxy_manager
#     proxy_manager = build_proxy_manager()

@app.post("/create_instance")
def create_instance(req: CreateInstanceRequest):
    global proxy_manager, mp_manager, task_queue_dict, result_queue_dict
    # proxy_manager.create_instance(req.uuid, req.gpu_ids)
    # A test process that enqueue items into task queue
    rank = 1
    index = 6
    p = Process(target=test_process, args=(task_queue_dict, result_queue_dict, req.instance_uuid, rank, index)) 
    p.start()
    p.join()
    return {"status": "created", "uuid": req.instance_uuid}

def test_process(task_queue_dict:Dict[int, Queue], result_queue_dict:Dict[int, Queue], instance_uuid: str, rank: int, index: int):
    item = (instance_uuid, 0, "print_args_and_return", [index], {})
    task_queue_dict[rank].put(item)
    result = result_queue_dict[rank].get()
    logger.info(f"Result from rank: {rank} is: {result}")
    


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
    mp_manager = Manager()
    task_queue_dict = mp_manager.dict()
    result_queue_dict = mp_manager.dict()
    proxy_manager = build_proxy_manager(world_size=2,vllm_config=vllm_config, mp_manager=mp_manager)

    uvicorn.run(app=app, host=args.host, port=args.port)