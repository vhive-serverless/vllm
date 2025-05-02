from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from multiprocessing import Manager, Process
from multiprocessing.managers import SyncManager
from queue import Queue
import uvicorn
from vllm.executor.proxy_manager import ProxyManager
from vllm.executor.proxy_manager_utils import GPUProxyClient
from vllm.entrypoints.openai.protocol import CreateInstanceRequest, DeleteInstanceRequest
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm import EngineArgs, LLMEngine
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.utils import (FlexibleArgumentParser, get_open_zmq_ipc_path,
                        is_valid_ipv6_address, set_ulimit)
logger = init_logger('vllm.entrypoints.liquid.liquid')

world_size = 2
mp_manager = Manager()
task_queue_dict = mp_manager.dict()
result_queue_dict = mp_manager.dict()
lock_dict = mp_manager.dict()

parser = FlexibleArgumentParser(
    description="vLLM OpenAI-Compatible RESTful API server.")
parser = make_arg_parser(parser)

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

    global lock_dict
    for i in range(world_size):
        lock_dict[i] = mp_manager.Lock()

    return ProxyManager(task_queue_dict, result_queue_dict, vllm_config)


# Server setup
app = FastAPI()


@app.post("/create_instance")
def create_instance(req: CreateInstanceRequest):
    global proxy_manager, mp_manager, task_queue_dict, result_queue_dict, lock_dict
    # proxy_manager.create_instance(req.uuid, req.gpu_ids)
    # A test process that enqueue items into task queue
    instance_uuid = req.uuid
    cli = req.cli
    envs = req.env
    args = req.args
    # Find the cuda visible devices
    gpu_ids = []
    prefix = "CUDA_VISIBLE_DEVICES="
    for env in envs:
        if env.startswith(prefix):
            visible_devices_str_list = env.removeprefix(prefix).split(',')
            gpu_ids = [int(d) for d in visible_devices_str_list]
    assert len(gpu_ids) != 0, f"Didn't specify CUDA_VISIBLE_DEVICES in the create request!"
    assert "serve" in args
    args = ["--model" if arg == "serve" else arg for arg in args]
    args = parser.parse_args(args)
    print(args)

    # First let the proxy manager adjust tensor model parallel groups
    proxy_manager.create_instance(instance_uuid, gpu_ids)
    return {"status": "created", "uuid": req.uuid}

def test_process(task_queue_dict:Dict[int, Queue], result_queue_dict:Dict[int, Queue], lock_dict:Dict[int, any], instance_uuid: str, rank: int, index: int):
    global world_size
    gpu_proxy_client_dict: Dict[int, GPUProxyClient] = {}
    for i in range(world_size):
        gpu_proxy_client_dict[i] = GPUProxyClient(i, task_queue_dict[i], result_queue_dict[i], lock_dict[i], instance_uuid)
    gpu_proxy_client_dict[rank].start()
    output = gpu_proxy_client_dict[rank].execute_method("print_args_and_return", [index])
    logger.info(f"output from rank: {rank} is: {output}")
    gpu_proxy_client_dict[rank].stop()
    return
    


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
    proxy_manager = build_proxy_manager(world_size=world_size,vllm_config=vllm_config, mp_manager=mp_manager)

    uvicorn.run(app=app, host=args.host, port=args.port)