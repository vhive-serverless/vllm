
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



def test_process(task_queue_dict:Dict[int, Queue], result_queue_dict:Dict[int, Queue], lock_dict:Dict[int, any], instance_uuid: str, rank: int):
    global world_size
    gpu_proxy_client_dict: Dict[int, GPUProxyClient] = {}
    for i in range(world_size):
        gpu_proxy_client_dict[i] = GPUProxyClient(i, task_queue_dict[i], result_queue_dict[i], lock_dict[i], instance_uuid)
    args = (6,1,9)
    expected_output = ""
    for i, arg in enumerate(args):
        expected_output += f"arg{i}: {arg};"
    gpu_proxy_client_dict[rank].start()
    output = gpu_proxy_client_dict[rank].execute_method("print_args_and_return", args)
    assert output == expected_output, f"expected: {expected_output}, got: {output}"
    gpu_proxy_client_dict[rank].stop()
    return
    



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Liquid OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    proxy_manager = build_proxy_manager(world_size=world_size,vllm_config=vllm_config, mp_manager=mp_manager)
    rank = 1
    index = 6
    instance_uuid = "process_0"
    p = Process(target=test_process, args=(task_queue_dict, result_queue_dict, lock_dict, instance_uuid, rank, index)) 
    p.start()
    p.join()
