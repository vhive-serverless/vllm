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
import time
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

def test_case_first_single():
    parser = FlexibleArgumentParser(
        description="Liquid OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    proxy_manager = build_proxy_manager(world_size=world_size,vllm_config=vllm_config, mp_manager=mp_manager)
    instance_uuid = "process_0"
    gpu_ids = [0]
    proxy_manager.create_instance(instance_uuid, gpu_ids)
    proxy_manager.delete_instance(instance_uuid)
    time.sleep(1)
    instance_uuid = "process_1"
    gpu_ids = [0,1]
    proxy_manager.create_instance(instance_uuid, gpu_ids)
    proxy_manager.delete_instance(instance_uuid)

def test_case_first_multi():
    parser = FlexibleArgumentParser(
        description="Liquid OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    proxy_manager = build_proxy_manager(world_size=world_size,vllm_config=vllm_config, mp_manager=mp_manager)
    instance_uuid = "process_0"
    gpu_ids = [0,1]
    proxy_manager.create_instance(instance_uuid, gpu_ids)
    proxy_manager.delete_instance(instance_uuid)
    time.sleep(1)
    instance_uuid = "process_1"
    gpu_ids = [0]
    proxy_manager.create_instance(instance_uuid, gpu_ids)
    proxy_manager.delete_instance(instance_uuid)

def test_case_multi_single():
    parser = FlexibleArgumentParser(
        description="Liquid OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    proxy_manager = build_proxy_manager(world_size=world_size,vllm_config=vllm_config, mp_manager=mp_manager)
    instance_uuid_0 = "process_0"
    gpu_ids = [0]
    proxy_manager.create_instance(instance_uuid_0, gpu_ids)
    time.sleep(1)
    instance_uuid_1 = "process_1"
    gpu_ids = [1]
    proxy_manager.create_instance(instance_uuid_1, gpu_ids)
    time.sleep(1)
    proxy_manager.delete_instance(instance_uuid_0)
    proxy_manager.delete_instance(instance_uuid_1)



    



if __name__ == "__main__":
    test_case_first_single()
    test_case_first_multi()
    test_case_multi_single()