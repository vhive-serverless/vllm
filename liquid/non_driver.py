
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)

from vllm.worker.worker_base import WorkerWrapperBase
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata, SequenceData, SamplingParams
from vllm.core.block_manager_v2 import BlockTable
from functools import partial
from typing import Optional, List, Dict, Any, Tuple, Generator
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from liquid.worker import NUM_SHARDS
import time
from liquid.utils import get_gpu_memory_usage
import torch.distributed as dist
import torch


model = "facebook/opt-125m"
engine_args = EngineArgs(model=model)
engine_config = engine_args.create_engine_config()

def _get_worker_kwargs(
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
    """Return worker init args for a given rank."""
    if distributed_init_method is None:
        distributed_init_method = get_distributed_init_method(
            get_ip(), 35281)

    shard_ids = [i for i in range(NUM_SHARDS)]
    engine_config.parallel_config.tensor_parallel_size = 2
    engine_config.parallel_config.world_size = 2
    return dict(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        load_config=engine_config.load_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        lora_config=engine_config.lora_config,
        vision_language_config=engine_config.vision_language_config,
        speculative_config=engine_config.speculative_config,
        is_driver_worker=rank == 0,
        shard_ids=shard_ids,
    )

def _create_worker(local_rank: int = 0,
                   rank: int = 0,
                   distributed_init_method: Optional[str] = None):

        worker_module_name = "liquid.worker.liquid_worker"
        worker_class_name = "Worker"

        wrapper = WorkerWrapperBase(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
        )
        wrapper.init_worker(**_get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

def receive_sharded_weights_iterator(src_rank, state_dict: Dict[str, torch.Tensor]):
    for name, param in state_dict:
        if hasattr(param, "num_shards"):
            dist.recv(param.data, src=src_rank)

def init_process(rank, world_size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)



def main() -> None:
    # init_process(1, 2)

    worker = _create_worker(local_rank=1,rank=1)
    # result_handler.start()
    worker.init_device()
    worker.load_model()

    worker.delete_shard(1)
    worker.delete_shard(2)
    worker.delete_shard(3)

    state_dict = worker.model_runner.model.named_parameters()
    receive_sharded_weights_iterator(0, state_dict) 


    num_gpu_blocks, num_cpu_blocks = worker.determine_num_available_blocks()
    print(f"number of gpu blocks: {num_gpu_blocks}, number of cpu blocks: {num_cpu_blocks}")

    worker.initialize_cache(
         10000, 1000,
    )

    worker.execute_model()

    


if __name__ == '__main__':
    main()