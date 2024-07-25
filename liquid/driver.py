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
from typing import Optional, List, Dict, Any
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

    engine_config.parallel_config.world_size = 2
    engine_config.parallel_config.tensor_parallel_size = 2

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

def init_process(rank, world_size, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def send_sharded_weights_iterator(sharded_weights_iterator, dst_rank):
    tensor_bytes = 0
    for name, tensor in sharded_weights_iterator:
        dist.send(tensor, dst=dst_rank)
        tensor_bytes += tensor.element_size() * tensor.numel()
    return tensor_bytes / (1024**3)




def main() -> None:
    # init_process(0, 2)

    driver_worker = _create_worker(local_rank=0,rank=0)
    # result_handler.start()
    driver_worker.init_device()
    driver_worker.load_model()

    # num_gpu_blocks, num_cpu_blocks = driver_worker.determine_num_available_blocks()
    

    # driver_worker.initialize_cache(
    #      num_gpu_blocks, num_cpu_blocks,
    # )
    mem_usage = get_gpu_memory_usage()
    print(f"After initialize kv_cache, memory usage: {mem_usage:.1f} GB")

    start = time.time()
    sharded_weights_iterator = driver_worker.get_sharded_weights_iterator(1) 
    latency = time.time() - start
    print(f"It takes: {latency:.1f} s to get sharded_weights")

    start = time.time()
    tensor_size_GB = send_sharded_weights_iterator(sharded_weights_iterator, 1)
    latency = time.time() - start
    bandwidth = tensor_size_GB / latency
    print(f"It takes: {latency:.1f} to send weights iterator for {tensor_size_GB:.2f} GB, bandwidth: {bandwidth:.2f} GB/s")


    start = time.time()
    driver_worker.delete_shard(1)
    delete_latency = time.time() - start
    mem_usage = get_gpu_memory_usage()
    print(f"It takes: {delete_latency:.1f} s to delete all tensors")
    print(f"After delete one shard, memory usage: {mem_usage:.1f} GB")

    num_gpu_blocks, num_cpu_blocks = driver_worker.determine_num_available_blocks()
    print(f"number of gpu blocks: {num_gpu_blocks}, number of cpu blocks: {num_cpu_blocks}")

    driver_worker.initialize_cache(
         10000, 1000,
    )
    prompt_token_ids = [6,1,9]
    sampling_params = SamplingParams(temperature=0, min_tokens=3, max_tokens=4)
    block_tables = {0:[0]}
    seq_data = SequenceData(prompt_token_ids=prompt_token_ids)
    seq_group_metadata = SequenceGroupMetadata(
         request_id="0",
         is_prompt=True,
         seq_data={0:seq_data},
         sampling_params=sampling_params,
         block_tables=block_tables
         
    )
    seq_group_metadata_list = [seq_group_metadata]

    execute_model_request = ExecuteModelRequest(
         seq_group_metadata_list=seq_group_metadata_list,
         blocks_to_swap_in=[],
         blocks_to_swap_out=[],
         blocks_to_copy=[],
    )

    sampler_outputs = driver_worker.execute_model(execute_model_req=execute_model_request)

    output_token_id = sampler_outputs[0].outputs[0].samples[0].output_token
    print(f"output_token_id: {output_token_id}")

if __name__ == '__main__':
    main()