"""A GPU worker class."""
import gc
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig, LiquidConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce,
                              update_active_ranks,
                              )
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tensor_model_parallel_cpu_group,
                                             )
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest, PoolerOutput, SamplerOutput
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker_base import WorkerBase
from vllm.liquid.utils import send_dict, receive_dict
import time
from vllm.logger import logger
from vllm.liquid.utils import get_cuda_mem_info
import vllm.liquid.liquid_state as liquid_state


class Worker(WorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        liquid_config: Optional[LiquidConfig],
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.liquid_config = liquid_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.liquid_config is not None:
            self.active_ranks = [0]
        # if self.is_driver_worker:
        #     assert self.rank == 0, "The driver worker must have rank 0."
        liquid_state.LIQUID_CONFIG = liquid_config
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")
        self.driver_rank = self.liquid_config.liquid_driver_gpu_id

        ModelRunnerClass = (EmbeddingModelRunner if
                            self.model_config.embedding_mode else ModelRunner)
        self.model_runner = ModelRunnerClass(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config,
            liquid_config=self.liquid_config,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[torch.tensor]] = None

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            self.liquid_config,
                                            self.model_config.dtype
                                            )
        # Set random seed.
        set_random_seed(self.model_config.seed)


    def update_active_ranks(self, active_ranks: List[int]):
        self.active_ranks = active_ranks
        update_active_ranks(self.active_ranks)

    def load_model(self):
        self.model_runner.load_model()

    def liquid_model_weights(self, shard_ids: List[int], src: int, dst: int, only_send_sharded_weights: bool = False):
        assert self.rank == src or self.rank == dst
        if self.rank == src:
            start = time.time()
            bytes_sent = self.model_runner.send_shards(shard_ids, dst, only_sharded=only_send_sharded_weights)
            send_latency = time.time() - start
            sent_bandwidth = bytes_sent / ((1024**3) * send_latency) 
            logger.info(f"send weights shards takes: {send_latency:.2f}s, sent out: {bytes_sent/(1024**3):.2f}GB, sent bw: {sent_bandwidth:.2f}GB/s")
        else:
            if not hasattr(self.model_runner, "model"):
                start = time.time()
                self.model_runner.initialize_sharded_model(shard_ids)
                print(f"It takes: {time.time() - start:.2f}s to init model weights")
                start = time.time()
                shards_weights = self.model_runner.recv_shards(shard_ids, src, only_sharded=False)
                print(f"It takes: {time.time() - start:.2f}s to recv shards")
                start = time.time()
                self.model_runner.model.load_shards_weights(shard_ids,shards_weights)
                print(f"It takes {time.time() - start:.2f} to load shards")
            else:
                logger.info(f"Before appending weights shards, {get_cuda_mem_info(self.rank)}")
                shards_weights = self.model_runner.recv_shards(shard_ids, src, only_sharded=True)
                torch.cuda.empty_cache()
                logger.info(f"After recving weights shards, {get_cuda_mem_info(self.rank)}")               
                self.model_runner.model.append_shards_weights(shard_ids, 
                    shards_weights = shards_weights)
                logger.info(f"After appending weights shards, {get_cuda_mem_info(self.rank)}")
            for name, weight in shards_weights.items():
                del weight
            del shards_weights

    def init_cache(self, num_gpu_blocks: int, shard_ids):
        if not hasattr(self, "cache_engine"):
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            self.cache_config.num_cpu_blocks = 1
            self._init_cache_engine(shard_ids)

    def determine_num_new_gpu_blocks(self) -> int:
        torch.cuda.set_device(f"cuda:{self.rank}")
        torch.cuda.empty_cache()
        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        left_empty_gpu_memory = total_gpu_memory * (1 - self.cache_config.gpu_memory_utilization)

        increased_gpu_memory = free_gpu_memory - left_empty_gpu_memory

        cache_block_size = self.get_gpu_block_size_bytes()
        
        num_gpu_blocks = int(increased_gpu_memory // cache_block_size)
        return num_gpu_blocks

    def extend_gpu_blocks(self, num_gpu_blocks: int):
        start = time.time()
        self.cache_engine.extend_gpu_blocks(num_gpu_blocks)
        extend_latency = time.time() - start
        logger.info(f"extend gpu in worker takes: {extend_latency:.2f}s")
        self.cache_config.num_gpu_blocks = num_gpu_blocks

    def move_and_shrink_gpu_blocks(self, src_to_dsts: List[Tuple[int,int]], num_gpu_blocks: int):
        self.cache_engine.move_gpu_blocks(src_to_dsts)
        self.cache_engine.shrink_gpu_blocks(src_to_dsts, num_gpu_blocks)


    def liquid_kv_cache(self, shard_ids: List[int], src: int, dst: int, load_kv_cache):
        assert self.rank == src or self.rank == dst
        if self.rank == src:
            bytes_sent = self.cache_engine.send_shards(shard_ids, dst)
        else:
            logger.info(f"Before recv kvc shards, {get_cuda_mem_info(self.rank)}")
            shards_cache = self.cache_engine.recv_shards(shard_ids, src)
            logger.info(f"After recv kvc shards, {get_cuda_mem_info(self.rank)}")
            if load_kv_cache:
                
                self.cache_engine.load_shards(shard_ids,shards_cache)
            else:
                self.cache_engine.append_shards(shard_ids, shards_cache)
            logger.info(f"After appending kvc shards, {get_cuda_mem_info(self.rank)}")
            for name, weight in shards_cache.items():
                del weight
            del shards_cache
            torch.cuda.empty_cache()
            logger.info(f"After cleaning received kvc shards, {get_cuda_mem_info(self.rank)}")


    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )
            
    def save_serverless_llm_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_serverless_llm_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )



    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes()
        num_gpu_blocks = int(
            (total_gpu_memory * self.cache_config.gpu_memory_utilization -
             peak_memory) // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        # num_cpu_blocks = max(num_cpu_blocks, 0)
        num_cpu_blocks = 1
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self, shard_ids: Optional[List[int]]=None):
        if self.liquid_config is None:
            shard_ids = [0]
            total_num_shards = 1
        else:
            total_num_shards = self.liquid_config.liquid_total_num_shards
            shard_ids = shard_ids if shard_ids is not None else list(range(total_num_shards))

        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config, shard_ids=shard_ids, total_num_shards=total_num_shards)
        self.gpu_cache = self.cache_engine.gpu_cache

    def _warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: torch.Tensor,
        blocks_to_swap_out: torch.Tensor,
        blocks_to_copy: torch.Tensor,
    ) -> None:
        # Issue cache operations.
        if blocks_to_swap_in.numel() > 0:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out.numel() > 0:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy.numel() > 0:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        if not self.is_driver_worker:
            self._execute_model_non_driver()
            return []

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            group = get_tensor_model_parallel_group()
            metadata_group = get_tensor_model_parallel_cpu_group()
            broadcast_tensor_dict({}, src=self.driver_rank, group=group, metadata_group=metadata_group)
            return []

        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        num_seq_groups = len(seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)
        data: Dict[str, Any] = {
            "num_seq_groups": num_seq_groups,
            "blocks_to_swap_in": blocks_to_swap_in,
            "blocks_to_swap_out": blocks_to_swap_out,
            "blocks_to_copy": blocks_to_copy,
        }
        group = get_tensor_model_parallel_group()
        metadata_group = get_tensor_model_parallel_cpu_group()

        broadcast_tensor_dict(data, src=self.driver_rank,group=group, metadata_group=metadata_group)

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)

        # Worker only supports single-step execution. Wrap the output in a list
        # to conform to interface.
        return [output]

    def is_active(self) -> bool:
        return self.rank in self.active_ranks

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        while self._execute_model_non_driver():
            pass

    def _execute_model_non_driver(self) -> bool:
        """Execute model in parallel worker.

        Returns True iff there are remaining sequences to process.
        """
        assert not self.is_driver_worker
        group = get_tensor_model_parallel_group()
        metadata_group = get_tensor_model_parallel_cpu_group()
        data = broadcast_tensor_dict(src=self.driver_rank, group=group, metadata_group=metadata_group)
        if not data:
            return False

        num_seq_groups = data.get("num_seq_groups", 0)
        blocks_to_swap_in = data.get("blocks_to_swap_in")
        blocks_to_swap_out = data.get("blocks_to_swap_out")
        blocks_to_copy = data.get("blocks_to_copy")
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return False

        self.model_runner.execute_model(None, self.gpu_cache)
        return True

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_gpu_block_size_bytes(self) -> int:
        return self.cache_engine.get_gpu_block_size(self.cache_config, self.model_config, self.parallel_config)

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config,
                                                )
    def delete_kv_cache(self):
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info()
        logger.info(f"Before delete all kv_cache, we have {free_mem/(1024**3):.1f}GB space on GPU0")

        last_free_mem = free_mem
        for i in range(self.cache_engine.num_layers):
            self.cache_engine.gpu_cache[i].delete_shard(1)
            # self.cache_engine.gpu_cache[i].data = torch.empty(0)
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info()
            freed_mem = free_mem - last_free_mem
            last_free_mem = free_mem

    def get_shards_weights(self, shard_ids: List[int]):
        return self.model_runner.model.get_shards_weights(shard_ids)

    def delete_shards_weights(self, shard_ids: List[int]):
        self.model_runner.model.delete_shards(shard_ids)

    def append_shards_weights(self, shard_ids: List[int], shards_weights):
        self.model_runner.model.append_shards_weights(shard_ids, shards_weights)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    liquid_config: Optional[LiquidConfig] = None,
    dtype: torch.dtype = torch.float,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
    if liquid_config is None:
        world_size = parallel_config.tensor_parallel_size
    else:
        world_size = len(liquid_config.liquid_gpu_range)

    init_distributed_environment(world_size,rank,
                                 distributed_init_method, local_rank, dtype=dtype, driver_rank = liquid_config.liquid_driver_gpu_id)
    if liquid_config is None:
        ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                        parallel_config.pipeline_parallel_size,
                                        )
    

def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")

