"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vllm.liquid.sharded_tensor import ShardedTensor
from vllm.liquid.utils import send_dict, receive_dict
from vllm.distributed.communication_op import get_liquid_communicator, get_device_world_group,get_tensor_model_parallel_group, get_tensor_model_parallel_cpu_group, get_tensor_model_parallel_rank
import time

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.shard_ids = shard_ids
        self.total_num_shards = total_num_shards
        current_num_shards  = len(self.shard_ids)

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.num_kv_heads = (self.total_num_kv_heads // total_num_shards) * current_num_shards

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        is_liquid = (total_num_shards!=1)
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
            is_liquid=is_liquid,
        )

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        # torch.cuda.memory._record_memory_history()
        for _ in range(self.num_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            if self.total_num_shards > 1:
                shard_dim = self.attn_backend.get_shard_dim()
                cache = ShardedTensor(torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device), shard_ids=self.shard_ids,num_shards=len(self.shard_ids), shard_dim=shard_dim)
            else:
                cache = torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device)
            kv_cache.append(cache)
        # torch.cuda.memory._dump_snapshot(f"./torch_mem_dump_{get_tensor_model_parallel_rank()}_{device}.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
        return kv_cache

    def extend_gpu_blocks(self, num_gpu_blocks: int):
        assert num_gpu_blocks > self.num_gpu_blocks
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size)
        latencys = []
        # torch.cuda.memory._record_memory_history()
        # first delete all original tensor

        for i in range(self.num_layers):
            start = time.time()
            new_cache = torch.empty(kv_cache_shape, dtype=self.dtype, device="cuda")
            original_num_blocks = self.gpu_cache[i].size(1)
            new_cache[:,:original_num_blocks, ...].copy_(self.gpu_cache[i])
            self.gpu_cache[i].data = new_cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            latency = time.time() - start
            latencys.append(latency)
        # torch.cuda.memory._dump_snapshot(f"./extend_gpu_memory_{get_tensor_model_parallel_rank()}.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
            


        self.num_gpu_blocks = num_gpu_blocks

    def move_gpu_blocks(self, src_to_dsts: List[Tuple[int,int]]):
        if src_to_dsts == []: return 
        blocks_to_copy = torch.tensor(src_to_dsts,
                                      device="cuda",
                                      dtype=torch.int64).view(-1, 2)
        self.copy(blocks_to_copy)
        torch.cuda.synchronize()
        

    def shrink_gpu_blocks(self, src_to_dsts: List[Tuple[int,int]], num_gpu_blocks: int):
        assert num_gpu_blocks < self.num_gpu_blocks
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(num_gpu_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for i in range(self.num_layers):
            start = time.time()
            new_cache = torch.zeros(kv_cache_shape, dtype=self.dtype, pin_memory=False, device="cuda")
            allocate_latency = time.time() - start

            start_copied_block_number = num_gpu_blocks - len(src_to_dsts)
            new_cache[:,start_copied_block_number:, ...] = self.gpu_cache[i][:,start_copied_block_number:num_gpu_blocks, ...]
            self.gpu_cache[i].data = new_cache
            torch.cuda.empty_cache()
        self.num_gpu_blocks = num_gpu_blocks 
        

    def send_shards(self, shard_ids: List[int], dst: int) -> int:
        shards_cache = self.get_shards(shard_ids)
        # print(f"sharded_weights.keys: {shards_weights.keys()}")
        liquid_comm = get_liquid_communicator()
        start = time.time()
        bytes_sent = liquid_comm.send_dict(shards_cache, dst)
        send_latency = time.time() - start
        sent_bandwidth = bytes_sent / ((1024**3) * send_latency) 
        logger.info(f"send kvc shards takes: {send_latency:.2f}s, sent out: {bytes_sent/(1024**3):.2f}GB, sent bw: {sent_bandwidth:.2f}GB/s")
        for name, cache in shards_cache.items():
            del cache
        del shards_cache
        torch.cuda.empty_cache()
        self.delete_shards(shard_ids)
        logger.info(f"Successfully send kv cache shards: {shard_ids} to rank: {dst}")
        return bytes_sent

    def recv_shards(self, shard_ids: List[int], src: int):
        liquid_comm = get_liquid_communicator()
        tensor_names = [f"layer_{i}" for i in range(len(self.gpu_cache))]
        shards_cache = liquid_comm.recv_dict(src, tensor_names)
        return shards_cache
        

    def get_shards(self, shard_ids: List[int]) -> Dict[str,torch.Tensor]:
        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1]+1
        results = {}
        for i, cache in enumerate(self.gpu_cache):
            results[f"layer_{i}"] = cache.get_shards(start_shard_id, end_shard_id)

        return results

    def delete_shards(self, shard_ids: List[int]) -> None:
        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1]+1
        shard_id = shard_ids[0]
        torch.cuda.empty_cache()
        free_mem,_ = torch.cuda.mem_get_info()
        latest_free_mem = free_mem
        for i, cache in enumerate(self.gpu_cache):
            cache.delete_shards(start_shard_id, end_shard_id) 
            torch.cuda.empty_cache()

        for shard_id in range(start_shard_id, end_shard_id):
            index = self.shard_ids.index(shard_id)
            self.shard_ids.pop(index)

        total_num_shards = self.total_num_shards
        current_num_shards = len(self.shard_ids)
        self.num_kv_heads = (self.total_num_kv_heads // total_num_shards) * current_num_shards

    def load_shards(self,shard_ids: List[int], shards_data: Dict[str, torch.Tensor]):
        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1]+1

        for i, cache in enumerate(self.gpu_cache):
            data = shards_data.pop(f"layer_{i}")
            cache.copy_(data)
            del data

        total_num_shards = self.total_num_shards
        current_num_shards = len(self.shard_ids)
        self.num_kv_heads = (self.total_num_kv_heads // total_num_shards) * current_num_shards

    def append_shards(self,shard_ids: List[int], shards_data: Dict[str, torch.Tensor]):
        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1]+1
        for i, cache in enumerate(self.gpu_cache):
            data = shards_data.pop(f"layer_{i}")
            cache.append_shards(start_shard_id, end_shard_id, data)
            del data
            # torch.cuda.empty_cache()

        for shard_id in range(start_shard_id, end_shard_id):
            self.shard_ids.append(shard_id)
        total_num_shards = self.total_num_shards
        current_num_shards = len(self.shard_ids)
        self.num_kv_heads = (self.total_num_kv_heads // total_num_shards) * current_num_shards


    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    def get_gpu_block_size(
            self,
            cache_config: CacheConfig,
            model_config: ModelConfig,
            parallel_config: ParallelConfig,

    ) -> int:
        current_num_shards = len(self.shard_ids)
        head_size = model_config.get_head_size()
        num_heads = int(model_config.get_num_kv_heads(parallel_config) * (current_num_shards / self.total_num_shards))
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
