"""CacheEngine class for managing the KV cache."""
from typing import List, Dict

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available
from vllm.liquid.sharded_tensor import ShardedTensor
from vllm.liquid.utils import send_dict, receive_dict
from vllm.distributed.communication_op import get_tcp_store, get_device_world_group,get_tensor_model_parallel_group, get_tensor_model_parallel_cpu_group

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
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.num_kv_heads = (self.num_kv_heads // total_num_shards) * current_num_shards

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
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
        for _ in range(self.num_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            cache = ShardedTensor(torch.zeros(kv_cache_shape,
                        dtype=self.dtype,
                        pin_memory=pin_memory,
                        device=device).view(2, num_blocks, self.num_kv_heads, self.head_size, self.block_size), shard_ids=self.shard_ids,num_shards=len(self.shard_ids), shard_dim=2)
            kv_cache.append(cache)
        return kv_cache

    def send_shards(self, shard_ids: List[int], dst: int):
        shards_cache = self.get_shards(shard_ids)
        # print(f"sharded_weights.keys: {shards_weights.keys()}")
        store = get_tcp_store()
        group = get_device_world_group()
        send_dict(shards_cache, dst, store, group)
        self.delete_shards(shard_ids)
        logger.info(f"Successfully send kv cache shards: {shard_ids} to rank: {dst}")

    def recv_shards(self, shard_ids: List[int], src: int):
        store = get_tcp_store()
        group = get_device_world_group()
        tensor_names = [f"layer_{i}" for i in range(len(self.gpu_cache))]
        shards_cache = receive_dict(src, store, tensor_names, group)
        return shards_cache
        

    def get_shards(self, shard_ids: List[int]) -> Dict[str,torch.Tensor]:
        if len(shard_ids) > 1:
            raise NotImplementedError(f"get shard with length > 1 is not implemented yet")
        shard_id = shard_ids[0]
        results = {}
        for i, cache in enumerate(self.gpu_cache):
            results[f"layer_{i}"] = cache.get_shard(shard_id)

        return results

    def delete_shards(self, shard_ids: List[int]) -> None:
        if len(shard_ids) > 1:
            raise NotImplementedError(f"delete shard with length > 1 is not implemented yet")
        shard_id = shard_ids[0]
        for cache in self.gpu_cache:
            cache.delete_shard(shard_id) 
        # TODO: handle cpu cache
        # for cache in self.cpu_cache:
        #     cache.delete_shard(shard_id)

        index = self.shard_ids.index(shard_id)
        self.shard_ids.pop(index)

    def load_shards(self,shard_ids: List[int], shards_data: Dict[str, torch.Tensor]):
        if len(shard_ids) > 1:
            raise NotImplementedError(f"get shard with length > 1 is not implemented yet")
        shard_id = shard_ids[0]
        assert shard_id in self.shard_ids
        for i, cache in enumerate(self.gpu_cache):
            cache.copy_(shards_data[f"layer_{i}"])

    def append_shards(self,shard_ids: List[int], shards_data: Dict[str, torch.Tensor]):
        if len(shard_ids) > 1:
            raise NotImplementedError(f"get shard with length > 1 is not implemented yet")
        shard_id = shard_ids[0]
        assert shard_id not in self.shard_ids, f"shard {shard_id} already in cache tensors: {self.shard_ids}"
        for i, cache in enumerate(self.gpu_cache):
            cache.append_shard(shard_id, shards_data[f"layer_{i}"])

        self.shard_ids.append(shard_id)


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
