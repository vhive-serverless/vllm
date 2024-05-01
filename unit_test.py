import random

import torch

from typing import Tuple

from vllm._C import cache_ops
from vllm.utils import is_hip
from vllm.utils import create_kv_caches_with_random



def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device="cuda:0")
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)
    print(f"on device: {key.device}, data_ptr: {key.data_ptr()}")
    key = key.to(device)
    print(f"on device: {key.device}, data_ptr: {key.data_ptr()}")
    value = value.to(device)
    slot_mapping = slot_mapping.to(device)
    print(f"key.device: {key.device}")

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                None, seed, device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    print(key.device, value.device, key_cache.device, value_cache.device, slot_mapping.device)
    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping, "auto")


if __name__ == '__main__':
    NUM_TOKENS = 6  # Arbitrary values for testing
    NUM_LAYERS = 1  # Arbitrary values for testing
    NUM_HEADS = 8  # Arbitrary values for testing
    HEAD_SIZES = 64
    BLOCK_SIZES = 8
    # reduce the size for ROCm test to avoid HIP OOM
    NUM_BLOCKS = 10240
    NUM_MAPPINGS = 256  # Arbitrary values for testing
    SEEDS = 0
    CUDA_DEVICES = [
        f"cuda:{i}" for i in range(2)
    ]
    DTYPES = torch.float
    for device in CUDA_DEVICES:
        for i in range(6):
            test_reshape_and_cache(
                num_tokens = NUM_TOKENS,
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZES,
                block_size=BLOCK_SIZES,
                num_blocks=NUM_BLOCKS,
                dtype=DTYPES,
                seed=SEEDS,
                device=device,
            )
            torch.cuda.synchronize()