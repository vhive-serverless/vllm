import random

import torch

from typing import Tuple

from vllm._C import cache_ops



def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device=device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)


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
    x = 8
    kv_caches = []

    for device in CUDA_DEVICES:
        key_cache = torch.empty(size=(NUM_BLOCKS, NUM_HEADS, HEAD_SIZES // x, BLOCK_SIZES, x ), dtype=DTYPES, device=device)
        value_cache = torch.empty(size=(NUM_BLOCKS, NUM_HEADS, HEAD_SIZES, BLOCK_SIZES), dtype=DTYPES, device=device)

        kv_caches.append((key_cache, value_cache))

    for i, device in enumerate(CUDA_DEVICES):
        key_cache, value_cache = kv_caches[i]
        test_reshape_and_cache(
            num_tokens = NUM_TOKENS,
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZES,
            block_size=BLOCK_SIZES,
            num_blocks=NUM_BLOCKS,
            dtype=DTYPES,
            seed=SEEDS,
            device=device,
            key_cache=key_cache,
            value_cache=value_cache
        )
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"GPU memory is broken, error:\n{e}")