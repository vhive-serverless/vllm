import enum
from functools import lru_cache
from typing import Optional, Type

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_cpu, is_hip

logger = init_logger(__name__)


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()


@lru_cache(maxsize=None)
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_blocksparse: bool = False,
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    from liquid.attention.backends.liquid_attn import LiquidBackend 
    return LiquidBackend
