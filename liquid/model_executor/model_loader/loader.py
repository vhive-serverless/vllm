# ruff: noqa: SIM117
import collections
import copy
import fnmatch
import glob
import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoadFormat,
                         LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, is_vllm_tensorized, load_with_tensorizer,
    tensorizer_weights_iterator)
from vllm.model_executor.model_loader.utils import (get_model_architecture,
                                                    set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    get_quant_config, initialize_dummy_weights, np_cache_weights_iterator,
    pt_weights_iterator, safetensors_weights_iterator)
from vllm.model_executor.models.vlm_base import VisionLanguageModelBase
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.model_loader.loader import BaseModelLoader, _get_model_initialization_kwargs, _get_quantization_config
import importlib

logger = init_logger(__name__)


def _initialize_model(model_config: ModelConfig, load_config: LoadConfig,
                      lora_config: Optional[LoRAConfig],
                      vision_language_config: Optional[VisionLanguageConfig],
                      cache_config: CacheConfig) -> nn.Module:
    """Initialize a model with the given configurations."""
    module = importlib.import_module("liquid.model_executor.models.opt")
    model_class = getattr(module, "OPTForCausalLM")
    quant_config = _get_quantization_config(model_config, load_config)

    return model_class(config=model_config.hf_config,
                       cache_config=cache_config,
                       quant_config=quant_config,
                       **_get_model_initialization_kwargs(
                           model_class, lora_config, vision_language_config))


class LiquidShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/save_sharded_state.py` for creating a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = ({} if load_config.model_loader_extra_config is None
                        else load_config.model_loader_extra_config.copy())
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(f"Unexpected extra config keys for load format "
                             f"{load_config.load_format}: "
                             f"{load_config.model_loader_extra_config.keys()}")

    @staticmethod
    def _filter_subtensors(
            tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str]):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(model_name_or_path,
                                            self.load_config.download_dir,
                                            allow_patterns, revision)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        from safetensors.torch import safe_open


        local_model_path = self._prepare_weights(model_config.model,
                                                 model_config.revision)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config,
                                          cache_config)
            
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )
            filepaths = glob.glob(pattern)
            if not filepaths:
                # TODO: support un-sharded checkpoints too
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!")
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s", tensor.shape,
                                key, param_shape)
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(
                    f"Missing keys {tuple(state_dict)} in loaded state!")
        return model.eval()

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        shard_idx: int,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from safetensors.torch import save_file

        from vllm.distributed import get_tensor_model_parallel_rank
        if pattern is None:
            pattern = LiquidShardedStateLoader.DEFAULT_PATTERN
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        part_idx = 0
        total_size = 0
        state_dict = LiquidShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )
