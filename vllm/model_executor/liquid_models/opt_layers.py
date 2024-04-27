
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import OPTConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.model_executor.models.opt import OPTLearnedPositionalEmbedding, OPTForCausalLM, OPTDecoderLayer

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OPTLayers(nn.Module):

    def __init__(self, 
                 config: OPTConfig, 
                 linear_method: Optional[LinearMethodBase] = None,
                 range_start:int = 0,
                 range_end:int = 0
                 ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        # Init decoder layers
        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, linear_method)
            for _ in range(range_end - range_start)
        ])

    
    def from_whole_model(self, model:OPTForCausalLM, range_start:int):
        origin_layers = model.model.decoder.layers
        # init the model weights from the whole model
        for i in range(len(self.layers)):
            self.layers[i].load_state_dict(origin_layers[i+range_start].state_dict())

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    )->torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata)

        return hidden_states