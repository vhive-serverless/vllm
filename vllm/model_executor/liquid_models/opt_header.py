
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
from vllm.model_executor.models.opt import OPTLearnedPositionalEmbedding, OPTForCausalLM

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OPTHeader(nn.Module):

    def __init__(self, 
                 config: OPTConfig, 
                 linear_method: Optional[LinearMethodBase] = None,
                 ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        # Init embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        # init project in layer
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(config.word_embed_proj_dim,
                                               config.hidden_size,
                                               bias=False,
                                               linear_method=linear_method)
        else:
            self.project_in = None


    def from_whole_model(self, model:OPTForCausalLM):
        # init the model weights from the whole model
        self.embed_tokens.load_state_dict(model.model.decoder.embed_tokens.state_dict()) 
        self.embed_positions.load_state_dict(model.model.decoder.embed_positions.state_dict())
        if self.project_in != None:
            self.project_in.load_state_dict(model.model.decoder.project_in.state_dict())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds, _ = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        return hidden_states