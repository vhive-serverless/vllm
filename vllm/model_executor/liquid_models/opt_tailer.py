
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
from vllm.model_executor.layers.sampler import Sampler

KVCache = Tuple[torch.Tensor, torch.Tensor]


class OPTTailer(nn.Module):

    def __init__(self, 
                 config: OPTConfig, 
                 linear_method: Optional[LinearMethodBase] = None,
                 lm_head_weight: torch.Tensor = None,
                 ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.lm_head_weight = lm_head_weight
        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(config.hidden_size,
                                                config.word_embed_proj_dim,
                                                bias=False,
                                                linear_method=linear_method)
        else:
            self.project_out = None
        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine)
        else:
            self.final_layer_norm = None
        
        self.sampler = Sampler(config.vocab_size)


    def from_whole_model(self, model:OPTForCausalLM):
        # init the model weights from the whole model
        if self.project_out is not None:
            self.project_out(model.model.decoder.project_out.state_dict()) 
        
        if self.final_layer_norm is not None:
            self.final_layer_norm.load_state_dict(model.model.decoder.final_layer_norm.state_dict())

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        return hidden_states