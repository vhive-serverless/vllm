
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple, Dict

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
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.liquid_models import LIQUIDCONFIG, LiquidConfigType, Slice

KVCache = Tuple[torch.Tensor, torch.Tensor]

class OPTLiquid(OPTForCausalLM):
    def __init__(self, config, linear_method: Optional[LinearMethodBase] = None):
        super().__init__(config, linear_method)
        self.liquid_config : LiquidConfigType = LIQUIDCONFIG
        self.liquid_to : bool = False

    def liquid(self) -> None:
        # send layers to devices based on self.liquid_config
        for layers_range in self.liquid_config:
            device = self.liquid_config[layers_range]
            layers = self.model.decoder.layers[layers_range[0]: layers_range[1]]
            # self.model.decoder.layers[layers_range[0]:layers_range[1]] = self.model.decoder.layers[layers_range[0]:layers_range[1]].to(device)
            layers = layers.to(device)
            print(f"Send layers {layers_range[0]}-{layers_range[1]} to device: {device}")

        self.liquid_to = True

    def from_opt_model(self, model:OPTForCausalLM):
        self.model.load_state_dict(model.model.state_dict()) 

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, cache_group: Dict[Slice, List[Tuple[torch.Tensor]]], input_metadata: InputMetadata) -> torch.Tensor:
        # forward must be performed after liquid
        # input_ids = input_ids.to("cuda:0")
        # positions = positions.to("cuda:0")
        assert self.liquid_to
        decoder = self.model.decoder
        inputs_embeds = decoder.embed_tokens(input_ids)
        pos_embeds = decoder.embed_positions(positions)
        if decoder.project_in is not None:
            inputs_embeds, _ = decoder.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds


        # for each group of layers, perform forward
        for layers_range in self.liquid_config:
            
            kv_caches:List[Tuple[torch.Tensor]] = cache_group[layers_range]
            device = self.liquid_config[layers_range]
            input_metadata.to(device)
            layers = decoder.layers[layers_range[0]:layers_range[1]]

            assert layers[0].fc1.weight.device == device
            if kv_caches[0][0] is not None:
                assert kv_caches[0][0].device == device
                print(f"kv_caches is on device: {device}, kv_caches.shape:{kv_caches[0][0].shape}")

            hidden_states = hidden_states.to(device)

            torch.cuda.set_stream(torch.cuda.Stream(device=hidden_states.device))
            torch.cuda.empty_cache()

            for i in range(len(layers)):
                print(f"---------Entering layer {i+layers_range[0]}, device: {device}---------")
                if kv_caches[i][0] is None:
                    continue
                layer = layers[i]
                hidden_states = layer(hidden_states, kv_caches[i], input_metadata)
            torch.cuda.synchronize()

        # after layers forwarding, send the hidden states back to cuda:0
        hidden_states = hidden_states.to("cuda:0")


        if decoder.final_layer_norm is not None:
            hidden_states = decoder.final_layer_norm(hidden_states)
        if decoder.project_out is not None:
            hidden_states, _ = decoder.project_out(hidden_states)
        return hidden_states

        
