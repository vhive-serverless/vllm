# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OPT model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple, Dict, Iterator

import torch
from torch import nn
from transformers import OPTConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.liquid.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.liquid.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.config import LiquidConfig


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ) -> None:
        super().__init__()
        current_num_shards = len(shard_ids)
        self.embed_dim = embed_dim
        total_num_heads = num_heads
        assert num_heads % total_num_shards == 0
        self.num_heads = total_num_heads * current_num_shards // total_num_shards
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_num_heads,
            bias=bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
            
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              shard_ids=shard_ids,
                              total_num_shards=total_num_shards,
                              )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.activation_fn = get_act_fn(config.activation_function,
                                        quant_config, config.ffn_dim)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(config.hidden_size,
                                                config.word_embed_proj_dim,
                                                bias=False,
                                                quant_config=quant_config)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(config.word_embed_proj_dim,
                                               config.hidden_size,
                                               bias=False,
                                               quant_config=quant_config)
        else:
            self.project_in = None

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

        self.layers = nn.ModuleList([
            OPTDecoderLayer(config, cache_config, quant_config, shard_ids, total_num_shards)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds, _ = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], attn_metadata)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ):
        super().__init__()
        self.decoder = OPTDecoder(config, cache_config, quant_config, shard_ids, total_num_shards)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, kv_caches, attn_metadata)


class OPTForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        liquid_config: Optional[LiquidConfig] = None,
        shard_ids: List[int] = [0],
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.liquid_config = liquid_config
        self.shard_ids = shard_ids
        total_num_shards = 1 if liquid_config is None else liquid_config.liquid_total_num_shards
        self.model = OPTModel(config, cache_config, quant_config, shard_ids, total_num_shards=total_num_shards)
        self.lm_head_weight = self.model.decoder.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        if self.liquid_config is not None:
            self.total_num_shards = self.liquid_config.liquid_total_num_shards

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def named_sharded_parameters(self):
        for name, param in self.named_parameters():
            if hasattr(param, "shard_ids"):
                yield name, param


    def get_shards_weights(self, shard_ids: List[int], only_sharded: bool = True) -> Dict[str, torch.Tensor]:
        results = {}
        if len(shard_ids) > 1:
            raise NotImplementedError(f"get shards with length of {len(shard_ids)} is not implemented yet!")

        shard_id = shard_ids[0]
        assert shard_id in self.shard_ids, f"{shard_id} not in the model"
        for name, param in self.named_parameters():
            if hasattr(param, "shard_ids"):
                results[name] = param.get_shard(shard_id)
            else:
                if not only_sharded:
                    results[name] = param
        return results

    def delete_shards(self, shard_ids: List[int]) -> None:
        if len(shard_ids) > 1:
            raise NotImplementedError(f"delete shards with length of {len(shard_ids)} is not implemented yet!")

        shard_id = shard_ids[0]
        assert shard_id in self.shard_ids, f"{shard_id} not in the model"
        for name, param in self.named_parameters():
            if hasattr(param, "num_shards"):
                param.delete_shard(shard_id)
                torch.cuda.empty_cache()
        

        for layer in self.model.decoder.layers:
            layer.self_attn.attn.delete_shard(shard_id)

        
        index = self.shard_ids.index(shard_id)
        self.shard_ids.pop(index)
        self.model.decoder.embed_tokens.update_sharded_indices(shard_ids=self.shard_ids, total_num_shards=self.total_num_shards)
                

    def load_shards_weights(self, shard_ids: List[int], shards_weights: Dict[str, torch.Tensor]):
        if len(shard_ids) > 1:
            raise NotImplementedError(f"load shards with length of {len(shard_ids)} is not implemented yet!")
        shard_id = shard_ids[0]
        assert shard_id in self.shard_ids, f"{shard_id} not in the model"
        for name, param in self.named_parameters():
            if name in shards_weights.keys():
                param.data.copy_(shards_weights[name])

        self.model.decoder.embed_tokens.update_sharded_indices(shard_ids=self.shard_ids, total_num_shards=self.total_num_shards)

    def append_shards_weights(self, shard_ids: List[int], shards_weights: Dict[str, torch.Tensor]):
        if len(shard_ids) > 1:
            raise NotImplementedError(f"load shards with length of {len(shard_ids)} is not implemented yet!")
        shard_id = shard_ids[0]
        assert shard_id not in self.shard_ids, f"{shard_id} already in the model"
        for name, param in self.named_parameters():
            if name in shards_weights.keys():
                assert hasattr(param, "shard_ids")
                param.append_shard(shard_id, shards_weights[name])
                del shards_weights[name]
        
        for layer in self.model.decoder.layers:
            layer.self_attn.attn.append_shard(shard_id)

        self.shard_ids.append(shard_id)
        self.model.decoder.embed_tokens.update_sharded_indices(shard_ids=self.shard_ids, total_num_shards=self.total_num_shards)


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                continue
            if name.startswith("decoder."):
                name = "model." + name

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
