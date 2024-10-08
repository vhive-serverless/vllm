# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
#                                                QKVParallelLinear,
#                                                RowParallelLinear)
from vllm.liquid.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
#TODO
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE)
from vllm.liquid.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)

from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip, print_warning_once
from vllm.config import LiquidConfig
from vllm.liquid.sharded_parameter import ShardedParameter, QKVShardedParameter, GateUpShardedParameter
from vllm.liquid.utils import get_cuda_mem_info


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,)
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config,
                                           shard_ids= shard_ids,
                                           total_num_shards=total_num_shards)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ) -> None:
        super().__init__()
        self._shard_ids = shard_ids
        self.total_num_shards = total_num_shards

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // self.total_num_heads
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.scaling = self.head_dim**-0.5
        self.update_param()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              shard_ids=shard_ids,
                              total_num_shards=total_num_shards)
    
    def update_param(self):
        num_shards = len(self._shard_ids)
        self.num_heads = self.total_num_heads * num_shards // self.total_num_shards
        self.num_kv_heads = max(1, self.total_num_kv_heads * num_shards // self.total_num_shards)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        shard_ids: List[int] = [0],
        total_num_shards: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config=config,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              shard_ids=shard_ids,
                              total_num_shards=total_num_shards)
            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        liquid_config: Optional[LiquidConfig] = None,
        shard_ids: List[int] = [0],
    ) -> None:
        super().__init__()
        total_num_shards = 1 if liquid_config is None else liquid_config.liquid_total_num_shards
        self.config = config
        self.shard_ids = shard_ids
        self.liquid_config = liquid_config
        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                shard_ids=shard_ids,
                                total_num_shards=total_num_shards)

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            shard_ids=shard_ids,
            total_num_shards=total_num_shards,
        )
        # seems not typing, let's just ignore it
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
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
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def check_weights(self):
        for name, weight in self.model.named_parameters():
            print(f"Name: {name}, Weight: {weight}")
            print(f"\n\n")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
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
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        print_warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        raise NotImplementedError
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")

    def named_sharded_parameters(self):
        for name, param in self.named_parameters():
            if hasattr(param, "shard_ids"):
                yield name, param

    def sorted_named_parameters(self, descending: bool = True):
        # sort the parameters first to avoid a memory fragmentation
        # Get the named parameters of the model
        named_params = list(self.named_parameters())
        
        # Sort the named parameters based on the number of elements (numel()) in descending order
        sorted_named_params = sorted(named_params, key=lambda x: x[1].numel(), reverse=descending)
        
        return sorted_named_params
    


    def get_shards_weights(self, shard_ids: List[int], only_sharded: bool = True) -> Dict[str, torch.Tensor]:
        results = {}
        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1] + 1

        for name, param in self.sorted_named_parameters():
            if isinstance(param, QKVShardedParameter):
                q_shard, k_shard, v_shard = param.get_shards(start_shard_id, end_shard_id)
                results[f"{name}_q"] = q_shard
                results[f"{name}_k"] = k_shard
                results[f"{name}_v"] = v_shard
            elif isinstance(param, GateUpShardedParameter):
                gate_shard, up_shard = param.get_shards(start_shard_id, end_shard_id)
                results[f"{name}_gate"] = gate_shard
                results[f"{name}_up"] = up_shard
            elif isinstance(param, ShardedParameter):
                results[name] = param.get_shards(start_shard_id, end_shard_id)
            else:
                if not only_sharded:
                    results[name] = param
        # sort the results to reduce memory fragmentation
        return results

    def delete_shards(self, shard_ids: List[int]) -> None:

        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1] + 1

        shard_dim = self.lm_head.weight.shard_dim
        lm_head_first_half, lm_head_last_half = self.lm_head.weight.chunk(2, shard_dim)
        embed_token_first_half, _ = self.model.embed_tokens.weight.chunk(2, shard_dim)
        lm_head_last_half.copy_(embed_token_first_half)
        del embed_token_first_half
        self.lm_head.weight.data = lm_head_first_half
        self.model.embed_tokens.weight.data = lm_head_last_half 

        self.lm_head.weight.delete_shard_indexs(start_shard_id, end_shard_id)
        self.model.embed_tokens.weight.delete_shard_indexs(start_shard_id, end_shard_id)
        
        # print(f"Before deleting shards, {get_cuda_mem_info()}")
        # torch.cuda.memory._record_memory_history(max_entries=100000, context="all")
        for name, param in self.sorted_named_parameters(True):
            if hasattr(param, "num_shards"):
                if name in ['lm_head.weight', 'model.embed_tokens.weight']:
                    continue
                param.delete_shards(start_shard_id, end_shard_id)
                # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # print(f"After deleting shards, {get_cuda_mem_info()}")
        # torch.cuda.memory._dump_snapshot(f"./torch_mem_dump.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)

        for layer in self.model.layers:
            for shard_id in range(start_shard_id, end_shard_id):
                layer.self_attn.attn.delete_shard(shard_id)
            layer.self_attn.update_param()

        for shard_id in range(start_shard_id, end_shard_id):
            index = self.shard_ids.index(shard_id)
            self.shard_ids.pop(index)
        self.model.embed_tokens.update_sharded_indices(shard_ids=self.shard_ids, total_num_shards=self.total_num_shards)
        for layer in self.model.layers:
            layer.self_attn.update_param()
                

    def load_shards_weights(self, shard_ids: List[int], shards_weights: Dict[str, torch.Tensor]):
        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1] + 1
        shard_id = shard_ids[0]
        assert shard_id in self.shard_ids, f"{shard_id} not in the model"
        for name, param in self.sorted_named_parameters():
            if isinstance(param, QKVShardedParameter):
                q_shard = shards_weights[f"{name}_q"]
                k_shard = shards_weights[f"{name}_k"]
                v_shard = shards_weights[f"{name}_v"]
                # print(param.requires_grad)
                # q_data, k_data, v_data = param.chunk(3, dim=param.shard_dim)
                q_data, k_data, v_data = param.customize_chunk(param.data)
                q_data.copy_(q_shard)
                k_data.copy_(k_shard)
                v_data.copy_(v_shard)
            elif isinstance(param, GateUpShardedParameter):
                gate_shard = shards_weights[f"{name}_gate"]
                up_shard = shards_weights[f"{name}_up"]
                gate_data, up_data = param.chunk(2, dim=param.shard_dim)
                gate_data.copy_(gate_shard)
                up_data.copy_(up_shard)
            else:
                param.data.copy_(shards_weights[name])
            # if name in shards_weights.keys():
            #     param.data.copy_(shards_weights[name])
        self.model.embed_tokens.update_sharded_indices(shard_ids=self.shard_ids, total_num_shards=self.total_num_shards)
        # self.shard_ids.append(shard_id)
        for layer in self.model.layers:
            layer.self_attn.update_param()

    def append_shards_weights(self, shard_ids: List[int], shards_weights: Dict[str, torch.Tensor]):

        if len(shard_ids) == 1:
            start_shard_id = shard_ids[0]
            end_shard_id = start_shard_id+1
        else:
            start_shard_id = shard_ids[0]
            end_shard_id = shard_ids[-1]+1
        # print(f"Before entering for loop, {get_cuda_mem_info()}")
        # torch.cuda.memory._record_memory_history(max_entries=100000, context="all")
        with torch.no_grad():
            for name, param in self.sorted_named_parameters(False):
                if isinstance(param, QKVShardedParameter):
                    q_shard = shards_weights[f"{name}_q"]
                    k_shard = shards_weights[f"{name}_k"]
                    v_shard = shards_weights[f"{name}_v"]
                    param.append_shards(start_shard_id, end_shard_id ,q_shard, k_shard, v_shard)
                    # param.extend_and_load_shard(q_shard, k_shard, v_shard)
                    del q_shard, k_shard, v_shard
                    del shards_weights[f"{name}_q"], shards_weights[f"{name}_k"], shards_weights[f"{name}_v"]
                    # torch.cuda.empty_cache()
                elif isinstance(param, GateUpShardedParameter):
                    gate_shard = shards_weights[f"{name}_gate"]
                    up_shard = shards_weights[f"{name}_up"]
                    param.append_shards(start_shard_id, end_shard_id ,gate_shard, up_shard)
                    # param.extend_and_load_shard(gate_shard, up_shard)
                    del gate_shard, up_shard
                    del shards_weights[f"{name}_gate"], shards_weights[f"{name}_up"]
                    # torch.cuda.empty_cache()
                elif isinstance(param, ShardedParameter):
                    param.append_shards(start_shard_id, end_shard_id ,shards_weights[name])
                    # param.extend_and_load_shard(shard_data=shards_weights[name])
                    del shards_weights[name]
        torch.cuda.empty_cache()
        # print(f"After exit for loop, {get_cuda_mem_info()}")
        # torch.cuda.memory._dump_snapshot(f"./torch_mem_dump.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
        
        for layer in self.model.layers:
            for shard_id in range(start_shard_id, end_shard_id):
                layer.self_attn.attn.append_shard(shard_id)

        for shard_id in range(start_shard_id, end_shard_id):
            self.shard_ids.append(shard_id)
        self.model.embed_tokens.update_sharded_indices(shard_ids=self.shard_ids, total_num_shards=self.total_num_shards)
        for layer in self.model.layers:
            layer.self_attn.update_param()
 
