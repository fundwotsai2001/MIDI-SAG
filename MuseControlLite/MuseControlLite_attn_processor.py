# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# For zero initialized 1D CNN in the attention processor
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
class StableAudioAttnProcessor2_0_echo(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.proj_gamma = zero_module(nn.Linear(1536, 1536, bias=False))
        self.proj_beta = zero_module(nn.Linear(1536, 1536, bias=False))
        self.hidden_proj = nn.Linear(1536, 1536, bias=False)
        self.con_proj = nn.Linear(768, 1536, bias=False)
        self.name = name
        self.proj_gamma.weight.requires_grad = True
        self.proj_beta.weight.requires_grad = True
        self.hidden_proj.weight.requires_grad = True
        self.con_proj.weight.requires_grad = True
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        hidden_states_original: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        ip_hidden_states = encoder_hidden_states_con
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        text_hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        text_hidden_states = text_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        text_hidden_states = text_hidden_states.to(query.dtype)
        # linear proj
        text_hidden_states = attn.to_out[0](text_hidden_states)
        # dropout
        text_hidden_states = attn.to_out[1](text_hidden_states)

        if input_ndim == 4:
            text_hidden_states = text_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            text_hidden_states = text_hidden_states + residual

        text_hidden_states = text_hidden_states / attn.rescale_output_factor
        new_hidden_states = hidden_states_original + text_hidden_states

        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions
        ###############################################################
        dtype = new_hidden_states.dtype

        ip_hs = ip_hidden_states.contiguous().to(dtype)
        base   = new_hidden_states  # [B, 1025, C]
        tail   = base[:, 1:, :].contiguous()                     # [B, 1024, C]
        # print("ip_hs", ip_hs.shape)
        c_prime = self.con_proj(ip_hs)                           # -> [B, 1024, C]
        h_prime = self.hidden_proj(tail)                         # -> [B, 1024, C]

        # out-of-place math only
        c = torch.tanh(h_prime) * torch.tanh(c_prime)            # [B, 1024, C]
        gamma = self.proj_gamma(c)                               # [B, 1024, C]
        beta  = self.proj_beta(c)                                # [B, 1024, C]
        # print("gamma", torch.sum(gamma))
        # print("beta", torch.sum(beta))
        # compute updated tail (no in-place)
        updated_tail = tail * gamma + beta           # [B, 1024, C]

        # rebuild the full sequence without writing into a view
        head = torch.zeros_like(base[:, :1, :])                       # [B, 1, C]
        new_hidden_states_FiLM = torch.cat([head, updated_tail], dim=1)  # [B, 1025, C]
        # print("new_hidden_states_FiLM",torch.sum(new_hidden_states_FiLM))
        new_hidden_states = new_hidden_states - hidden_states_original + new_hidden_states_FiLM
        ###############################################################
        return new_hidden_states
class StableAudioAttnProcessor2_0_rotary_free(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))     
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        audio_mid_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key_2 = attn.to_k(encoder_hidden_states_2)
        value_2 = attn.to_v(encoder_hidden_states_2)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        key_2 = key_2.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value_2 = value_2.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            key_2 = torch.repeat_interleave(key_2, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
            value_2 = torch.repeat_interleave(value_2, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if attn.norm_k is not None:
            key_2 = attn.norm_k(key_2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)                                                                                                    
        # TODO: add support for attn.scale when we move to Torch 2.1
        _scale = 2097152 / 44100                                                                                                                                       
        if isinstance(audio_mid_s, (list, tuple)):
            frames = [int(mid / _scale * 1024) for mid in audio_mid_s]
            if len(set(frames)) == 1:
                # All samples share the same split point — fully vectorized
                mid = frames[0]
                audio_duration_embeds = query[:, :, :1, :]
                hidden_states = F.scaled_dot_product_attention(
                    query[:, :, :mid, :], key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states_2 = F.scaled_dot_product_attention(
                    torch.cat((audio_duration_embeds, query[:, :, mid:, :]), dim=2),
                    key_2, value_2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = torch.cat((hidden_states, hidden_states_2[:, :, 1:, :]), dim=2)
            else:
                # Different split points per sample
                parts = []
                for b, mid in enumerate(frames):
                    q = query[b:b+1]
                    mask = attention_mask[b:b+1] if attention_mask is not None else None
                    hs = F.scaled_dot_product_attention(
                        q[:, :, :mid, :], key[b:b+1], value[b:b+1],
                        attn_mask=mask, dropout_p=0.0, is_causal=False
                    )
                    hs2 = F.scaled_dot_product_attention(
                        torch.cat((q[:, :, :1, :], q[:, :, mid:, :]), dim=2),
                        key_2[b:b+1], value_2[b:b+1],
                        attn_mask=mask, dropout_p=0.0, is_causal=False
                    )
                    parts.append(torch.cat((hs, hs2[:, :, 1:, :]), dim=2))
                hidden_states = torch.cat(parts, dim=0)
        else:
            mid = int(audio_mid_s / _scale * 1024)
            audio_duration_embeds = query[:, :, :1, :]
            hidden_states = F.scaled_dot_product_attention(
                query[:, :, :mid, :], key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_2 = F.scaled_dot_product_attention(
                torch.cat((audio_duration_embeds, query[:, :, mid:, :]), dim=2),
                key_2, value_2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = torch.cat((hidden_states, hidden_states_2[:, :, 1:, :]), dim=2)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions
        ###############################################################
        ip_hidden_states = encoder_hidden_states_con
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]

        # Generate position_ids for query, keys, values
        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
        position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        
        # Rotate query, keys, values 
        cos, sin = self.rotary_emb(query, position_ids_query)
        query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
        cos, sin = self.rotary_emb(ip_key, position_ids_key)
        ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
        cos, sin = self.rotary_emb(ip_value, position_ids_value)
        ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
        
        ip_hidden_states = F.scaled_dot_product_attention(
                query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ip_hidden_states = self.conv_out(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ###############################################################

        # Combine the output of the two cross-attention layers
        hidden_states = hidden_states + self.scale * ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StableAudioAttnProcessor2_0_free(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
    def apply_partial_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        rot_dim = freqs_cis[0].shape[-1]
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]

        x_rotated = apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        out = torch.cat((x_rotated, x_unrotated), dim=-1)
        return out

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        audio_mid_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key_2 = attn.to_k(encoder_hidden_states_2)
        value_2 = attn.to_v(encoder_hidden_states_2)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        key_2 = key_2.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value_2 = value_2.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            key_2 = torch.repeat_interleave(key_2, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
            value_2 = torch.repeat_interleave(value_2, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if attn.norm_k is not None:
            key_2 = attn.norm_k(key_2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)                                                                                                    
        # TODO: add support for attn.scale when we move to Torch 2.1
        _scale = 2097152 / 44100                                                                                                                                       
        if isinstance(audio_mid_s, (list, tuple)):
            frames = [int(mid / _scale * 1024) for mid in audio_mid_s]
            if len(set(frames)) == 1:
                # All samples share the same split point — fully vectorized
                mid = frames[0]
                audio_duration_embeds = query[:, :, :1, :]
                hidden_states = F.scaled_dot_product_attention(
                    query[:, :, :mid, :], key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states_2 = F.scaled_dot_product_attention(
                    torch.cat((audio_duration_embeds, query[:, :, mid:, :]), dim=2),
                    key_2, value_2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = torch.cat((hidden_states, hidden_states_2[:, :, 1:, :]), dim=2)
            else:
                # Different split points per sample
                parts = []
                for b, mid in enumerate(frames):
                    q = query[b:b+1]
                    mask = attention_mask[b:b+1] if attention_mask is not None else None
                    hs = F.scaled_dot_product_attention(
                        q[:, :, :mid, :], key[b:b+1], value[b:b+1],
                        attn_mask=mask, dropout_p=0.0, is_causal=False
                    )
                    hs2 = F.scaled_dot_product_attention(
                        torch.cat((q[:, :, :1, :], q[:, :, mid:, :]), dim=2),
                        key_2[b:b+1], value_2[b:b+1],
                        attn_mask=mask, dropout_p=0.0, is_causal=False
                    )
                    parts.append(torch.cat((hs, hs2[:, :, 1:, :]), dim=2))
                hidden_states = torch.cat(parts, dim=0)
        else:
            mid = int(audio_mid_s / _scale * 1024)
            audio_duration_embeds = query[:, :, :1, :]
            hidden_states = F.scaled_dot_product_attention(
                query[:, :, :mid, :], key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_2 = F.scaled_dot_product_attention(
                torch.cat((audio_duration_embeds, query[:, :, mid:, :]), dim=2),
                key_2, value_2, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = torch.cat((hidden_states, hidden_states_2[:, :, 1:, :]), dim=2)

        # No overlap

        # print("hidden_states", hidden_states.shape)
        # print("hidden_states_2", hidden_states_2.shape)
        # print("query_1", query_1.shape)
        # print("query_2", query_2.shape)
        # print("key_2", key_2.shape)
        # print("query", query.shape)
        # print("value_2", value_2.shape)
        # print("hidden_states", hidden_states.shape)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
# Original attention processor for 
class StableAudioAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
    def apply_partial_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        rot_dim = freqs_cis[0].shape[-1]
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]

        x_rotated = apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        out = torch.cat((x_rotated, x_unrotated), dim=-1)
        return out

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed 
        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(torch.float32)
            key = key.to(torch.float32)

            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = query[..., :rot_dim], query[..., rot_dim:]
            query_rotated = apply_rotary_emb(query_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

            query = torch.cat((query_rotated, query_unrotated), dim=-1)

            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = apply_rotary_emb(key_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

                key = torch.cat((key_rotated, key_unrotated), dim=-1)

            query = query.to(query_dtype)
            key = key.to(key_dtype)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # print("hidden_states", hidden_states.shape)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
# The attention processor used in MuseControlLite, using 1 decoupled cross-attention layer
class StableAudioAttnProcessor2_0_rotary(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))     
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        ip_hidden_states = encoder_hidden_states_con
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions
        ###############################################################
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]

        # Generate position_ids for query, keys, values
        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
        position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        
        # Rotate query, keys, values 
        cos, sin = self.rotary_emb(query, position_ids_query)
        query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
        cos, sin = self.rotary_emb(ip_key, position_ids_key)
        ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
        cos, sin = self.rotary_emb(ip_value, position_ids_value)
        ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
        
        ip_hidden_states = F.scaled_dot_product_attention(
                query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ip_hidden_states = self.conv_out(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ###############################################################

        # Combine the output of the two cross-attention layers
        hidden_states = hidden_states + self.scale * ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
class StableAudioAttnProcessor2_0_rotary_mm(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(768, 768, bias=False)
        self.to_v_ip = nn.Linear(768, 768, bias=False)
        self.norm = nn.LayerNorm(768, elementwise_affine=True, eps=1e-5)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))     
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.norm.weight.requires_grad = True
        self.norm.bias.requires_grad = True
        self.conv_out.weight.requires_grad = True
        self.Wqc = nn.Linear(768, 768, bias=False)
        self.Wki = nn.Linear(1536, 1536, bias=False)
        self.Wvi = nn.Linear(1536, 1536, bias=False)
        self.Oc = (nn.Linear(1536, 768, bias=False))
        self.Wqc.weight.requires_grad = True
        self.Wki.weight.requires_grad = True
        self.Wvi.weight.requires_grad = True
        self.Oc.weight.requires_grad = True
        if int(layer_id) == 23:
            self.Wqc.weight.requires_grad = False
            self.Wki.weight.requires_grad = False
            self.Wvi.weight.requires_grad = False
            self.Oc.weight.requires_grad = False
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio: QI.KP x VP
        ###############################################################
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_text_audio = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_text_audio = hidden_states_text_audio.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_text_audio = hidden_states_text_audio.to(query.dtype)
        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions: QI.KC x VC
        ###############################################################
        encoder_hidden_states_con = self.norm(encoder_hidden_states_con)
        ip_key = self.to_k_ip(encoder_hidden_states_con)
        ip_value = self.to_v_ip(encoder_hidden_states_con)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]

        # Generate position_ids for query, keys, values
        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_1025 = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_1024 = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        # Rotate query, keys, values 
        cos_q, sin_q = self.rotary_emb(query, position_ids_1025)
        query_pos = (query * cos_q.unsqueeze(1)) + (self.rotate_half(query) * sin_q.unsqueeze(1))

        cos_k, sin_k = self.rotary_emb(ip_key, position_ids_1024)
        ip_key = (ip_key * cos_k.unsqueeze(1)) + (self.rotate_half(ip_key) * sin_k.unsqueeze(1))
        ip_value = (ip_value * cos_k.unsqueeze(1)) + (self.rotate_half(ip_value) * sin_k.unsqueeze(1))

        ip_hidden_states = F.scaled_dot_product_attention(
                query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ip_hidden_states = self.conv_out(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ###############################################################


        # Conditions should see themselves! encoder_hidden_states = QC.KC x VC + QC.KI x VI
        query_c = self.Wqc(encoder_hidden_states_con)
        key_i = self.Wki(hidden_states)
        value_i = self.Wvi(hidden_states)
        query_c = query_c.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            query_c = torch.repeat_interleave(query_c, heads_per_kv_head, dim=1)
        query_c_pos = (query_c * cos_k.unsqueeze(1)) + (self.rotate_half(query_c) * cos_k.unsqueeze(1))
        key_i = (key_i * cos_q.unsqueeze(1)) + (self.rotate_half(key_i) * sin_q.unsqueeze(1))
        value_i = (value_i * cos_q.unsqueeze(1)) + (self.rotate_half(value_i) * sin_q.unsqueeze(1))

        condtion_self_hidden_state = F.scaled_dot_product_attention(
                query_c_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False)
        condtion_cross_hidden_state = F.scaled_dot_product_attention(
                query_c_pos, key_i, value_i, attn_mask=None, dropout_p=0.0, is_causal=False)
        condtion_self_hidden_state = condtion_self_hidden_state.transpose(1, 2).reshape(batch_size, -1, 24 * head_dim)
        condtion_cross_hidden_state = condtion_cross_hidden_state.transpose(1, 2).reshape(batch_size, -1, 24 * head_dim)
        ###############################################################

        # Combine the output of the two cross-attention layers, hidden_states = QI.KC x VC + QI.KP x VP
        hidden_states = hidden_states_text_audio + self.scale * ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # Combine the output of the attention outputs with QC, encoder_hidden_states_con = QC.KC x VC + QC.KI x VI 
        condition_hidden_states = condtion_self_hidden_state + condtion_cross_hidden_state
        # linear proj
        condition_hidden_states = self.Oc(condition_hidden_states)

        if input_ndim == 4:
            condition_hidden_states = condition_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            condition_hidden_states = condition_hidden_states + residual

        condition_hidden_states = condition_hidden_states / attn.rescale_output_factor

        return hidden_states, condition_hidden_states
# The attention processor used in MuseControlLite, using 2 decoupled cross-attention layer. It needs further examination, don't use it now.
class StableAudioAttnProcessor2_0_rotary_mm_v2(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(768, 768, bias=False)
        self.to_v_ip = nn.Linear(768, 768, bias=False)
        self.Wqc = nn.Linear(768, 768, bias=False)
        self.norm = nn.LayerNorm(768, elementwise_affine=True, eps=1e-5)
        self.name = name
        self.conv_out_1 = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))  
        self.conv_out_2 = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))   
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        # self.to_k_ip.weight.requires_grad = True
        # self.to_v_ip.weight.requires_grad = True
        # self.conv_out.weight.requires_grad = True
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio: QI.KP x VP
        ###############################################################
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_text_audio = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_text_audio = hidden_states_text_audio.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_text_audio = hidden_states_text_audio.to(query.dtype)
        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions: QI.KC x VC
        ###############################################################
        encoder_hidden_states_con = self.norm(encoder_hidden_states_con)
        ip_key = self.to_k_ip(encoder_hidden_states_con)
        ip_value = self.to_v_ip(encoder_hidden_states_con)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]

        # Generate position_ids for query, keys, values
        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_1025 = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_1024 = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        # Rotate query, keys, values 
        cos_q, sin_q = self.rotary_emb(query, position_ids_1025)
        query_pos = (query * cos_q.unsqueeze(1)) + (self.rotate_half(query) * sin_q.unsqueeze(1))

        cos_k, sin_k = self.rotary_emb(ip_key, position_ids_1024)
        ip_key = (ip_key * cos_k.unsqueeze(1)) + (self.rotate_half(ip_key) * sin_k.unsqueeze(1))
        ip_value = (ip_value * cos_k.unsqueeze(1)) + (self.rotate_half(ip_value) * sin_k.unsqueeze(1))

        ip_hidden_states = F.scaled_dot_product_attention(
                query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ip_hidden_states = self.conv_out_1(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ###############################################################


        # Conditions should see themselves! encoder_hidden_states = QC.KC x VC + QC.KI x VI
        query_c = self.Wqc(encoder_hidden_states_con)
        # key_i = self.Wki(hidden_states)
        # value_i = self.Wvi(hidden_states)
        query_c = query_c.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        # key_i = key_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value_i = value_i.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            query_c = torch.repeat_interleave(query_c, heads_per_kv_head, dim=1)
        query_c_pos = (query_c * cos_k.unsqueeze(1)) + (self.rotate_half(query_c) * cos_k.unsqueeze(1))
        # key_i = (key_i * cos_q.unsqueeze(1)) + (self.rotate_half(key_i) * sin_q.unsqueeze(1))
        # value_i = (value_i * cos_q.unsqueeze(1)) + (self.rotate_half(value_i) * sin_q.unsqueeze(1))

        condtion_self_hidden_state = F.scaled_dot_product_attention(
                query_c_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False)
        condtion_self_hidden_state = condtion_self_hidden_state.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        condtion_self_hidden_state = condtion_self_hidden_state.to(query.dtype)
        condtion_self_hidden_state = condtion_self_hidden_state.transpose(1, 2)
        condtion_self_hidden_state = self.conv_out_2(condtion_self_hidden_state)
        condtion_self_hidden_state = condtion_self_hidden_state.transpose(1, 2)
        # condtion_cross_hidden_state = F.scaled_dot_product_attention(
        #         query_c_pos, key_i, value_i, attn_mask=None, dropout_p=0.0, is_causal=False)
        # condtion_self_hidden_state = condtion_self_hidden_state.transpose(1, 2).reshape(batch_size, -1, 24 * head_dim)
        # condtion_cross_hidden_state = condtion_cross_hidden_state.transpose(1, 2).reshape(batch_size, -1, 24 * head_dim)
        ###############################################################
        # Combine the output of the two cross-attention layers, hidden_states = QI.KC x VC + QI.KP x VP
        pad_zero = torch.zeros((batch_size, 1, 1536), device="cuda")
        condtion_self_hidden_state = torch.cat([pad_zero, condtion_self_hidden_state], dim=1)
        hidden_states = hidden_states_text_audio + self.scale * ip_hidden_states + condtion_self_hidden_state
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # # Combine the output of the attention outputs with QC, encoder_hidden_states_con = QC.KC x VC + QC.KI x VI 
        # condition_hidden_states = condtion_self_hidden_state + condtion_cross_hidden_state
        # # linear proj
        # condition_hidden_states = self.Oc(condition_hidden_states)

        # if input_ndim == 4:
        #     condition_hidden_states = condition_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     condition_hidden_states = condition_hidden_states + residual

        # condition_hidden_states = condition_hidden_states / attn.rescale_output_factor

        return hidden_states
class StableAudioAttnProcessor2_0_rotary_double(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.layer_id = layer_id
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_ip_audio = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip_audio = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))
        self.conv_out_audio = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))
        self.rotary_emb = LlamaRotaryEmbedding(64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
        # Below is for copying the weight of the original weight to the decoupled cross-attention
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # if self.layer_id == "0":
        #     hidden_states_sliced = hidden_states[:,1:,:]
        #     # Create a tensor of zeros with shape (bs, 1, 768)
        #     bs, _, dim2 = hidden_states_sliced.shape
        #     zeros = torch.zeros(bs, 1, dim2).cuda()
        #     # Concatenate the zero tensor along the second dimension (dim=1)
        #     hidden_states_sliced = torch.cat((hidden_states_sliced, zeros), dim=1)
        #     query_sliced = attn.to_q(hidden_states_sliced)
        #     query_sliced = query_sliced.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     query = query_sliced
        ip_hidden_states = encoder_hidden_states_con
        ip_hidden_states_audio = encoder_hidden_states_audio
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        
        ip_key_audio = self.to_k_ip_audio(ip_hidden_states_audio)
        ip_value_audio = self.to_v_ip_audio(ip_hidden_states_audio)
        ip_key_audio = ip_key_audio.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_audio_length = ip_key_audio.shape[2]
        ip_value_audio = ip_value_audio.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
            ip_key_audio = torch.repeat_interleave(ip_key_audio, heads_per_kv_head, dim=1)
            ip_value_audio = torch.repeat_interleave(ip_value_audio, heads_per_kv_head, dim=1)
            
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]
        ip_value_audio_length = ip_value_audio.shape[2]

        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        
        # Generate position_ids for keys
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
        
        # Generate position_ids for keys
        position_ids_query_audio = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_audio_length / seq_len_query)
        position_ids_query_audio = position_ids_query_audio.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key_audio = torch.arange(ip_key_audio_length, dtype=torch.long, device=key.device)
        position_ids_key_audio = position_ids_key_audio.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value_audio = torch.arange(ip_value_audio_length, dtype=torch.long, device=value.device)
        position_ids_value_audio = position_ids_value_audio.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        cos, sin = self.rotary_emb(query, position_ids_query)
        cos_audio, sin_audio = self.rotary_emb(query, position_ids_query_audio)
        query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
        query_pos_audio = (query * cos_audio.unsqueeze(1)) + (self.rotate_half(query) * sin_audio.unsqueeze(1))

        cos, sin = self.rotary_emb(ip_key, position_ids_key)
        cos_audio, sin_audio = self.rotary_emb(ip_key_audio, position_ids_key_audio)
        ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
        ip_key_audio = (ip_key_audio * cos_audio.unsqueeze(1)) + (self.rotate_half(ip_key_audio) * sin_audio.unsqueeze(1)) 

        cos, sin = self.rotary_emb(ip_value, position_ids_value)
        cos_audio, sin_audio = self.rotary_emb(ip_value_audio, position_ids_value_audio)
        ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
        ip_value_audio = (ip_value_audio * cos_audio.unsqueeze(1)) + (self.rotate_half(ip_value_audio) * sin_audio.unsqueeze(1))   

        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states = F.scaled_dot_product_attention(
                    query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states_audio = F.scaled_dot_product_attention(
                    query_pos_audio, ip_key_audio, ip_value_audio, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)

        ip_hidden_states_audio = ip_hidden_states_audio.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states_audio = ip_hidden_states_audio.to(query.dtype)
        ip_hidden_states_audio = ip_hidden_states_audio.transpose(1, 2)

        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states = self.conv_out(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        
        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states_audio = self.conv_out_audio(ip_hidden_states_audio)
        ip_hidden_states_audio = ip_hidden_states_audio.transpose(1, 2)

        # Combine the tensors
        hidden_states = hidden_states + self.scale * ip_hidden_states  + ip_hidden_states_audio
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
