from dataclasses import dataclass
from enum import StrEnum, auto
from typing import List, Optional
from torch import nn
import torch
from inference.config.model_config import ModelConfig
from transformers import AutoConfig

from inference.engine.kv_cache import KVCache



class ForwardMode(StrEnum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    lengths: torch.Tensor
    start_offsets: torch.Tensor
    batch_of_pos: torch.Tensor
    pos_in_seq: torch.Tensor
    attn_mask: torch.Tensor
    kv_caches: List[KVCache]
    end_positions: Optional[torch.Tensor] = None



def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)

    angles = positions[:, None] * inv_freq[None, :]

    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(
    x: torch.Tensor, position_embeddings: torch.Tensor, position_ids: torch.Tensor
):
    _, _, head_dim = x.shape
    cos, sin = position_embeddings

    cos = cos[position_ids].unsqueeze(0)
    sin = sin[position_ids].unsqueeze(0)

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    rotated = torch.cat((-x2, x1), dim=-1)

    return (x * cos) + (rotated * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x)
        return self.weight * output


class FeedForward(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )
        return down_proj


class Attention(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()

        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def _prefill(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ):
        pass

    def _decode(self):
        pass

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
        position_embeddings: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        seq_len, _ = x.shape  # shape [seq_len, hidden_dim]

        queries = self.q_proj(x)  # shape [seq_len, hidden_dim]
        keys = self.k_proj(x)  # shape [seq_len, hidden_dim]
        values = self.v_proj(x)  # shape [seq_len, hidden_dim]

        queries = queries.reshape(
            seq_len, self.num_attention_heads, self.head_dim
        ).transpose(0, 1)  # shape [num_heads, seq_len, head_dim]
        keys = keys.reshape(seq_len, self.num_key_value_heads, self.head_dim).transpose(
            0, 1
        )  # shape [num_kv_heads, seq_len, head_dim]
        values = values.reshape(
            seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 1)  # shape [num_kv_heads, seq_len, head_dim]

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = apply_rope(queries, position_embeddings, position_ids)
        keys = apply_rope(keys, position_embeddings, position_ids)

        # KVCache here
        if forward_batch.forward_mode == ForwardMode.PREFILL:
            for idx, batch_seq_len in enumerate(forward_batch.lengths):
                r_cache = forward_batch.kv_caches[idx]
                start = forward_batch.start_offsets[idx]
                end = forward_batch.end_positions[idx] + 1
                r_cache.keys[layer_idx, :batch_seq_len] = keys[:, start:end, :].permute(
                    1, 0, 2
                )
                r_cache.values[layer_idx, :batch_seq_len] = values[
                    :, start:end, :
                ].permute(1, 0, 2)
                r_cache.length[0] = batch_seq_len
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            gathered_keys = []
            gathered_values = []
            for idx, batch_seq_len in enumerate(forward_batch.pos_in_seq):
                r_cache = forward_batch.kv_caches[idx]

                r_cache.keys[layer_idx, batch_seq_len.item()] = keys[
                    :, idx : idx + 1, :
                ].permute(1, 0, 2)
                r_cache.values[layer_idx, batch_seq_len.item()] = values[
                    :, idx : idx + 1, :
                ].permute(1, 0, 2)

                gathered_keys.append(r_cache.keys[layer_idx, : batch_seq_len + 1])
                gathered_values.append(r_cache.values[layer_idx, : batch_seq_len + 1])
            keys = torch.cat(gathered_keys, dim=0).transpose(0, 1)
            values = torch.cat(gathered_values, dim=0).transpose(0, 1)

        keys = keys.repeat_interleave(self.num_kv_groups, dim=0)
        values = values.repeat_interleave(self.num_kv_groups, dim=0)

        matmul_res = torch.matmul(queries, keys.transpose(-2, -1)) * (
            self.head_dim**-0.5
        )
        matmul_res = matmul_res + forward_batch.attn_mask
        attn_score = torch.nn.functional.softmax(
            matmul_res, dim=-1, dtype=torch.float32
        ).to(dtype=x.dtype)
        attn_output = torch.matmul(attn_score, values)
        attn_output = attn_output.transpose(0, 1).reshape(
            seq_len, self.num_attention_heads * self.head_dim
        )

        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, *, config: AutoConfig, idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config)
        self.mlp = FeedForward(config=config)

        self.input_layernorm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            dim=self.hidden_size, eps=config.rms_norm_eps
        )
        self.idx = idx

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        residual = x
        x = self.input_layernorm(x)

        x = self.self_attn(
            x=x,
            position_ids=position_ids,
            layer_idx=self.idx,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3Model(nn.Module):
    def __init__(self, *, config: AutoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config=self.config, idx=idx)
                for idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(dim=self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        self.position_embeddings = compute_rope_params(
            head_dim=self.config.head_dim,
            theta_base=self.config.rope_theta,
            context_length=self.config.max_position_embeddings,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        token_embeddings = self.embed_tokens(input_ids)
        x = token_embeddings

        for layer in self.layers:
            x = layer(
                x=x,
                position_embeddings=self.position_embeddings,
                position_ids=position_ids,
                forward_batch=forward_batch,
            )

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class Qwen3ModelInstance(nn.Module):
    def __init__(self, *, config: ModelConfig):
        super().__init__()

        self.config = config.hf_config
        self.model = Qwen3Model(config=self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        forward_batch: ForwardBatch,
    ):
        return self.model(
            input_ids,
            position_ids,
            forward_batch,
        )

