import time
from typing import Optional
from torch import nn
import torch
from inference.config.model_config import ModelConfig
from transformers import AutoConfig


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
    x: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
):
    if position_ids.dim() > 1:
        position_ids = position_ids.view(-1)
    cos_table, sin_table = position_embeddings

    cos = cos_table[position_ids].unsqueeze(0).unsqueeze(0)
    sin = sin_table[position_ids].unsqueeze(0).unsqueeze(0)

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    return (x * cos + rotated * sin).to(dtype=x.dtype)


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

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        cache_positions: Optional[torch.LongTensor] = None,
    ):
        bsz, seq_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(
            bsz, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache_positions is None:
            # full‐sequence prefill
            positions = torch.arange(seq_len, device=x.device)
        else:
            # autoregressive decoding
            positions = cache_positions  # shape [q_len]

        queries = apply_rope(queries, position_embeddings, positions)
        keys = apply_rope(keys, position_embeddings, positions)

        k_cache, v_cache = cache

        k_cache[layer_idx, :, positions, :, :] = keys.permute(0, 2, 1, 3)
        v_cache[layer_idx, :, positions, :, :] = values.permute(0, 2, 1, 3)

        K = int(positions.max().item()) + 1
        k_all = k_cache[layer_idx, :, :K, :, :]
        v_all = v_cache[layer_idx, :, :K, :, :]

        keys = k_all.permute(0, 2, 1, 3).repeat_interleave(self.num_kv_groups, dim=1)
        values = v_all.permute(0, 2, 1, 3).repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, seq_len, self.num_attention_heads * self.head_dim
        )

        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, *, config: AutoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Attention(config=config)
        self.mlp = FeedForward(config=config)

        self.input_layernorm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            dim=self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        cache_positions: Optional[torch.LongTensor],
    ):
        residual = x
        x = self.input_layernorm(x)

        x = self.self_attn(
            x,
            position_embeddings,
            attention_mask,
            cache,
            self.layer_idx,
            cache_positions,
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
                Qwen3DecoderLayer(config=self.config, layer_idx=idx)
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

        self.register_buffer(
            "k_cache",
            torch.zeros(
                (
                    self.config.num_hidden_layers,
                    1,
                    self.config.max_position_embeddings,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
            ),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                (
                    self.config.num_hidden_layers,
                    1,
                    self.config.max_position_embeddings,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
            ),
        )

    def forward(self, in_idx, position_ids: torch.LongTensor, use_cache: bool = False):
        token_embeddings = self.embed_tokens(in_idx)
        x = token_embeddings
        attention_mask = None
        for layer in self.layers:
            x = layer(
                x,
                self.position_embeddings,
                attention_mask,
                (self.k_cache, self.v_cache),
                position_ids,
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
        self, idx: torch.Tensor, position_ids: torch.LongTensor, use_cache=False
    ):
        return self.model(idx, position_ids, use_cache)
