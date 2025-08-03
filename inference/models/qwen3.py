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
    x: torch.Tensor, position_embeddings: torch.Tensor, position_ids: torch.Tensor
):
    cos, sin = position_embeddings

    cos = cos[position_ids]
    sin = sin[position_ids]

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    head_dim = x.size(-1)

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

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        position_embeddings: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        prefill: Optional[bool] = False,
    ):
        bsz, seq_len, _ = x.shape  # shape [batch_size, seq_len, hidden_dim]
        k_cache, v_cache = cache

        queries = self.q_proj(x)  # shape [batch_size, seq_len, num_heads, head_dim]
        keys = self.k_proj(x)  # shape [batch_size, seq_len, num_kv_heads, head_dim]
        values = self.v_proj(x)  # shape [batch_size, seq_len, num_kv_heads, head_dim]

        queries = queries.view(
            bsz, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # shape [batch_size, num_heads, seq_len, head_dim]
        keys = keys.view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # shape [batch_size, num_kv_heads, seq_len, head_dim]
        values = values.view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # shape [batch_size, num_kv_heads, seq_len, head_dim]

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = apply_rope(queries, position_embeddings, position_ids)
        keys = apply_rope(keys, position_embeddings, position_ids)

        if prefill:
            k_cache[layer_idx, :bsz, :seq_len] = keys.permute(0, 2, 1, 3)
            v_cache[layer_idx, :bsz, :seq_len] = values.permute(0, 2, 1, 3)
        else:
            cur_pos = int(position_ids[0].item())
            k_cache[layer_idx, :bsz, cur_pos] = keys.squeeze(2)
            v_cache[layer_idx, :bsz, cur_pos] = values.squeeze(2)

            keys = k_cache[layer_idx, :bsz, : cur_pos + 1].permute(0, 2, 1, 3)
            values = v_cache[layer_idx, :bsz, : cur_pos + 1].permute(0, 2, 1, 3)

        keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
        values = values.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = (
            torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask=None, is_causal=prefill
            )
            .transpose(1, 2)
            .reshape(bsz, seq_len, self.num_attention_heads * self.head_dim)
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
        position_embeddings: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        prefill: Optional[bool] = False,
    ):
        residual = x
        x = self.input_layernorm(x)

        x = self.self_attn(
            x, self.idx, position_embeddings, cache, position_ids, prefill
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
            persistent=False,
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
            persistent=False,
        )

    def forward(
        self,
        in_idx,
        position_ids: torch.Tensor,
        prefill: Optional[bool] = False,
    ):
        token_embeddings = self.embed_tokens(in_idx)
        x = token_embeddings

        for layer in self.layers:
            x = layer(
                x,
                self.position_embeddings,
                (self.k_cache, self.v_cache),
                position_ids,
                prefill,
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
        idx: torch.Tensor,
        position_ids: torch.Tensor,
        prefill: Optional[bool] = False,
    ):
        return self.model(idx, position_ids, prefill)
