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


def apply_rope(x: torch.Tensor, position_embeddings: torch.Tensor):
    cos, sin = position_embeddings
    _, _, seq_len, head_dim = x.shape

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x)
        return self.weight * output


class FeedForward(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)

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
            self.hidden_size, self.num_attention_heads * self.head_dim
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size
        )

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
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

        queries = apply_rope(queries, position_embeddings)
        keys = apply_rope(keys, position_embeddings)

        keys = keys.repeat_interleave(self.num_kv_groups, dim=1)
        values = values.repeat_interleave(self.num_kv_groups, dim=1)

        attn_score = queries @ keys.transpose(2, 3)
        attn_score = attn_score.masked_fill(attention_mask, -torch.inf)
        attn_weight = torch.nn.functional.softmax(
            attn_score / self.head_dim**-0.5, dim=-1, dtype=torch.float32
        ).to(queries.dtype)

        attn_output = (
            (attn_weight @ values)
            .transpose(1, 2)
            .reshape(bsz, seq_len, self.num_attention_heads * self.head_dim)
        )

        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, *, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attention = Attention(config=config)
        self.ffn = FeedForward(config=config)

        self.input_layernorm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)
        self.output_layernorm = RMSNorm(dim=self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        residual = x
        x = self.input_layernorm(x)

        x = self.self_attention(x, position_embeddings, attention_mask)
        x = residual + x

        residual = x
        x = self.output_layernorm(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Qwen3Model(nn.Module):
    def __init__(self, *, config: AutoConfig):
        super().__init__()
        self.config = config.hf_config
        self.vocab_size = self.config.vocab_size
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config=self.config)
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(
            dim=self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.out_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size)

        self.position_embeddings = compute_rope_params(
            head_dim=self.hidden_size,
            theta_base=self.config,
            context_length=self.config.max_position_embeddings,
        )

    def forward(self, in_idx):
        token_embeddings = self.embeddings(in_idx)
        x = token_embeddings

        

class Qwen3ModelInstance(nn.Module):
    def __init__(self, *, config: ModelConfig):
        super().__init__()

        self.config = config.hf_config
        self.model = Qwen3Model(config=self.config)

    def load_weights(self, *, weights):
        print(weights)
