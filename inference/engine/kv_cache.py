from dataclasses import dataclass
import torch


DEVICE = "cpu"


@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor
    length: torch.Tensor

    @classmethod
    def new(
        cls, num_hidden_layers: int, ctx_length: int, num_kv_heads: int, head_dim: int
    ) -> "KVCache":
        return KVCache(
            keys=torch.zeros(
                (num_hidden_layers, ctx_length, num_kv_heads, head_dim), device=DEVICE
            ),
            values=torch.zeros(
                (num_hidden_layers, ctx_length, num_kv_heads, head_dim), device=DEVICE
            ),
            length=torch.tensor([0], device=DEVICE),
        )
