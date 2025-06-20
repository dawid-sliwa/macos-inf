from torch import nn

from inference.config.model_config import ModelConfig


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, *, config: ModelConfig):
        super().__init__()


class Qwen3Model(nn.Module):
    def __init__(self, *, config: ModelConfig):
        super().__init__()
        self.config = config.hf_config
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(self.config.vocab_size, config.hidden_size)

    def load_weights(self, *, weights):
        print(weights)


class Qwen3ForCasualLM(nn.Module):
    def __init__(self, *, config: ModelConfig):
        super().__init__()

        self.config = config
        self.model = Qwen3Model(config=config)
