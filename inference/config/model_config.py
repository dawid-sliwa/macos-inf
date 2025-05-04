from typing import Optional

from inference.hf_utils import get_config


class ModelConfig:
    def __init__(
        self,
        *,
        model_path: str,
        context_length: Optional[int] = None,
        dtype: str = "auto",
    ):
        self.model_path = model_path
        self.hf_config = get_config(self.model_path)