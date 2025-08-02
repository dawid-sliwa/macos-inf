from typing import Union

import torch
from inference.config.model_config import ModelConfig
from inference.model_loader.model_loader import ModelLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from inference.models.qwen3 import Qwen3ModelInstance
from inference.openai_protocol import ChatCompletionRequest


class AsyncLLM:
    def __init__(self, model_config: ModelConfig):
        loader = ModelLoader()

        self.model = Qwen3ModelInstance(config=model_config)
        self.loader = loader.load_model(config=model_config, model=self.model)
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
            AutoTokenizer.from_pretrained(model_config.model_path)
        )

    async def create_chat_completion(self, request: ChatCompletionRequest):
        request_prompt = self.tokenizer.apply_chat_template(
            conversation=request.messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        eos_id = self.tokenizer.eos_token_id

        # prefill with initial input prompt getting back logits of new token
        input_tensor = torch.tensor(request_prompt).unsqueeze(0)
        positions_ids = torch.arange(input_tensor.shape[1]).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor, positions_ids, True)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        input_tensor = torch.cat((input_tensor, idx_next), dim=1)
        seq_len = input_tensor.shape[1]

        for idx in range(16):
            positions_ids = torch.tensor([[seq_len + idx]])
            input_idx = input_tensor[:, -1].unsqueeze(0)
            with torch.no_grad():
                logits = self.model(input_idx, positions_ids, False)

            logits = logits[:, -1, :]

            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next.item() == eos_id:
                break

            input_tensor = torch.cat((input_tensor, idx_next), dim=1)

        return self.tokenizer.decode(input_tensor.squeeze(0).tolist())
