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

        # 1. PREFILL
        prompt_ids = torch.tensor(request_prompt).unsqueeze(0)
        pos_ids = torch.arange(
            prompt_ids.shape[1],
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(prompt_ids, pos_ids, prefill=True)

        next_id = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)
        input_ids = torch.cat([prompt_ids, next_id], dim=1)
        curr_len = input_ids.shape[1]

        for _ in range(128):
            last_token = input_ids[:, -1:]
            pos_ids = torch.tensor([[curr_len - 1]], device=input_ids.device)

            with torch.no_grad():
                logits = self.model(last_token, pos_ids, prefill=False)

            next_id = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)

            if next_id.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_id], dim=1)
            curr_len += 1

        return self.tokenizer.decode(input_ids.squeeze(0).tolist())
