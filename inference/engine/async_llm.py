import time
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

        input_tensor = torch.tensor(request_prompt).unsqueeze(0)
        seq_len = input_tensor.shape[1]
        initial_position_ids = torch.arange(seq_len).unsqueeze(0)
        with torch.no_grad():
            _ = self.model(input_tensor, initial_position_ids, use_cache=True)

        start = time.time()
        for _ in range(32):
            last_idx = input_tensor[:, -1:].clone()
            cur_pos_id = torch.tensor([input_tensor.size(1)])

            with torch.no_grad():
                logits = self.model(last_idx, cur_pos_id, use_cache=True)

            logits = logits[:, -1, :]

            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next.item() == eos_id:
                break

            input_tensor = torch.cat((input_tensor, idx_next), dim=1)

        print(f"generation took: {time.time() - start}")

        return self.tokenizer.decode(input_tensor.squeeze(0).tolist())
