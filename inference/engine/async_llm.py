import asyncio
from collections import deque
from dataclasses import dataclass
from typing import List, Union

import torch
from inference.config.model_config import ModelConfig
from inference.engine.kv_cache import KVCache
from inference.model_loader.model_loader import ModelLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from inference.models.qwen3 import Qwen3ModelInstance
from inference.openai_protocol import ChatCompletionRequest


@dataclass
class ActiveRequest:
    id: str
    max_new_tokens: int
    cache: KVCache


DEVICE = "cpu"


class AsyncLLM:
    def __init__(self, model_config: ModelConfig):
        loader = ModelLoader()

        self.model = Qwen3ModelInstance(config=model_config)
        self.config = model_config.hf_config
        self.loader = loader.load_model(config=model_config, model=self.model)
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
            AutoTokenizer.from_pretrained(model_config.model_path)
        )
        self.tokenizer.padding_side = "left"

        self.max_slots = 6
        self.max_ctx_length = 1024
        self.prefill_queue = asyncio.Queue()
        self.slots = deque([i for i in range(self.max_slots)])
        self.active = {}

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

        print(input_ids)

        return self.tokenizer.decode(input_ids.squeeze(0).tolist())

    async def test_create_chat_compl(self, requests: List[ChatCompletionRequest]):
        conversations = []
        for request in requests:
            conversations.append(request.messages)

        request_prompt = self.tokenizer.apply_chat_template(
            conversation=conversations,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        eos_id = self.tokenizer.eos_token_id

        kv_cache1 = KVCache.new(
            self.config.num_hidden_layers,
            1024,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        kv_cache2 = KVCache.new(
            self.config.num_hidden_layers,
            1024,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )

        input_ids = request_prompt["input_ids"]
        input_ids = input_ids.to(DEVICE)
        attention_mask = request_prompt["attention_mask"]

        lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        bsz, seq_len = input_ids.shape

        position_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=DEVICE)
        for b in range(bsz):
            L = lengths[b].item()
            position_ids[b, seq_len - L : seq_len] = torch.arange(L, device=DEVICE)

        cache_positions = torch.full(
            (bsz, seq_len), -1, dtype=torch.long, device=DEVICE
        )
        for b in range(bsz):
            L = lengths[b].item()
            cache_positions[b, seq_len - L : seq_len] = torch.arange(L, device=DEVICE)

        S = int(lengths.max().item())
        key_index = torch.arange(S, device=DEVICE).contiguous().view(1, 1, 1, S)

        qpos = position_ids.contiguous().view(bsz, 1, seq_len, 1)

        causal = key_index > qpos
        src_pad = key_index >= lengths.contiguous().view(bsz, 1, 1, 1)


        tgt_pad = (position_ids == 0) & (
            torch.arange(seq_len, device=DEVICE).unsqueeze(0)
            < (seq_len - lengths).unsqueeze(1)
        )
        tgt_pad = tgt_pad.contiguous().view(bsz, 1, seq_len, 1).expand(bsz, 1, seq_len, S)

        final_mask = ~(causal | src_pad | tgt_pad)

        with torch.no_grad():
            logits = self.model(
                input_ids,
                [kv_cache1, kv_cache2],
                final_mask,
                position_ids,
                cache_positions,
                True,
            )

        next_id = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)
        print(self.tokenizer.decode([next_id[0].item()]), self.tokenizer.decode([next_id[1].item()]))

        return "asd"
