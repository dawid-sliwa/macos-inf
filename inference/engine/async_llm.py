import asyncio
from collections import deque
from dataclasses import dataclass
import queue
import traceback
from typing import List, Optional, Union
from uuid import uuid4

import torch
from inference.config.model_config import ModelConfig
from inference.engine.kv_cache import KVCache
from inference.model_loader.model_loader import ModelLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from inference.models.qwen3continous import (
    ForwardBatch,
    ForwardMode,
    Qwen3ModelInstance,
)
from inference.openai_protocol import ChatCompletionRequest
import logging


@dataclass
class InferenceRequest:
    id: str
    max_new_tokens: int
    request: ChatCompletionRequest
    out_tokens: asyncio.Queue
    cache: Optional[KVCache] = None
    last_token: Optional[int] = None
    cur_len: Optional[int] = None
    remaining: Optional[int] = None


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
        self.eos_id = self.tokenizer.eos_token_id

        self.max_slots = 6
        self.max_ctx_length = 1024
        self.prefill_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.slots = deque([i for i in range(self.max_slots)])
        self.active: dict[str, InferenceRequest] = {}

        self._worker: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()

    def start(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._worker_task())

    def stop(self):
        if self._worker is not None:
            self._shutdown.set()
            self._worker.cancel()

    async def _worker_task(self):
        while not self._shutdown.is_set():
            logging.info("running engine loop")
            try:
                await self.accept_request()
                if self.active:
                    await self.engine_step()
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                traceback.print_exc()
                logging.error(f"hit error: {e}")
                break

    async def accept_request(self):
        while self.slots and not self.prefill_queue.empty():
            req = self.prefill_queue.get()
            logging.info(f"accepting request: {req.id} for prefill")
            slot = self.slots.popleft()

            token_ids = self.tokenizer.apply_chat_template(
                conversation=req.request.messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            lengths = torch.tensor([len(token_ids)], dtype=torch.long)
            total_len = int(lengths.sum())
            bsz = lengths.numel()
            start_offsets = torch.zeros(bsz, dtype=torch.long)
            flat_tokens = torch.tensor(token_ids, dtype=torch.long)
            batch_of_pos = torch.repeat_interleave(
                torch.arange(bsz, dtype=torch.long), lengths
            )
            global_idx = torch.arange(total_len, dtype=torch.long)

            pos_in_seq = global_idx - start_offsets[batch_of_pos]

            same_batch = batch_of_pos[:, None] == batch_of_pos[None, :]
            causal = pos_in_seq[:, None] >= pos_in_seq[None, :]

            allowed = same_batch & causal
            attn_mask = torch.where(
                allowed,
                torch.zeros((), device=allowed.device),
                torch.full((), float("-inf"), device=allowed.device),
            )

            end_positions = start_offsets + lengths - 1
            kv_cache = KVCache.new(
                num_hidden_layers=self.config.num_hidden_layers,
                ctx_length=4096,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
            )
            fwd_batch_prefill = ForwardBatch(
                forward_mode=ForwardMode.PREFILL,
                lengths=lengths,
                start_offsets=start_offsets,
                batch_of_pos=batch_of_pos,
                pos_in_seq=pos_in_seq,
                attn_mask=attn_mask,
                kv_caches=[kv_cache],
                end_positions=end_positions,
            )

            with torch.inference_mode():
                logits = self.model(
                    input_ids=flat_tokens,
                    position_ids=pos_in_seq,
                    forward_batch=fwd_batch_prefill,
                )
                next_ids = torch.argmax(logits[end_positions], dim=1, keepdim=True)
                new_ids = next_ids.flatten(-2, 1)
                # we try index to 0 because we do on sequence per forward pass in the prefill
                next_token = self.tokenizer.decode(new_ids[0])
                await req.out_tokens.put(next_token)

            req.cur_len = len(token_ids)
            req.remaining = self.max_ctx_length - req.cur_len
            req.last_token = int(new_ids[0].item())
            req.cache = kv_cache
            self.active[slot] = req

    async def prefill(self, request: InferenceRequest):
        self.prefill_queue.put(request, block=False)

    async def engine_step(self):
        slots = list(self.active.keys())
        bsz = len(slots)
        lengths = []
        kv_caches = []
        tokens = []
        for slot_id in slots:
            slot_request = self.active[slot_id]
            lengths.append(slot_request.cur_len)
            tokens.append(slot_request.last_token)
            kv_caches.append(slot_request.cache)

        token_ids = torch.tensor(tokens, dtype=torch.long)
        decode_lengths = torch.tensor(lengths, dtype=torch.long)
        pos_ids = torch.arange(bsz, dtype=torch.long)

        bsz = decode_lengths.numel()
        start_offsets_decode = torch.zeros(bsz, dtype=torch.long)
        if bsz > 1:
            start_offsets_decode[1:] = torch.cumsum(decode_lengths[:-1], dim=0)

        batch_of_pos_keys = torch.repeat_interleave(
            torch.arange(bsz, dtype=torch.long), decode_lengths
        )

        allowed = pos_ids[:, None] == batch_of_pos_keys[None, :]

        attn_mask_decode = torch.where(
            allowed,
            torch.zeros((), device=allowed.device),
            torch.full((), float("-inf"), device=allowed.device),
        )
        pos_in_seq_decode = decode_lengths - 1
        fwd_batch_decode = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            lengths=decode_lengths,
            start_offsets=start_offsets_decode,
            batch_of_pos=pos_ids,
            pos_in_seq=pos_in_seq_decode,
            attn_mask=attn_mask_decode,
            kv_caches=kv_caches,
        )
        with torch.inference_mode():
            logits = self.model(
                input_ids=token_ids,
                position_ids=pos_in_seq_decode,
                forward_batch=fwd_batch_decode,
            )
            next_ids = torch.argmax(logits, dim=1, keepdim=True)
            new_ids = next_ids.flatten(-2, 1)
        finished = []
        for idx, slot_id in enumerate(slots):
            slot_request = self.active[slot_id]
            
            request_token = int(new_ids[idx].item())
            if request_token == self.eos_id or slot_request.remaining == 0:
                await slot_request.out_tokens.put(None)
                finished.append(slot_id)
                continue

            detokenized = self.tokenizer.decode(request_token)
            await slot_request.out_tokens.put(detokenized)
            slot_request.cur_len += 1
            slot_request.remaining -= 1
            slot_request.last_token = request_token

        for finished_id in finished:
            del self.active[finished_id]
            self.slots.append(finished_id)
            

    async def chat_cmpl_continous_batching(self, request: ChatCompletionRequest):
        chat_cmpl_request = InferenceRequest(
            id=str(uuid4()),
            max_new_tokens=self.max_ctx_length,
            cache=None,
            request=request,
            out_tokens=asyncio.Queue(),
        )
        logging.info(f"putting a request with id:{chat_cmpl_request.id} for prefill")
        await self.prefill(chat_cmpl_request)

        while True:
            token = await chat_cmpl_request.out_tokens.get()
            logging.info("generation")
            if token is None:
                break
            yield token
