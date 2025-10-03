# macos-serving

This repository contains a minimal continuous batching LLM engine. The current implementation is a straightforward PyTorch prototype focused on exercising the serving loop and request scheduler.

## Current Capabilities

- Runs a single dense model configuration: Qwen 0.6B.
- Demonstrates continuous batching to keep GPU execution saturated under streaming request load.

## Vision

- Transition the backend to Metal-accelerated kernels purpose-built for Apple silicon.
- Broaden model coverage and support quantized variants suited for on-device serving.
- Introduce production-grade observability, adaptive batching heuristics, and deployment tooling tailored for macOS.

## Quick Start

1. Make sure you have uv installed

```bash
uvx macos-serving --model_path Qwen/Qwen3-0.6B
```
This command will install all required dependencies and run an API on port 4444 by default.

After server goes up you can go to `http://localhost:4444` some simple chat page will be exposed there that you can use to test out inference process.
Additionally you can use exposed API as openai compatible server and do something like this
```
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4444/v1", api_key="")
messages = [
    {"role": "system", "content": "You are helpful assistant, answer user questions"},
    {"role": "user", "content": "Who was Shakesphere? Tell me in detail"},
]
response = client.chat.completions.create(messages=messages, model="w/e")
print(response)
```
Oh and right now there is no sophisticated sampling implemented, during decoding next token is just sampled using argmax from logits.

For now this is just an experimental-exploration project in future there is a plan to further optimize it and play around with metal kernes.

