from contextlib import asynccontextmanager
import json
import time
import traceback
import uuid
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from inference.config.model_config import ModelConfig
from inference.engine.async_llm import AsyncLLM
from inference.openai_protocol import ChatCompletionRequest
from inference.server_args import ServerArgs
import logging

router = APIRouter()

logging.basicConfig(level=logging.INFO)


def get_engine(request: Request) -> AsyncLLM:
    return request.app.state.engine


@router.get("/health")
async def health_route():
    return {"status": "ok"}


async def stream_response(engine: AsyncLLM, data: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'test', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    response = engine.chat_cmpl_continous_batching(request=data)

    async for token in response:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "test",
            "choices": [
                {"index": 0, "delta": {"content": token}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)} \n\n"

    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "test",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def non_stream_response(engine: AsyncLLM, data: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    choices = {}
    choices[0] = {
        "index": 0,
        "message": {"role": "assistant", "content": ""},
        "finish_reason": None,
    }
    data = engine.chat_cmpl_continous_batching(request=data)

    async for token in data:
        choices[0]["message"]["content"] += token

    choices[0]["finish_reason"] = "stop"

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": "test",
        "choices": list(choices.values()),
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@router.post("/v1/chat/completions")
async def create_chat_completion(data: ChatCompletionRequest, request: Request):
    engine = get_engine(request)
    try:
        if data.stream:
            return StreamingResponse(
                stream_response(engine=engine, data=data),
                media_type="text/event-stream",
            )
        else:
            return await non_stream_response(engine=engine, data=data)
    except Exception:
        traceback.print_exc()


def build_app(args: ServerArgs):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = ModelConfig(model_path=args.model_path)

        engine = AsyncLLM(model_config=config)
        app.state.engine = engine
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="Macos inf", lifespan=lifespan)

    app.include_router(router=router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.server_args = args

    return app


def server_start(args: ServerArgs):
    app = build_app(args=args)

    uvicorn.run(app=app, host=args.host, port=args.port, loop="uvloop")
