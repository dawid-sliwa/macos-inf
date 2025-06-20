from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference.config.model_config import ModelConfig
from inference.engine.async_llm import AsyncLLM
from inference.server_args import ServerArgs

router = APIRouter()


@router.get("/health")
async def health_route():
    return {"status": "ok"}


def build_app(args: ServerArgs):
    app = FastAPI(title="Macos inf")

    app.include_router(router=router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.server_args = args

    config = ModelConfig(model_path=args.model_path)

    engine = AsyncLLM(model_config=config)
    return app


def server_start(args: ServerArgs):
    app = build_app(args=args)

    uvicorn.run(app=app, host=args.host, port=args.port, loop="uvloop")
