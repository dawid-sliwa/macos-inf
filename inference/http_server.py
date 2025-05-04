from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference.server_args import ServerArgs

app = FastAPI(title="Macos inf")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_route():
    return {"status": "ok"}


def server_start(args: ServerArgs):
    app.server_args = args

    uvicorn.run(
        app=app,
        host=args.host,
        port=args.port,
        loop="uvloop"
    )