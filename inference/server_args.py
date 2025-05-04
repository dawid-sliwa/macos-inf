from dataclasses import dataclass
from argparse import ArgumentParser
from typing import List


@dataclass
class ServerArgs:
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8080

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--model_path", type=str, required=True)


def prepare_args(argv: List[str]) -> ServerArgs:
    parser = ArgumentParser()

    ServerArgs.add_args(parser=parser)
    raw = parser.parse_args(argv)
    server_args = ServerArgs(**vars(raw))

    return server_args
