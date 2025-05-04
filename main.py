import sys
from inference.http_server import server_start
from inference.server_args import prepare_args


if __name__ == "__main__":
    server_args = prepare_args(sys.argv[1:])
    server_start(args=server_args)
