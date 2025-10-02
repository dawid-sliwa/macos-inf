import sys
from inference.http_server import server_start
from inference.server_args import prepare_args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    server_args = prepare_args(argv)
    server_start(args=server_args)
