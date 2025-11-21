import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

import argparse
from cfb_predictor import cli
from cfb_predictor.data import RequestController

def main():
    parser, subparsers = cli.initialize_parsers()
    cli.initialize_data_gather(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

    
if __name__ == "__main__":
    main()