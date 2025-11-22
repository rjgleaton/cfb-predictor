import argparse
from cfb_predictor.data import RequestController
from typing import Optional, Tuple

def initialize_parsers() -> Tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    # Create parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    parser = argparse.ArgumentParser(
        description="CFB Predictor CLI", 
        parents=[parent_parser]
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    return parser, subparsers

def initialize_data_gather(subparsers: argparse.ArgumentParser) -> None:
    gather_parser = subparsers.add_parser("gather", help="Data gathering options")

    gather_parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        default=None,
        help="API key for data retrieval"
    )
    gather_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="all",
        help="Name of configuration set to use for data gathering or 'all' to gather all configurations (default: all)"
    )
    gather_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing data files"
    )
    gather_parser.add_argument(
        "--config_folder",
        type=str,
        default=None,
        help="Path to the folder containing configuration files"
    )
    gather_parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Path to the folder to save gathered data"
    )
    gather_parser.set_defaults(func=handle_data_gather)

def handle_data_gather(args: argparse.Namespace) -> None:
    controller = RequestController(
        api_key=args.api_key,
        config_folder=args.config_folder,
        output_folder=args.output_folder
    )

    controller.retrieve_data(
        config_name=args.config,
        overwrite=args.overwrite
    )

    