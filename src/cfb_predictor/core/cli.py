import argparse
import inspect
from cfb_predictor.data import RequestController
from cfb_predictor.data.process import (
    process_and_save_all,
    process_game_stats,
    process_games,
    process_games_advanced_stats,
    process_lines,
    process_player_portal,
    process_player_returning,
    process_roster,
    process_talent,
    process_teams,
)
from typing import Optional, Tuple, Literal, Dict, Callable

#region Literals and Maps
file_types = Literal["all", "game_stats", "games", "games_advanced_stats", "lines", "player_portal", "player_returning", "roster", "talent", "teams"]
process_map: Dict[file_types, Callable] = {
    "all": process_and_save_all,
    "game_stats": process_game_stats,
    "games": process_games,
    "games_advanced_stats": process_games_advanced_stats,
    "lines": process_lines,
    "player_portal": process_player_portal,
    "player_returning": process_player_returning,
    "roster": process_roster,
    "talent": process_talent,
    "teams": process_teams,
}
#endregion

#region CLI Initialization
def initialize_parsers() -> Tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    # Create parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parent_parser.add_argument("-vv", "--debug", action="store_true", help="Enable debug logging")

    parser = argparse.ArgumentParser(
        description="CFB Predictor CLI", 
        parents=[parent_parser]
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    return parser, subparsers

def initialize_info(subparsers: argparse.ArgumentParser) -> None:
    info_parser = subparsers.add_parser("info", help="Retrieve API information")
    info_parser = _add_api_key_arg(info_parser)
    info_parser.set_defaults(func=handle_info)

def initialize_data_gather(subparsers: argparse.ArgumentParser) -> None:
    gather_parser = subparsers.add_parser("gather", help="Data gathering options")

    gather_parser = _add_api_key_arg(gather_parser)
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


def initialize_data_processor(subparsers: argparse.ArgumentParser) -> None:
    process_parser = subparsers.add_parser("process", help="Data processing options")

    process_parser.add_argument(
        "-f",
        "--files",
        type=str,
        required=True,
        help="File type to process"
    )
    process_parser.add_argument(
        "--input_folder",
        type=str,
        required=False,
        default=None,
        help="Path to the folder containing raw data files"
    )
    process_parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default=None,
        help="Path to the folder to save processed data files"
    )
    process_parser.add_argument(
        "--output_file_name",
        type=str,
        required=False,
        default=None,
        help="Name of the output file for processed data"
    )

    process_parser.set_defaults(func=handle_data_process)

#endregion

#region CLI Handlers
def handle_info(args: argparse.Namespace) -> None:
    controller = RequestController(
        api_key=args.api_key
    )
    controller.report_info()

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

def handle_data_process(args: argparse.Namespace) -> None:
    process_function = process_map.get(args.files, None)
    if not process_function:
        raise ValueError(f"Invalid file type specified [{args.files}]! Please choose from {list(process_map.keys())}.")

    # Dynamically build kwargs since not all process funcs have every argument
    kwargs = {
        key: getattr(args, key) for key in inspect.signature(process_function).parameters.keys()
    }

    process_function(**kwargs)

#endregion

#region Helper Functions
def _add_api_key_arg(sub_parser):
    sub_parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        default=None,
        help="API key for data retrieval"
    )
    return sub_parser
#endregion