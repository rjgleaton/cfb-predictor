import pandas as pd
import json
from pathlib import Path
import logging
from typing import Optional, Callable, Dict, Any, Tuple
from tqdm import tqdm
import warnings

from cfb_predictor.core.exceptions import LargeDataframeError
from cfb_predictor.core.logging import tqdm_logging_redirect

logger = logging.getLogger(__name__)
curr_directory = Path(__file__).parent


class JsonDataProcessor:
    def __init__(
        self,
        input_folder: Optional[Path] = None,
        output_folder: Optional[Path] = None,
        preprocess_fn: Optional[Callable] = None,
    ):
        self.input_folder = input_folder or curr_directory.parent / "gather" / "output"
        self.output_folder = output_folder or curr_directory.parent / "files"
        self.preprocess_fn = preprocess_fn
    
    @staticmethod
    def _flatten(data: Dict[str, Any], id_key: str = "", **extra_kwargs) -> Dict[str, Any]:
        def flatten_recursive(obj, parent_key: str = ''):
            items = {}
            
            if isinstance(obj, dict):
                # Check if this dict has a 'team' key
                if id_key in obj:
                    prefix = obj.pop(id_key)
                    
                    # Process all keys except 'team'
                    for k, v in obj.items():
                        new_key = f"{parent_key}_{k}" if parent_key else k
                        items.update(flatten_recursive(v, new_key))
                    
                    # Prepend home/away prefix to all keys
                    return {f"{prefix}_{k}": v for k, v in items.items()}
                else:
                    # Regular dict without 'team' key
                    for k, v in obj.items():
                        new_key = f"{parent_key}_{k}" if parent_key else k
                        items.update(flatten_recursive(v, new_key))
                    return items
                    
            elif isinstance(obj, list):
                # Process each item in the list
                for item in obj:
                    items.update(flatten_recursive(item, parent_key))
                return items
            else:
                # Base case: primitive value
                return {parent_key: obj}
            
        return flatten_recursive(data)
    
    def process(
        self,
        file_name_pattern: str,
        output_file_name: str,
        flatten: bool = False,
        overwrite: bool = True,
    ) -> pd.DataFrame:

        output_path = self.output_folder / output_file_name
        if output_path.exists() and not overwrite:
            logger.info(f"Output file {output_path} already exists. Loading existing file.")
            return pd.read_csv(output_path)

        df = pd.DataFrame()

        files = list(self.input_folder.glob(file_name_pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {file_name_pattern}")

        with tqdm_logging_redirect():
            pbar = tqdm(files, desc="Processing JSON files", position=0, leave=True)
            for file_path in pbar:
                pbar.set_description(f"Processing {file_path.stem}")
                with open(file_path, 'r') as file:
                    data = json.load(file)

                if self.preprocess_fn:
                    preprocessed_data = self.preprocess_fn(data)
                else:
                    preprocessed_data = {"data": data}

                if flatten:
                    if isinstance(preprocessed_data['data'], list):
                        data = [
                            self._flatten(item, id_key=preprocessed_data.get('id_key', ''))
                            for item in preprocessed_data['data']
                        ]
                    else:
                        data = self._flatten(**preprocessed_data)

                normalized_df = pd.json_normalize(data)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    if df.empty:
                        df = normalized_df
                    else:
                        df = pd.concat([df, normalized_df], ignore_index=True)

                if len(df.columns) > 250:
                    raise LargeDataframeError(f"DataFrame columns exceed maximum allowed columns. {len(df.columns)=}")
                

        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        return df

def process_and_save_all(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
) -> None:
    
    kwargs = {
        "input_folder": input_folder,
        "output_folder": output_folder,
    }

    process_funcs = [
        process_game_stats,
        process_lines,
        process_player_portal,
        process_player_returning,
        process_roster,
        process_talent,
        process_teams,
    ]
    
    for func in process_funcs:
        _ = func(**kwargs)

    return None

def process_game_stats(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "games_full.csv",
) -> pd.DataFrame:
    games_df = process_games(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    games_advanced_df = process_games_advanced_stats(
        input_folder=input_folder,
        output_folder=output_folder,
    )

    games_df = games_df.drop_duplicates(subset='id')
    games_df = games_df[games_df['id'].isin(games_advanced_df['id'])]

    merged_df = pd.merge(
        games_df,
        games_advanced_df,
        on='id',
        how='left',
        suffixes=('', '_advanced'),
    )

    output_file = JsonDataProcessor(output_folder=output_folder).output_folder / output_file_name
    merged_df.to_csv(output_file, index=False)
    logger.info(f"Merged game stats saved to {output_file}")
    return merged_df

def process_games(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "games.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return processor.process(
        file_name_pattern="games__*.json",
        output_file_name=output_file_name,
        flatten=False,
        overwrite=True,
    )

def process_games_advanced_stats(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "games_advanced.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        preprocess_fn=_preprocess_advanced_stats,
    )
    return processor.process(
        file_name_pattern="game_box_advanced*.json",
        output_file_name=output_file_name,
        flatten=True,
        overwrite=True,
    )

def process_lines(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "lines.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        preprocess_fn=_preprocess_lines,
    )
    return processor.process(
        file_name_pattern="lines__*.json",
        output_file_name=output_file_name,
        flatten=True,
        overwrite=True,
    )

def process_player_portal(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "player_portal.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return processor.process(
        file_name_pattern="player_portal_*.json",
        output_file_name=output_file_name,
        overwrite=True,
    )

def process_player_returning(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "player_returning.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return processor.process(
        file_name_pattern="player_returning_*.json",
        output_file_name=output_file_name,
        overwrite=True,
    )

def process_roster(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "roster.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return processor.process(
        file_name_pattern="roster_*.json",
        output_file_name=output_file_name,
        overwrite=True,
    )

def process_talent(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "talent.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return processor.process(
        file_name_pattern="talent_*.json",
        output_file_name=output_file_name,
        overwrite=True,
    )

def process_teams(
    input_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    output_file_name: str = "teams.csv",
) -> pd.DataFrame:
    processor = JsonDataProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return processor.process(
        file_name_pattern="teams__.json",
        output_file_name=output_file_name,
        overwrite=True,
    )

def _preprocess_lines(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"data": data, "id_key": "provider"}

def _preprocess_advanced_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    data.pop('players', None) # Drop this for now

    home_team = data['gameInfo'].pop('homeTeam')
    away_team = data['gameInfo'].pop('awayTeam')

    def recursive_rename_teams(obj: Any) -> Any:
        if isinstance(obj, dict):
            if 'team' in obj:
                team_name = 'home' if obj['team'] == home_team else 'away'
                obj['team'] = team_name

                for k, v in obj.items():
                    if k != 'team':
                        obj[k] = recursive_rename_teams(v)
                return obj
            else:
                return {k: recursive_rename_teams(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_rename_teams(item) for item in obj]
        else:
            return obj

    data.update(data.pop('gameInfo'))

    data['teams'] = recursive_rename_teams(data['teams'])
    data.update(data.pop('teams'))

    return {"data": data, "id_key": "team"}

