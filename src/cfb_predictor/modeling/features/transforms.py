import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

class TeamNormalizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        shared_cols=['id', 'season', 'week', 'week_of_year', 'startDate'],
        include_opponent_stats: bool = False,
        ):
        
        self.shared_cols = shared_cols
        self.include_opponent_stats = include_opponent_stats

    def fit(self, X, y=None):
        # Get all columns with home_/away_ prefixes
        self.home_stat_cols = [col for col in X.columns if col.startswith('home_')]
        self.away_stat_cols = [col for col in X.columns if col.startswith('away_')]
        
        # Build rename mappings
        self.home_to_base = {col: col.replace('home_', '') for col in self.home_stat_cols}
        self.away_to_opp = {col: col.replace('away_', 'opponent_') for col in self.away_stat_cols}
        self.away_to_base = {col: col.replace('away_', '') for col in self.away_stat_cols}
        self.home_to_opp = {col: col.replace('home_', 'opponent_') for col in self.home_stat_cols}
        return self

    def _create_team_df(self, df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
        team = "home" if is_home else "away"
        opponent = "away" if is_home else "home"

        rename_map = {
            f'{team}Team': 'team',
            f'{team}Id': 'teamId',
            f'{opponent}Team': 'opponent',
            f'{opponent}Id': 'opponentId',
            f'{team}Conference': 'conference',
            f'{team}Classification': 'classification',
            f'{team}Points': 'points',
            f'{opponent}Points': 'opponent_points',
        }

        def get_stats_map(team_map, opponent_map):
            if self.include_opponent_stats:
                return {**team_map, **opponent_map}
            return team_map

        if is_home:
            rename_map = {**rename_map, **get_stats_map(self.home_to_base, self.away_to_opp)}  
        else:
            rename_map = {**rename_map, **get_stats_map(self.away_to_base, self.home_to_opp)}


        team_df = df.reset_index(drop=True).copy()
        team_df = team_df.rename(columns=rename_map)

        if not self.include_opponent_stats:
            # Drop opponent stats cols if we don't want to include them
            team_df = team_df.drop(columns=[col for col in team_df.columns if opponent in col])

        team_df['is_home'] = int(is_home)
        return team_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        home_df = self._create_team_df(df, is_home=True)
        away_df = self._create_team_df(df, is_home=False)
        result = pd.concat([home_df, away_df], axis=0, ignore_index=True)

        return result.sort_values(['season', 'week_of_year', 'team']).reset_index(drop=True)


class WeeksOffCalculator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        group_cols: List[str] = ['teamId', 'season'],
        week_col: str = 'week_of_year'
        ):
        self.group_cols = group_cols
        self.week_col = week_col
    
    def fit(self, X, y=None):
        _check_cols_in_df(X, self.group_cols + [self.week_col])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.sort_values(self.week_col)
        X['weeks_off'] = X.groupby(self.group_cols)[self.week_col].diff().fillna(0).astype(int) - 1
        X['weeks_off'] = X['weeks_off'].clip(lower=0)
        return X


class NodeIdAdder(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            team_id_cols: List[str] = ['team', 'teamId'],
            opponent_id_cols: List[str] = ['opponent', 'opponentId'],
            additional_cols: List[str] = ['season', 'week_of_year']
        ):
        self.team_id_cols = team_id_cols
        self.opponent_id_cols = opponent_id_cols
        self.additional_cols = additional_cols

    def fit(self, X, y=None):
        _check_cols_in_df(X, [*self.team_id_cols, *self.opponent_id_cols, *self.additional_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X['node_id'] = X[self.team_id_cols + self.additional_cols].astype(str).agg('_'.join, axis=1)
        X['opponent_node_id'] = X[self.opponent_id_cols + self.additional_cols].astype(str).agg('_'.join, axis=1)
        return X
    
class TeamConferenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            conference_col: str = 'conference',
            classification_col: str = 'classification'
        ):
        self.conference_col = conference_col
        self.classification_col = classification_col

    def fit(self, X, y=None):
        _check_cols_in_df(X, [self.conference_col, self.classification_col])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X['classification'] = (X[self.classification_col].str.lower() == 'fbs').astype(int)

        conf_id_map = {conf: idx for idx, conf in enumerate(X[self.conference_col].unique())}
        X['conference_id'] = X[self.conference_col].map(conf_id_map)
        return X

class TeamTalentMerger(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            talent_df: pd.DataFrame,
            talent_df_team_col: str = 'team',
            talent_df_season_col: str = 'year',
            team_col: str = 'team',
            season_col: str = 'season',
            talent_col: str = 'talent'
        ):
        self.talent_df = talent_df
        self.talent_df_team_col = talent_df_team_col
        self.talent_df_season_col = talent_df_season_col
        self.team_col = team_col
        self.season_col = season_col
        self.talent_col = talent_col

    def fit(self, X, y=None):
        _check_cols_in_df(X, [self.team_col, self.season_col])
        _check_cols_in_df(self.talent_df, [self.talent_df_team_col, self.talent_df_season_col, self.talent_col])

        self.col_diff = []
        if self.team_col != self.talent_df_team_col:
            self.col_diff.append(self.talent_df_team_col)
        if self.season_col != self.talent_df_season_col:
            self.col_diff.append(self.talent_df_season_col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        merged_df = X.merge(
            self.talent_df[[self.talent_df_team_col, self.talent_df_season_col, self.talent_col]],
            left_on=[self.team_col, self.season_col],
            right_on=[self.talent_df_team_col, self.talent_df_season_col],
            how='left'
        ).drop(columns=self.col_diff)
        return merged_df

def _check_cols_in_df(df: pd.DataFrame, cols: List[str]):
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")