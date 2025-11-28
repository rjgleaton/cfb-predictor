from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

import pandas as pd
from typing import Dict, List

@dataclass
class CFBMatchupSample:
    team_node_idxs: torch.Tensor
    opponent_node_idxs: torch.Tensor
    conference_ids: torch.Tensor
    win: torch.Tensor
    margin: torch.Tensor

class CFBMatchupDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        node_idx_map: Dict[str, int],
        team_node_id_col: str = 'previous_node_id',
        opponent_node_id_col: str = 'previous_opponent_node_id',
        conference_id_col: str = 'conference_id',
        team_points_cols: str = 'points',
        opponent_points_cols: str = 'opponent_points'
    ):
        self.df = df
        self.node_idx_map = node_idx_map
        self.team_node_id_col = team_node_id_col
        self.opponent_node_id_col = opponent_node_id_col
        self.conference_id_col = conference_id_col
        self.team_points_cols = team_points_cols
        self.opponent_points_cols = opponent_points_cols

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> CFBMatchupSample:
        row = self.df.iloc[idx]
        team_node_idxs = self.node_idx_map[row[self.team_node_id_col]]
        opponent_node_idxs = self.node_idx_map[row[self.opponent_node_id_col]]
        conference_ids = row[self.conference_id_col]

        win = int(row[self.team_points_cols] > row[self.opponent_points_cols])
        margin = row[self.team_points_cols] - row[self.opponent_points_cols]

        return CFBMatchupSample(
            team_node_idxs=torch.tensor(team_node_idxs, dtype=torch.long),
            opponent_node_idxs=torch.tensor(opponent_node_idxs, dtype=torch.long),
            conference_ids=torch.tensor(conference_ids, dtype=torch.long),
            win=torch.tensor(win, dtype=torch.long),
            margin=torch.tensor(margin, dtype=torch.float32)
        )

def collate_cfb_matchup_samples(
    samples: List[CFBMatchupSample]
) -> CFBMatchupSample:
    team_node_idxs = torch.stack([s.team_node_idxs for s in samples], dim=0)
    opponent_node_idxs = torch.stack([s.opponent_node_idxs for s in samples], dim=0)
    conference_ids = torch.stack([s.conference_ids for s in samples], dim=0)
    win = torch.stack([s.win for s in samples], dim=0)
    margin = torch.stack([s.margin for s in samples], dim=0)

    return CFBMatchupSample(
        team_node_idxs=team_node_idxs,
        opponent_node_idxs=opponent_node_idxs,
        conference_ids=conference_ids,
        win=win,
        margin=margin
    )
