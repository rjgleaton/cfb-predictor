from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

import pandas as pd
from typing import Dict, List

@dataclass
class CFBMatchupSample:
    team_node_idxs: torch.Tensor
    opponent_node_idxs: torch.Tensor
    win: torch.Tensor
    margin: torch.Tensor

class CFBMatchupDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        node_idx_map: Dict[str, int],
        team_node_id_col: str = 'node_id',
        opponent_node_id_col: str = 'opponent_node_id',
        team_points_cols: str = 'points',
        opponent_points_cols: str = 'opponent_points'
    ):
        self.df = df
        self.node_idx_map = node_idx_map
        self.team_node_id_col = team_node_id_col
        self.opponent_node_id_col = opponent_node_id_col
        self.team_points_cols = team_points_cols
        self.opponent_points_cols = opponent_points_cols

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int | List[int]) -> CFBMatchupSample:
        rows = self.df.iloc[idx]
        team_node_idxs = rows[self.team_node_id_col].map(self.node_idx_map).tolist()
        opponent_node_idxs = rows[self.opponent_node_id_col].map(self.node_idx_map).tolist()

        win = (rows[self.team_points_cols] > rows[self.opponent_points_cols]).astype(int).tolist()
        margin = (rows[self.team_points_cols] - rows[self.opponent_points_cols]).tolist()

        return CFBMatchupSample(
            team_node_idxs=torch.tensor(team_node_idxs, dtype=torch.long),
            opponent_node_idxs=torch.tensor(opponent_node_idxs, dtype=torch.long),
            win=torch.tensor(win, dtype=torch.long),
            margin=torch.tensor(margin, dtype=torch.float32)
        )

def collate_cfb_matchup_samples(
    samples: List[CFBMatchupSample]
) -> CFBMatchupSample:
    team_node_idxs = torch.stack([s.team_node_idxs for s in samples], dim=0)
    opponent_node_idxs = torch.stack([s.opponent_node_idxs for s in samples], dim=0)
    win = torch.stack([s.win for s in samples], dim=0)
    margin = torch.stack([s.margin for s in samples], dim=0)

    return CFBMatchupSample(
        team_node_idxs=team_node_idxs,
        opponent_node_idxs=opponent_node_idxs,
        win=win,
        margin=margin
    )
