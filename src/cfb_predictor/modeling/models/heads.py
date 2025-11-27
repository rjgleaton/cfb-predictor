import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GamePredictionOutput:
    win_logits: torch.Tensor
    margin_logits: torch.Tensor

class GamePredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.win_head = nn.Linear(hidden_dim, 2)  
        self.margin_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> GamePredictionOutput:
        x = self.relu(self.fc1(x))
        win_logits = self.win_head(x)
        margin_logits = self.margin_head(x).squeeze(-1)
        return GamePredictionOutput(win_logits=win_logits, margin_logits=margin_logits)