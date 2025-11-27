import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj

from typing import Optional

from cfb_predictor.modeling.models.heads import GamePredictionHead, GamePredictionOutput

#TODO - Edge_weight learned parameter?
class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_confs: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        conf_emb_dim: int = 4
    ):
        super().__init__()
        self.conf_embedding = nn.Embedding(num_confs, conf_emb_dim)
        self.conv1 = pyg_nn.GCNConv(in_dim + conf_emb_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(
        self, 
        x: torch.Tensor,
        conf_ids: torch.Tensor,
        edge_index: Adj, 
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        conf_emb = self.conf_embedding(conf_ids)
        x = torch.cat([x, conf_emb], dim=-1) # [B, N, in_dim + conf_emb_dim]

        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
class GCNPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_confs: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        conf_emb_dim: int = 4,
        head_hidden_dim: int = 64
    ):
        super().__init__()
        self.encoder = GCNEncoder(
            in_dim=in_dim,
            num_confs=num_confs,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            conf_emb_dim=conf_emb_dim
        )
        # We concatenate the two team embeddings for prediction
        self.head = GamePredictionHead(input_dim=out_dim*2, hidden_dim=head_hidden_dim)

    def forward(
        self, 
        x: torch.Tensor,
        conf_ids: torch.Tensor,
        team_idxs: torch.Tensor,
        opponent_idxs: torch.Tensor,
        edge_index: Adj, 
        edge_weight: Optional[torch.Tensor] = None
    ) -> GamePredictionOutput:
        
        node_embeddings = self.encoder(
            x=x,
            conf_ids=conf_ids,
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        # Concatenate the embeddings of the two teams for prediction
        embeddings = torch.cat([node_embeddings[team_idxs], node_embeddings[opponent_idxs]], dim=-1)
        output = self.head(embeddings)
        return output