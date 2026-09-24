from abc import abstractmethod
import torch
import torch.nn as nn
class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                node_feature0: torch.tensor,
                node_feature1: torch.tensor,
                node_feature2: torch.tensor,
                cross_atlas_adjacency_weight) -> torch.tensor:
        pass
