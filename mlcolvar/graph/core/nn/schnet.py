import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from typing import Union, Optional

"""
The SchNet components. This module is taken from the pgy package:
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py
"""

__all__ = ['InteractionBlock', 'ShiftedSoftplus']


class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
        cutoff_l: float = -1.0,
        aggr: Union[str, nn.Sequential] = 'mean',
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        if cutoff_l > 0:
            self.mlp_l = nn.Sequential(
                nn.Linear(num_gaussians, num_filters),
                ShiftedSoftplus(),
                nn.Linear(num_filters, num_filters),
            )
        else:
            self.mlp_l = None
        self.conv = CFConv(
            hidden_channels,
            hidden_channels,
            num_filters,
            self.mlp,
            cutoff,
            self.mlp_l,
            cutoff_l,
            aggr
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        if self.mlp_l is not None:
            nn.init.xavier_uniform_(self.mlp_l[0].weight)
            self.mlp_l[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.mlp_l[2].weight)
            self.mlp_l[2].bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr, edge_masks_le)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        network: nn.Sequential,
        cutoff: float,
        network_l: Optional[nn.Sequential] = None,
        cutoff_l: float = -1.0,
        aggr: Union[str, nn.Sequential] = 'mean'
    ) -> None:
        super().__init__(aggr=aggr)
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.network_l = network_l
        self.network = network
        self.cutoff_l = cutoff_l
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        C = 0.5 * torch.cos(edge_weight * math.pi / self.cutoff) + 0.5
        W = self.network(edge_attr) * C.view(-1, 1)

        if edge_masks_le is not None:
            assert self.network_l is not None
            assert self.cutoff_l > self.cutoff

            C_l = 0.5 * torch.cos(edge_weight * math.pi / self.cutoff_l) + 0.5
            C_l_1 = 0.5 - 0.5 * torch.cos(edge_weight * math.pi / self.cutoff)
            C_l = C_l * (
                C_l_1 * (edge_weight < self.cutoff)  # le shorter than cutoff
                + 1.0 * (edge_weight > self.cutoff)  # le longer than cutoff
            )
            W_l = self.network_l(edge_attr) * C_l.view(-1, 1)
            W = W * ~edge_masks_le + W_l * edge_masks_le

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return x_j * W


class ShiftedSoftplus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(x) - self.shift
