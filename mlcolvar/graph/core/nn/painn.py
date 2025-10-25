import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from typing import Union, Optional, Tuple

"""
The PaiNN components. This module is directly taken from repo:
https://github.com/MaxH1996/PaiNN-in-PyG
"""

# __all__ = ['MessagePassingPaiNN', 'UpdatePaiNN', 'AttentionGatePaiNN']


class MessagePassingPaiNN(MessagePassing):

    propagate_type = {
        'x': torch.Tensor,
        'W': torch.Tensor,
        'C': torch.Tensor,
        'edge_lengths': torch.Tensor,
        'edge_vectors': torch.Tensor,
        'flat_shape_s': int,
        'flat_shape_v': int,
    }

    def __init__(
        self,
        n_hidden_channels: int,
        n_gaussians: int,
        cutoff: float,
        cutoff_l: float = -1.0,
        aggr: Union[str, nn.Sequential] = 'mean',
    ) -> None:
        super(MessagePassingPaiNN, self).__init__(aggr=aggr)

        self.cutoff = cutoff
        self.cutoff_l = cutoff_l
        self.n_hidden_channels = n_hidden_channels

        self.lin1 = nn.Linear(n_hidden_channels, n_hidden_channels)
        self.lin2 = nn.Linear(n_hidden_channels, 3 * n_hidden_channels)
        self.silu = nn.SiLU()

        self.lin_rbf = nn.Linear(n_gaussians, 3 * n_hidden_channels)
        if cutoff_l > 0:
            self.lin_rbf_l = nn.Linear(n_gaussians, 3 * n_hidden_channels)
        else:
            self.lin_rbf_l = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_rbf.weight)
        self.lin_rbf.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_rbf.weight)
        self.lin_rbf.bias.data.fill_(0)
        if self.lin_rbf_l is not None:
            nn.init.xavier_uniform_(self.lin_rbf_l.weight)
            self.lin_rbf_l.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_rbf_l.weight)
            self.lin_rbf_l.bias.data.fill_(0)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]

        W = self.lin_rbf(edge_attr)
        C = 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff) + 0.5

        if edge_masks_le is not None:
            assert self.cutoff_l > self.cutoff
            assert self.lin_rbf_l is not None

            indices_l = edge_masks_le.nonzero()[:, 0]
            lengths_l = edge_lengths[indices_l]
            edge_attr_l = edge_attr[indices_l]

            W_l = self.lin_rbf_l(edge_attr_l)
            W = W.index_copy_(0, indices_l, W_l)

            C_l = 0.5 * torch.cos(lengths_l * math.pi / self.cutoff_l) + 0.5
            C_l_1 = 0.5 - 0.5 * torch.cos(lengths_l * math.pi / self.cutoff)
            C_l = C_l * (
                C_l_1 * (lengths_l < self.cutoff)
                + 1.0 * (lengths_l > self.cutoff)
            )
            C = C.index_copy_(0, indices_l, C_l)

        x = torch.cat([s, v], dim=-1)

        x = self.propagate(
            edge_index,
            x=x,
            W=W,
            C=C,
            edge_lengths=edge_lengths,
            edge_vectors=edge_vectors,
            flat_shape_s=flat_shape_s,
            flat_shape_v=flat_shape_v,
        )

        return x

    def message(
        self,
        x_j: torch.Tensor,
        W: torch.Tensor,
        C: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor,
        flat_shape_s: int,
        flat_shape_v: int,
    ) -> torch.Tensor:

        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)

        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)

        # Split

        left, dsm, right = torch.split(
            phi * W * C.view(-1, 1), self.n_hidden_channels, dim=-1
        )

        # v_j channel
        v_j = v_j.reshape(-1, flat_shape_v // 3, 3)
        hadamard_right = torch.einsum('ij,ik->ijk', right, edge_vectors)
        hadamard_left = torch.einsum('ijk,ij->ijk', v_j, left)
        dvm = (hadamard_left + hadamard_right).flatten(-2)

        # Prepare vector for update
        x_j = torch.cat((dsm, dvm), dim=-1)

        return x_j

    def update(
        self,
        out_aggr: torch.Tensor,
        flat_shape_s: int,
        flat_shape_v: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)

        return s_j, v_j.reshape(-1, flat_shape_v // 3, 3)


class UpdatePaiNN(torch.nn.Module):

    def __init__(self, n_hidden_channels: int) -> None:
        super(UpdatePaiNN, self).__init__()

        self.n_hidden_channels = n_hidden_channels
        self.lin1 = nn.Linear(2 * n_hidden_channels, n_hidden_channels)
        self.lin2 = nn.Linear(n_hidden_channels, 3 * n_hidden_channels)
        self.linu = nn.Linear(
            n_hidden_channels, n_hidden_channels, bias=False
        )
        self.linv = nn.Linear(
            n_hidden_channels, n_hidden_channels, bias=False
        )
        self.silu = nn.SiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linv.weight)
        nn.init.xavier_uniform_(self.linu.weight)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]

        v_u = v.reshape(-1, flat_shape_v // 3, 3)
        v_ut = torch.transpose(
            v_u, 1, 2
        )  # need transpose to get lin.comb a long feature dimension
        U = torch.transpose(self.linu(v_ut), 1, 2)
        V = torch.transpose(self.linv(v_ut), 1, 2)

        # form the dot product
        UV = torch.einsum('ijk,ijk->ij', U, V)

        # s_j channel
        nV = torch.norm(V, dim=-1)

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin1(s_u)
        s_u = self.silu(s_u)
        s_u = self.lin2(s_u)

        # final split
        top, middle, bottom = torch.split(s_u, self.n_hidden_channels, dim=-1)

        # outputs
        dvu = torch.einsum('ijk,ij->ijk', v_u, top)
        dsu = middle * UV + bottom

        return dsu, dvu.reshape(-1, flat_shape_v // 3, 3)


class AttentionGatePaiNN(nn.Module):

    def __init__(self, n_hidden_channels: int) -> None:
        super(AttentionGatePaiNN, self).__init__()

        self.n_hidden_channels = n_hidden_channels
        self.gate = nn.Sequential(
            nn.Linear(n_hidden_channels * 2, n_hidden_channels),
            nn.SiLU(),
            nn.Linear(n_hidden_channels, 1)
        )

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate[0].weight)
        self.gate[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.gate[2].weight)
        self.gate[2].bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, v = torch.split(
            x, [self.n_hidden_channels, self.n_hidden_channels * 3], dim=-1
        )
        sv = torch.cat(
            [s, torch.norm(v.reshape(-1, self.n_hidden_channels, 3), dim=-1)],
            dim=1
        )
        return self.gate(sv)
