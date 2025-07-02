import math
import torch
from typing import Callable, Optional

from mlcolvar.graph.utils import torch_tools


class SAKEInteraction(torch.nn.Module):
    """
    SAKE Layer, implemented based on code from
    E(n) Equivariant Convolutional Layer
    """

    def __init__(
        self,
        n_bases: int,
        cutoff: float,
        n_hidden_channels: int = 16,
        cutoff_l: float = -1.0,
        n_heads: int = 4,
        activation: Callable = torch.nn.CELU(alpha=2.0),
    ) -> None:

        super().__init__()
        self.cutoff_l = cutoff_l
        self.cutoff = cutoff
        self.n_heads = n_heads

        n_edge_out = n_hidden_channels * 2 + 1 + n_bases
        self.mlp_edge_in = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_channels * 2, n_hidden_channels),
            activation,
            torch.nn.Linear(n_hidden_channels, n_bases),
        )
        self.mlp_edge_out = torch.nn.Sequential(
            torch.nn.Linear(n_edge_out, n_hidden_channels),
            activation,
            torch.nn.Linear(n_hidden_channels, n_hidden_channels),
            activation,
        )
        self.mlp_node = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_channels * 3, n_hidden_channels),
            activation,
            torch.nn.Linear(n_hidden_channels, n_hidden_channels),
            activation,
        )
        self.mlp_semantic_atten = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_channels, n_heads),
            activation,
            torch.nn.Linear(n_heads, 1),
        )
        self.mlp_mu = torch.nn.Sequential(
            torch.nn.Linear(self.n_heads, n_hidden_channels),
            activation,
            torch.nn.Linear(n_hidden_channels, n_hidden_channels),
            activation,
        )

        self.mlp_spatial_att = torch.nn.Linear(n_hidden_channels, n_heads)
        self.mlp_rbf = torch.nn.Linear(n_bases, n_hidden_channels)
        self.mlp_spatial_att_l = (
            torch.nn.Linear(n_hidden_channels, n_heads)
            if cutoff_l > 0 else None
        )
        self.mlp_rbf_l = (
            torch.nn.Linear(n_bases, n_hidden_channels)
            if cutoff_l > 0 else None
        )

    def reset_parameters(self) -> None:
        """
        Resets all learnable parameters of the module.
        """
        torch.nn.init.xavier_uniform_(self.mlp_edge_in[0].weight)
        self.mlp_edge_in[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_edge_in[2].weight)
        self.mlp_edge_in[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_edge_out[0].weight)
        self.mlp_edge_out[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_edge_out[2].weight)
        self.mlp_edge_out[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_node[0].weight)
        self.mlp_node[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_node[2].weight)
        self.mlp_node[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_semantic_atten[0].weight)
        self.mlp_semantic_atten[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_semantic_atten[2].weight)
        self.mlp_semantic_atten[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_mu[0].weight)
        self.mlp_mu[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_mu[2].weight)
        self.mlp_mu[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_spatial_att.weight)
        self.mlp_spatial_att.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_rbf.weight)
        self.mlp_rbf.bias.data.fill_(0)
        if self.cutoff_l > 0:
            torch.nn.init.xavier_uniform_(self.mlp_spatial_att_l.weight)
            self.mlp_spatial_att_l.bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.mlp_rbf_l.weight)
            self.mlp_rbf_l.bias.data.fill_(0)

    def edge_model(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        rbf_values: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source = x[edge_index[0]]
        target = x[edge_index[1]]

        x_combined = torch.cat([source, target], dim=1)
        x_combined = self.mlp_edge_in(x_combined)

        out = torch.cat(
            [source, target, edge_lengths, rbf_values * x_combined], dim=1
        )
        out = self.mlp_edge_out(out)
        return out

    def spatial_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        edge_vectors = torch.repeat_interleave(
            edge_vectors.unsqueeze(dim=1), self.n_heads, dim=1
        )
        C = 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff) + 0.5

        atten = self.mlp_spatial_att(edge_attrs).unsqueeze(dim=2)
        atten = atten * edge_vectors
        atten = atten * C.view(-1, 1).unsqueeze(-1)

        if edge_masks_le is not None:
            edge_masks_le = edge_masks_le.unsqueeze(-1)
            assert self.mlp_spatial_att_l is not None
            assert self.cutoff_l > self.cutoff

            C_l = 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff_l) + 0.5
            C_l_1 = 0.5 - 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff)
            C_l = C_l * (
                C_l_1 * (edge_lengths < self.cutoff)
                + 1.0 * (edge_lengths > self.cutoff)
            )

            atten_l = self.mlp_spatial_att_l(edge_attrs).unsqueeze(dim=2)
            atten_l = atten_l * edge_vectors
            atten_l = atten_l * C_l.view(-1, 1).unsqueeze(-1)

            atten = atten * ~edge_masks_le + atten_l * edge_masks_le

        all_aggs = torch_tools.scatter_sum(
            atten, edge_index[0].unsqueeze(1), dim=0
        )
        out = self.mlp_mu(torch.norm(all_aggs, dim=2))

        return out

    def dist_x_semantic_atten(
        self,
        edge_lengths: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        C = 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff) + 0.5
        if edge_masks_le is not None:
            assert self.cutoff_l > self.cutoff
            C_l = 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff_l) + 0.5
            C_l_1 = 0.5 - 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff)
            C_l = C_l * (
                C_l_1 * (edge_lengths < self.cutoff)
                + 1.0 * (edge_lengths > self.cutoff)
            )
            C = C * ~edge_masks_le + C_l * edge_masks_le
        atten_semantic = self.mlp_semantic_atten(edge_attrs)
        return atten_semantic * C

    def node_model(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        atten: torch.Tensor,
    ) -> torch.Tensor:
        agg = torch_tools.scatter_sum(edge_attrs, edge_index[0], dim=0)
        agg = torch.cat([x, agg, atten], dim=1)
        out = self.mlp_node(agg)
        out = x + out
        return out

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        rbf_values: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_masks_le: Optional[torch.Tensor] = None,
    ):
        edge_attrs = self.edge_model(
            x, edge_index, edge_lengths, rbf_values, edge_masks_le
        )
        edge_attrs = edge_attrs * self.dist_x_semantic_atten(
            edge_lengths, edge_attrs, edge_masks_le
        )
        atten = self.spatial_attention(
            x,
            edge_index,
            edge_lengths,
            edge_vectors,
            edge_attrs,
            edge_masks_le
        )
        x = self.node_model(x, edge_index, edge_attrs, atten)
        return x
