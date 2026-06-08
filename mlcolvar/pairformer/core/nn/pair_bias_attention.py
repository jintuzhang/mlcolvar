from typing import Optional

import torch
from torch import nn

import einops
import einops._torch_specific

einops._torch_specific.allow_ops_in_compiled_graph()


def exists(val) -> bool:
    """
    returns whether val is not none
    """
    return val is not None


def default(x, y):
    """
    returns x if it exists, otherwise y
    """
    return x if exists(x) else y


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    This implementation is taken from:
    https://github.com/NVIDIA-BioNeMo/la-proteina/blob/main/proteinfoundation/nn/modules/pair_bias_attn.py
    which was originally modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.to_qkv.weight)
        self.to_qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.to_g.weight)
        self.to_g.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.to_out_node.weight)
        self.to_out_node.bias.data.fill_(0)
        if self.to_bias is not None:
            nn.init.xavier_uniform_(self.to_bias.weight)

    def forward(
        self,
        node_feats: torch.Tensor,
        pair_feats: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = (
            einops.rearrange(self.to_bias(pair_feats), 'b ... h -> b h ...')
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: einops.rearrange(
                t, 'b ... (h d) -> b h ... d', h=h
            ), (
                q, k, v, g
            )
        )
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = einops.rearrange(
            torch.sigmoid(g) * attn_feats, 'b h n d -> b n (h d)', h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform attention update
        """
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if exists(mask):
            mask = einops.rearrange(mask, 'b i j -> b () i j')
            sim = sim.masked_fill(~mask, torch.finfo(sim.dtype).min)
        attn = torch.softmax(sim + b, dim=-1)
        return torch.einsum('b h i j, b h j d -> b h i d', attn, v)
