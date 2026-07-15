from abc import ABC, abstractmethod
from functools import partial, partialmethod
from typing import Optional, List, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlcolvar.pairformer.core.nn.pairformer.utils import (
    softmax_no_cast,
    permute_final_dims,
    chunk_layer,
    flatten_final_dims,
    _attention,
    _local_attention,
    _tri_attention,
    create_local_attn_bias,
)
from mlcolvar.pairformer.core.nn.pairformer.kernel import TRITON_AVAILABLE

__all__ = ['PairformerBlock']


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) is int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        if self.r > 0 and self.training:
            shape = list(x.shape)
            if self.batch_dim is not None:
                for bd in self.batch_dim:
                    shape[bd] = 1
            mask = x.new_ones(shape)
            mask = self.dropout(mask)
            x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class PairformerLinear(nn.Linear):
    """
    Lightweight Linear wrapper that mirrors torch.nn.Linear but keeps the
    naming consistent across the Pairformer modules.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super(PairformerLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class PairformerLayerNorm(nn.Module):
    def __init__(
        self,
        c_in,
        create_scale: bool = True,
        create_offset: bool = True,
        eps=1e-5,
    ):
        super(PairformerLayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps

        if self.create_scale:
            self.weight = nn.Parameter(torch.ones(c_in))
        else:
            self.weight = None
        if self.create_offset:
            self.bias = nn.Parameter(torch.zeros(c_in))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor):
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )
        return out


class PairformerAttention(nn.Module):
    """
    Multi-head attention that supports both the AlphaFold-style triangle
    biases and the standard/local attention paths used elsewhere in Pairformer.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool = True,
        q_bias: bool = False,
        local_attention_method: Optional[str] = None,
        use_efficient_implementation: bool = False,
        zero_init_output: bool = False,
    ):
        super(PairformerAttention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.gating = gating
        self.local_attention_method = local_attention_method
        self.use_efficient_implementation = use_efficient_implementation

        self.linear_q = PairformerLinear(
            self.c_q, self.c_hidden * self.num_heads, bias=q_bias
        )
        self.linear_k = PairformerLinear(
            self.c_k, self.c_hidden * self.num_heads, bias=False
        )
        self.linear_v = PairformerLinear(
            self.c_v, self.c_hidden * self.num_heads, bias=False
        )
        self.linear_o = PairformerLinear(
            self.c_hidden * self.num_heads, self.c_q, bias=False
        )

        if zero_init_output:
            nn.init.zeros_(self.linear_o.weight)
            if self.linear_o.bias is not None:
                nn.init.zeros_(self.linear_o.bias)

        self.linear_g = None
        self.sigmoid = None
        if self.gating:
            self.linear_g = PairformerLinear(
                self.c_q, self.c_hidden * self.num_heads, bias=False
            )
            self.sigmoid = nn.Sigmoid()

    def reset_parameters(self) -> None:
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters()
        self.linear_o.reset_parameters()
        if self.gating:
            self.linear_g.reset_parameters()

    @staticmethod
    def _triangle_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        biases: List[torch.Tensor],
    ) -> torch.Tensor:
        key = permute_final_dims(key, (1, 0))
        attn = torch.matmul(query, key)
        for b in biases:
            attn += b
        attn = softmax_no_cast(attn, -1)
        attn = torch.matmul(attn, value)
        return attn

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        if apply_scale:
            q = q / math.sqrt(self.c_hidden)
        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g
        o = flatten_final_dims(o, num_dims=2)
        o = self.linear_o(o)
        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        triangle_attention: str = 'torch',
        attn_bias: Optional[torch.Tensor] = None,
        trunked_attn_bias: Optional[torch.Tensor] = None,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inf: float = 1e10,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        use_triangle_path = biases is not None or triangle_attention != 'torch'
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)

        if use_triangle_path:
            assert triangle_attention in ['torch', 'triattention']
            if triangle_attention == 'triattention':
                o = _tri_attention(q, k, v, biases)
            else:
                o = self._triangle_attention(q, k, v, biases)
                o = o.transpose(-2, -3)
            return self._wrap_up(o, q_x)

        if attn_bias is not None and len(attn_bias.shape) != len(q.shape):
            attn_bias = attn_bias.unsqueeze(dim=-3)

        if (
            trunked_attn_bias is not None
            and len(trunked_attn_bias.shape) != len(q.shape) + 1
        ):
            trunked_attn_bias = trunked_attn_bias.unsqueeze(dim=-4)

        if n_queries and n_keys:
            if self.local_attention_method == 'global_attention_with_bias':
                local_attn_bias = create_local_attn_bias(
                    q.shape[-2], n_queries, n_keys, inf=inf, device=q.device
                )
                local_attn_bias = local_attn_bias.reshape(
                    (1,) * (len(q.shape[:-2])) + local_attn_bias.shape
                )
                if attn_bias is not None:
                    if inplace_safe:
                        local_attn_bias += attn_bias
                    else:
                        local_attn_bias = local_attn_bias + attn_bias
                o = _attention(
                    q=q,
                    k=k,
                    v=v,
                    attn_bias=local_attn_bias,
                    use_efficient_implementation=self.use_efficient_implementation,
                    inplace_safe=inplace_safe,
                )
            elif self.local_attention_method == 'local_cross_attention':
                o = _local_attention(
                    q=q,
                    k=k,
                    v=v,
                    n_queries=n_queries,
                    n_keys=n_keys,
                    attn_bias=attn_bias,
                    trunked_attn_bias=trunked_attn_bias,
                    inf=inf,
                    use_efficient_implementation=self.use_efficient_implementation,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            else:
                raise ValueError(
                    f'Invalid local attention method: {self.local_attention_method}'
                )
        else:
            o = _attention(
                q=q,
                k=k,
                v=v,
                attn_bias=attn_bias,
                use_efficient_implementation=self.use_efficient_implementation,
                inplace_safe=inplace_safe,
            )

        o = o.transpose(-2, -3)
        o = self._wrap_up(o, q_x)
        return o


LinearNoBias = partial(PairformerLinear, bias=False)


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Implements Algorithms 11 and 12.
    """

    @abstractmethod
    def __init__(self, c_z, c_hidden, _outgoing):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(BaseTriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = PairformerLinear(self.c_z, self.c_z, bias=False)
        self.linear_z = PairformerLinear(self.c_hidden, self.c_z, bias=False)

        self.layer_norm_in = PairformerLayerNorm(self.c_z)
        self.layer_norm_out = PairformerLayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self) -> None:
        self.linear_g.reset_parameters()
        self.linear_z.reset_parameters()

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__(
            c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing
        )

        self.linear_a_p = PairformerLinear(self.c_z, self.c_hidden, bias=False)
        self.linear_a_g = PairformerLinear(self.c_z, self.c_hidden, bias=False)
        self.linear_b_p = PairformerLinear(self.c_z, self.c_hidden, bias=False)
        self.linear_b_g = PairformerLinear(self.c_z, self.c_hidden, bias=False)

    def reset_parameters(self) -> None:
        self.linear_a_p.reset_parameters()
        self.linear_a_g.reset_parameters()
        self.linear_b_p.reset_parameters()
        self.linear_b_g.reset_parameters()

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences.

        It works as follows. We will make reference to variables used in the
        default forward implementation below. Naively, triangle multiplication
        attention requires the manifestation of 5 tensors the size of z:
        1) z, the 'square' input tensor, 2) a, the first projection of z,
        3) b, the second projection of b, 4) g, a z-sized mask, and 5) a
        z-sized tensor for intermediate computations. For large N, this is
        prohibitively expensive; for N=4000, for example, z is more than 8GB
        alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a
        chunk of the output depend only on the tensor a and corresponding
        vertical and horizontal chunks of z. This suggests an algorithm that
        loops over pairs of chunks of z: hereafter 'columns' and 'rows' of
        z, even though each 'column' and 'row' in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing
        output chunks to a new tensor would bring total memory consumption
        down to 3x the size of z. However, more memory can be saved by writing
        output chunks directly to z in-place. WLOG, we choose to write output
        chunks vertically, overwriting the ith 'column' of z at the end of
        the ith iteration of the main loop. Despite this overwriting, the
        ith column is always one column ahead of previously overwritten columns
        and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For
        this reason, we introduce the z-cache, a tensor one-half the size of
        z. The z-cache initially contains the left half (2nd and 3rd quadrants)
        of z. For 0 < i < N/2, the missing left part of the ith row of z is
        recovered from this cache at the beginning of the ith iteration. Once i
        exceeds n/2, the cache is 'reoriented' to encompass the 3rd and 4th
        quadrants of z instead. Though the 3rd quadrant of the original z is
        entirely overwritten at this point, it can be recovered from the z-cache
        itself. Thereafter, the ith row of z can be recovered in its entirety
        from the reoriented z-cache. After the final iteration, z has been
        completely overwritten and contains the triangular multiplicative
        update. If with_add is True, it instead contains the sum of z and the
        triangular multiplicative update. In either case, peak memory
        consumption is just 2.5x the size of z, disregarding memory used for
        chunks and other small variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.weight.shape[0]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i : i + inplace_chunk_size, :, :]
                    mask_chunk = mask[..., i : i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache, z):
                # 'Reorient' the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[quadrant_3_slicer] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[z_cache_slicer])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [
                i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])
            ]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(
                i_range + after_half, initial_offsets + after_half_offsets
            )
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z,
                    i,
                    i + offset,
                    b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask,
                    i,
                    i + offset,
                    b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[z_chunk_slicer] = slice_tensor(
                            z_cache,
                            i,
                            i + offset,
                            row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(
                            z_cache,
                            z_cache_offset,
                            z_cache_offset + offset,
                            row_dim,
                        )

                b_chunk = compute_projection(
                    z_chunk_b, mask_chunk, a=False, chunked=False
                )
                del z_chunk_b

                x_chunk = torch.matmul(
                    a,
                    b_chunk,
                )
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk.sigmoid_()
                del z_chunk_g

                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if with_add:
                z += x
            else:
                z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
        triangle_multiplicative: str = 'torch',
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        _input_inplace_safe = inplace_safe is True
        if triangle_multiplicative == 'torch':
            if inplace_safe:
                x = self._inference_forward(
                    z,
                    mask,
                    inplace_chunk_size=_inplace_chunk_size,
                    with_add=_add_with_inplace,
                )
                return x

            if mask is None:
                mask = z.new_ones(z.shape[:-1])

            mask = mask.unsqueeze(-1)

            if _input_inplace_safe and _add_with_inplace:
                z_in = z.clone()

            z = self.layer_norm_in(z)
            a = mask
            a = a * self.sigmoid(self.linear_a_g(z))
            a = a * self.linear_a_p(z)
            b = mask
            b = b * self.sigmoid(self.linear_b_g(z))
            b = b * self.linear_b_p(z)

            # Prevents overflow of torch.matmul in combine projections in
            # reduced-precision modes
            x = self._combine_projections(a, b)

            del a, b
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.sigmoid(self.linear_g(z))
            x = x * g
            if _input_inplace_safe and _add_with_inplace:
                x = x + z_in
            return x
        else:
            raise ValueError(
                f'triangle_multiplicative must be "torch", but got {triangle_multiplicative}'
            )


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = PairformerLayerNorm(self.c_in)

        self.linear = PairformerLinear(c_in, self.no_heads, bias=False)

        self.mha = PairformerAttention(
            c_q=self.c_in,
            c_k=self.c_in,
            c_v=self.c_in,
            c_hidden=self.c_hidden,
            num_heads=self.no_heads,
            gating=True,
            q_bias=False,
            zero_init_output=True,
        )

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()
        self.mha.reset_parameters()

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        triangle_attention: str = 'torch',
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        'triangle! triangle!'
        mha_inputs = {
            'q_x': x,
            'kv_x': x,
            'biases': biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                triangle_attention=triangle_attention,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        triangle_attention: str = 'torch',
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                triangle_attention=triangle_attention,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """

    __init__ = partialmethod(
        TriangleMultiplicativeUpdate.__init__, _outgoing=True
    )


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """

    __init__ = partialmethod(
        TriangleMultiplicativeUpdate.__init__, _outgoing=False
    )


class Transition(nn.Module):
    """
    Implements Algorithm 11 in AF3
    """

    def __init__(self, c_in: int, n: int) -> None:
        """
        Args:
            c_in (int, optional): the input dimension.
            n (int, optional): factor by which c_in is multiplied to obtain hidden dimension.
        """
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.layernorm1 = PairformerLayerNorm(c_in)
        self.linear_no_bias_a = LinearNoBias(
            in_features=c_in, out_features=n * c_in
        )
        self.linear_no_bias_b = LinearNoBias(
            in_features=c_in, out_features=n * c_in
        )
        self.linear_no_bias = LinearNoBias(
            in_features=n * c_in, out_features=c_in
        )

    def reset_parameters(self) -> None:
        self.linear_no_bias_a.reset_parameters()
        self.linear_no_bias_b.reset_parameters()
        self.linear_no_bias.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        if self.training:
            x = self.layernorm1(x)
            a = self.linear_no_bias_a(x)
            b = self.linear_no_bias_b(x)
            x = self.linear_no_bias(F.silu(a) * b)
            return x
        else:
            other_dims = x.shape[:-1]
            dim_size = x.shape[-1]
            size = x.shape[-2]
            x = x.reshape(-1, dim_size)
            chunk_num = 1 if size < 3200 else 8
            chunks = torch.chunk(x, chunk_num, dim=-2)
            outputs = torch.empty(
                (x.shape[0], self.c_in), dtype=x.dtype, device=x.device
            )
            start = 0
            for chunk in chunks:
                y = self.layernorm1(chunk)
                a = self.linear_no_bias_a(y)
                a = F.silu(a, True)
                b = self.linear_no_bias_b(y)
                del y
                b *= a
                del a
                b = self.linear_no_bias(b)
                outputs[start : start + b.shape[0]] = b
                start += b.shape[0]
                del b
            outputs = outputs.reshape(*other_dims, self.c_in)
            return outputs


class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3
    """

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = PairformerLayerNorm(
            c_a, create_scale=False, create_offset=False
        )
        self.layernorm_s = PairformerLayerNorm(c_s, create_offset=False)
        self.linear_s = PairformerLinear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def reset_parameters(self) -> None:
        self.linear_s.reset_parameters()
        self.linear_nobias_s.reset_parameters()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


class BiasInitLinear(PairformerLinear):
    """Support biasinit for nn.Linear Called just like torch.nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        biasinit: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Args:
            in_features (int): in_features
            out_features (int): out_features
            bias (bool, optional): whether add bias. Defaults to True.
            biasinit (float, optional): the initial bias value. Defaults to 0.0.
        """
        super(BiasInitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **kwargs,
        )
        nn.init.zeros_(tensor=self.weight)
        if bias:
            nn.init.constant_(tensor=self.bias, val=biasinit)

    def reset_parameters(self) -> None:
        pass


class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        """
        Args:
            has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
            create_offset_ln_z (bool, optional): the value of create_offset for the LayerNorm applied to z. Defaults to False.
            n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
            cross_attention_mode (bool, optional): If cross_attention_model = True, the adaptive layernorm will be applied
                to query and key/value seperately.
        """
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z
        self.cross_attention_mode = cross_attention_mode
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if self.cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = PairformerLayerNorm(c_a)
            if self.cross_attention_mode:
                self.layernorm_kv = PairformerLayerNorm(c_a)

        # Line 6-11
        self.local_attention_method = 'local_cross_attention'
        self.attention = PairformerAttention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_bias=True,
            local_attention_method=self.local_attention_method,
            zero_init_output=not self.has_s,
        )
        self.layernorm_z = PairformerLayerNorm(
            c_z, create_offset=self.create_offset_ln_z
        )
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(
            in_features=c_z, out_features=n_heads
        )

        # Line 13
        if self.has_s:
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    def reset_parameters(self) -> None:
        self.attention.reset_parameters()
        self.linear_nobias_z.reset_parameters()

    def local_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            q (torch.Tensor): query embedding
                [..., N_atom, c_a]
            kv (torch.Tensor): key/value embedding
                [..., N_atom, c_a]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(q.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(
            q_x=q,
            kv_x=kv,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return q

    def standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            q (torch.Tensor): the query embedding
                [..., N_token, c_a]
            kv (torch.Tensor): the key/value embedding
                [..., N_token, c_a]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """

        # Multi-head attention bias
        if enable_efficient_fusion:
            weight = (
                self.linear_nobias_z.weight * self.layernorm_z.weight[None, :]
            )[:, :, None, None]
            bias = F.conv2d(z, weight)
        else:
            bias = self.linear_nobias_z(self.layernorm_z(z))
            bias = permute_final_dims(
                bias, [2, 0, 1]
            )  # [..., n_heads, N_token, N_token]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(
            q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe
        )

        return q

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a=a, s=s)
            else:
                kv = self.layernorm_kv(a)
        else:
            kv = None

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a


# =========================Pairformer Block=========================
class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    c_hidden_mul is set as openfold
    Ref to:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123
    """

    def __init__(
        self,
        n_heads: int = 1,
        c_z: int = 8,
        c_s: int = 0,
        c_hidden_mul: int = 16,
        c_hidden_pair_att: int = 16,
        no_heads_pair: int = 1,
        dropout: float = 0.1,
        triangle_multiplicative: Optional[str] = 'torch',
        triangle_attention: Optional[str] = 'triattention',
        pair_transition: bool = True,
    ) -> None:
        """
        Args:
            n_heads (int, optional): number of head [for AttentionPairBias].
            c_z (int, optional): hidden dim [for pair embedding].
            c_s (int, optional):  hidden dim [for single embedding].
            c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
            c_hidden_pair_att (int, optional): hidden dim [for TriangleAttention].
            no_heads_pair (int, optional): number of head [for TriangleAttention].
            dropout (float, optional): dropout ratio [for TriangleUpdate].
            triangle_multiplicative: Triangle multiplicative implementation type.
                - 'torch' (default): PyTorch native implementation
                - None: Disable triangle update
            triangle_attention: Triangle attention implementation type.
                - 'triattention' (default) : Optimized tri-attention module
                - 'torch': PyTorch native implementation
        """
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        if triangle_multiplicative is not None:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z=c_z, c_hidden=c_hidden_mul
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z=c_z, c_hidden=c_hidden_mul
            )
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(dropout)
        if pair_transition:
            self.pair_transition = Transition(c_in=c_z, n=4)
        else:
            self.pair_transition = None
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False,
                create_offset_ln_z=True,
                n_heads=n_heads,
                c_a=c_s,
                c_z=c_z,
            )
            self.single_transition = Transition(c_in=c_s, n=4)
        self._triangle_multiplicative = triangle_multiplicative
        self._triangle_attention = triangle_attention
        assert (
            (triangle_attention is not None)
            or (triangle_multiplicative is not None)
        ), (
            'Options `triangle_attention` and `triangle_multiplicative` can '
            'not be None at the same time!'
        )

    def reset_parameters(self) -> None:
        if self._triangle_multiplicative is not None:
            self.tri_mul_in.reset_parameters()
            self.tri_mul_out.reset_parameters()
        self.tri_att_start.reset_parameters()
        self.tri_att_end.reset_parameters()
        if self.pair_transition is not None:
            self.pair_transition.reset_parameters()
        if self.c_s > 0:
            self.attention_pair_bias.reset_parameters()
            self.single_transition.reset_parameters()

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s[Optional] and z
                [..., N_token, c_s] | None
                [..., N_token, N_token, c_z]
        """
        if inplace_safe:  # z: [N_token, N_token, c_z] input
            if self._triangle_multiplicative is not None:
                z = self.tri_mul_out(
                    z,
                    mask=pair_mask,
                    inplace_safe=inplace_safe,
                    _add_with_inplace=True,
                    triangle_multiplicative=self._triangle_multiplicative,
                )  # [N_token, N_token, c_z] step1
                z = self.tri_mul_in(
                    z,
                    mask=pair_mask,
                    inplace_safe=inplace_safe,
                    _add_with_inplace=True,
                    triangle_multiplicative=self._triangle_multiplicative,
                )  # [N_token, N_token, c_z] step2
            if self._triangle_attention is not None:
                z += self.tri_att_start(
                    z,
                    mask=pair_mask,
                    triangle_attention=self._triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )  # [N_token, N_token, c_z] step3
                z = z.transpose(
                    -2, -3
                ).contiguous()  # [N_token, N_token, c_z] step4
                z += self.tri_att_end(
                    z,
                    mask=(
                        pair_mask.transpose(-1, -2)
                        if pair_mask is not None
                        else None
                    ),
                    triangle_attention=self._triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )  # [N_token, N_token, c_z] step5
                z = z.transpose(
                    -2, -3
                ).contiguous()  # [N_token, N_token, c_z] step6
                if self.pair_transition is not None:
                    z += self.pair_transition(z)  # [N_token, N_token, c_z] step7
        else:
            if self._triangle_multiplicative is not None:
                tmu_update = self.tri_mul_out(
                    z,
                    mask=pair_mask,
                    inplace_safe=inplace_safe,
                    _add_with_inplace=False,
                    triangle_multiplicative=self._triangle_multiplicative,
                )
                z = z + self.dropout_row(tmu_update)
                del tmu_update
                tmu_update = self.tri_mul_in(
                    z,
                    mask=pair_mask,
                    inplace_safe=inplace_safe,
                    _add_with_inplace=False,
                    triangle_multiplicative=self._triangle_multiplicative,
                )
                z = z + self.dropout_row(tmu_update)
                del tmu_update
            if self._triangle_attention is not None:
                z = z + self.dropout_row(
                    self.tri_att_start(
                        z,
                        mask=pair_mask,
                        triangle_attention=self._triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                )
                z = z.transpose(-2, -3)
                z = z + self.dropout_row(
                    self.tri_att_end(
                        z,
                        mask=(
                            pair_mask.transpose(-1, -2)
                            if pair_mask is not None
                            else None
                        ),
                        triangle_attention=self._triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                )
                z = z.transpose(-2, -3)

                if self.pair_transition is not None:
                    z = z + self.pair_transition(z)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=z,
            )
            s = s + self.single_transition(s)
        return s, z


def test_pairformer_block() -> None:
    torch.manual_seed(0)

    def build_inputs(
        batch: int, num_atoms: int, feature_dim: int, dtype=torch.float64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        single = torch.randn(batch, num_atoms, feature_dim)
        mask = torch.ones(batch, num_atoms, num_atoms)
        pair = []
        for idx in range(feature_dim):
            channel = single[..., idx].unsqueeze(-1)
            pair.append(torch.cdist(channel, channel))
        pair = torch.stack(pair, dim=-1)
        return single.to(dtype), pair.to(dtype), mask.to(dtype)

    pairformer = PairformerBlock(
        n_heads=1,
        c_z=2,
        c_s=0,
        c_hidden_mul=16,
        c_hidden_pair_att=16,
        no_heads_pair=1,
        triangle_attention='torch',
    ).eval().to(torch.float64)

    single, pair, mask = build_inputs(2, 2, 2)
    _, outputs = pairformer(s=None, z=pair.clone(), pair_mask=mask)

    assert ((
        outputs - torch.tensor(
            [[[
                [-0.05670540173976374, 0.7206232278741938],
                [3.156071279509658, 0.03987509164800429],
            ], [
                [3.156071279509658, 0.03987509164800429],
                [-0.05670540173976374, 0.7206232278741938],
            ]], [[
                [-0.021746629553362817, 0.4317078516107618],
                [1.788341532105362, 1.8598599940061387],
            ], [
                [1.788341532105362, 1.8598599940061387],
                [-0.021746629553362817, 0.4317078516107618],
            ]]], dtype=torch.float64,
        )
    ) < 1E-14).all()

    if TRITON_AVAILABLE:

        import triton
        import packaging.version as pv

        if pv.parse(triton.__version__) > pv.parse('3.5.1'):
            dtype = torch.float64
            tol = 1E-10
        else:
            dtype = torch.float32
            tol = 1E-6

        pairformer_1 = PairformerBlock(
            n_heads=1,
            c_z=64,
            c_s=0,
            c_hidden_mul=64,
            c_hidden_pair_att=64,
            no_heads_pair=1,
            triangle_attention='triattention',
        ).eval().to(dtype).to('cuda')

        pairformer_2 = PairformerBlock(
            n_heads=1,
            c_z=64,
            c_s=0,
            c_hidden_mul=64,
            c_hidden_pair_att=64,
            no_heads_pair=1,
            triangle_attention='torch',
        ).eval().to(dtype).to('cuda')

        single, pair, mask = build_inputs(2, 64, 64, dtype)
        _, outputs_1 = pairformer_1(
            s=None, z=pair.clone().to('cuda'), pair_mask=mask.to('cuda')
        )
        _, outputs_2 = pairformer_2(
            s=None, z=pair.clone().to('cuda'), pair_mask=mask.to('cuda')
        )

        assert ((outputs_1 - outputs_1) < tol).all()


if __name__ == '__main__':
    test_pairformer_block()
