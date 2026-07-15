import math
import optree
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Union, Any, List, Callable, Optional, Sequence, Tuple, Dict

BLOCK_ARG = Any
BLOCK_ARGS = List[BLOCK_ARG]


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    return torch.nn.functional.softmax(t, dim=dim)


def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def _fetch_dims(tree: Any) -> List[Tuple[int, ...]]:
    return optree.tree_flatten(optree.tree_map(lambda x: x.shape, tree))[0]


@torch.jit.ignore
def _flat_idx_to_idx(
    flat_idx: int,
    dims: Tuple[int],
) -> Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d

    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: int,
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> Sequence[Tuple[int]]:
    """
    Produces an ordered sequence of tensor slices that, when used in
    sequence on a tensor with shape dims, yields tensors that contain every
    leaf in the contiguous range [start, end]. Care is taken to yield a
    short sequence of slices, and perhaps even the shortest possible (I'm
    pretty sure it's the latter).

    end is INCLUSIVE.
    """

    # start_edges and end_edges both indicate whether, starting from any given
    # dimension, the start/end index is at the top/bottom edge of the
    # corresponding tensor, modeled as a tree
    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]

    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)

    # Base cases. Either start/end are empty and we're done, or the final,
    # one-dimensional tensor can be simply sliced
    if len(start) == 0:
        return [tuple()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    slices = []
    path = []

    # Dimensions common to start and end can be selected directly
    for s, e in zip(start, end):
        if s == e:
            path.append(slice(s, s + 1))
        else:
            break

    path = tuple(path)
    divergence_idx = len(path)

    # start == end, and we're done
    if divergence_idx == len(dims):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [
            path + (slice(sdi, sdi + 1),) + s
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1 :],
                [d - 1 for d in dims[divergence_idx + 1 :]],
                dims[divergence_idx + 1 :],
                start_edges=start_edges[divergence_idx + 1 :],
                end_edges=[1 for _ in end_edges[divergence_idx + 1 :]],
            )
        ]

    def lower():
        edi = end[divergence_idx]
        return [
            path + (slice(edi, edi + 1),) + s
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1 :]],
                end[divergence_idx + 1 :],
                dims[divergence_idx + 1 :],
                start_edges=[1 for _ in start_edges[divergence_idx + 1 :]],
                end_edges=end_edges[divergence_idx + 1 :],
            )
        ]

    # If both start and end are at the edges of the subtree rooted at
    # divergence_idx, we can just select the whole subtree at once
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(
            path + (slice(start[divergence_idx], end[divergence_idx] + 1),)
        )
    # If just start is at the edge, we can grab almost all of the subtree,
    # treating only the ragged bottom edge as an edge case
    elif start_edges[divergence_idx]:
        slices.append(
            path + (slice(start[divergence_idx], end[divergence_idx]),)
        )
        slices.extend(lower())
    # Analogous to the previous case, but the top is ragged this time
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(
            path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),)
        )
    # If both sides of the range are ragged, we need to handle both sides
    # separately. If there's contiguous meat in between them, we can index it
    # in one big chunk
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(
                path + (slice(start[divergence_idx] + 1, end[divergence_idx]),)
            )
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
    t: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be
    memory-intensive in certain situations. The only reshape operations
    in this function are performed on sub-tensors that scale with
    (flat_end - flat_start), the chunk size.
    """

    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set is inclusive
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # Get an ordered list of slices to perform
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    sliced_tensors = [t[s] for s in slices]

    return torch.cat(
        [s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors]
    )


# With tree_map, a poor man's JAX tree_map
def dict_map(
    fn: Callable[[Any], Any], dic: Dict[Any, Any], leaf_type: type
) -> Dict[Any, Any]:
    new_dict: Dict[Any, Any] = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn: Callable[[Any], Any], tree: Any, leaf_type: type) -> Any:
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError('Not supported')


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def chunk_layer(
    layer: Callable,
    inputs: dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    """
    Implements the 'chunking' procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple 'pytrees,'
    consisting only of (arbitrarily nested) lists, tuples, and dicts with
    torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must
            be tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a 'sub-batch' is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary
            in most cases, and is ever so slightly slower than the default
            setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not (len(inputs) > 0):
        raise ValueError('Must provide at least one input')

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t):
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        reshape_fn = lambda t: t.view([-1] + list(t.shape[no_batch_dims:]))
        prepped_outputs = tensor_tree_map(reshape_fn, _out)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (
        flat_batch_dim % chunk_size != 0
    )

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        # Chunk the input
        if not low_mem:
            select_chunk = lambda t: (
                t[i : i + chunk_size] if t.shape[0] != 1 else t
            )
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:

            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError('Not supported')

        i += chunk_size

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    use_efficient_implementation: bool = False,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Attention.

    Args:
        q (torch.Tensor): query tensor of shape [..., n_q, d]
        k (torch.Tensor): key tensor of shape [..., n_kv, d]
        v (torch.Tensor): value tensor of shape[..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias tensor of shape [..., n_q, n_kv]. Defaults to None.
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to False.

    Returns:
        torch.Tensor: output of tensor [..., n_q, d]
    """
    assert k.shape == v.shape

    # Upcast to compute attn_weights
    # q = q.to(dtype=torch.float32)
    # k = k.to(dtype=torch.float32)
    # if attn_bias is not None:
    #     attn_bias = attn_bias.to(dtype=torch.float32)

    if use_efficient_implementation:
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
            scale=1.0,
        )
        return attn_output

    # [..., n_kv, d] -> [..., d, n_kv]
    k = k.transpose(-1, -2)

    # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
    attn_weights = q @ k

    if attn_bias is not None:
        if inplace_safe:
            attn_weights += attn_bias
        else:
            attn_weights = attn_weights + attn_bias

    # [..., n_q, n_kv]
    attn_weights = F.softmax(attn_weights, dim=-1)

    # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
    attn_output = attn_weights @ v

    return attn_output


def rearrange_qk_to_dense_trunk(
    q: Union[torch.Tensor, List[torch.Tensor]],
    k: Union[torch.Tensor, List[torch.Tensor]],
    dim_q: Union[int, List[int]],
    dim_k: Union[int, List[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> Tuple[
    Union[torch.Tensor, List[torch.Tensor]],
    Union[torch.Tensor, List[torch.Tensor]],
    Dict[str, Any],
]:
    """Rearrange q/k into blocked tensors for local operations.

    Args:
        q (torch.Tensor): query tensor. Could be a tensor or a list of tensors.
            [..., n_q, ...] (n_q is at dimension dim_q)
        k (torch.Tensor | List[torch.Tensor]): key tensor. Could be a tensor or a list of tensors.
            [..., n_k, ...] (n_k is at dimension dim_k)
        dim_q (int): along which dimension to build the trunks. Could be an int or a list of int.
        dim_k (int): along which dimension to build the trunks. Could be an int or a list of int.
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.

    Returns:
        tuple[Union[torch.Tensor, list[torch.Tensor]]]:
            q_trunked: torch.Tensor or list of tensors. Same as the input type.
                [..., n_trunks, n_queries, ...]
            k_trunked: torch.Tensor or list of tensors. Same as the input type.
                [..., n_trunks, n_keys, ...]
            padding_info (dict):
                mask_trunked: torch.Tensor
                    [n_trunks, n_queries, n_keys]
                q_pad: query padded dimension
    """

    assert n_keys >= n_queries
    assert n_queries & 0x01 == 0
    assert n_keys & 0x01 == 0

    def basic_checks(x, dim_x):
        if isinstance(x, list):
            x_is_list = True
            assert isinstance(dim_x, list)
        else:
            x_is_list = False
            x = [x]
            dim_x = [dim_x]
        n_x = x[0].size(dim_x[0])
        for i in range(len(dim_x)):
            if dim_x[i] < 0:
                dim_x[i] = len(x[i].shape) + dim_x[i]
            assert x[i].size(dim_x[i]) == n_x
        return x, dim_x, x_is_list, n_x, len(x)

    q, dim_q, q_is_list, n, num_q = basic_checks(q, dim_q)
    k, dim_k, k_is_list, n_k, num_k = basic_checks(k, dim_k)

    assert n == n_k
    n_trunks = int(math.ceil(n / n_queries))
    q_pad_length = n_trunks * n_queries - n

    q_new = [
        pad_at_dim(q[i], dim=dim_q[i], pad_length=(0, q_pad_length))
        for i in range(num_q)
    ]
    q_trunked = [
        reshape_at_dim(
            q_new[i], dim=dim_q[i], target_shape=(n_trunks, n_queries)
        )
        for i in range(num_q)
    ]

    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)

    k_new = [
        pad_at_dim(k[i], dim=dim_k[i], pad_length=(pad_left, pad_right))
        for i in range(num_k)
    ]
    k_trunked = [
        k_new[i].unfold(dim_k[i], size=n_keys, step=n_queries)
        for i in range(num_k)
    ]
    k_trunked = [
        move_final_dim_to_dim(k_trunked[i], dim=dim_k[i] + 1)
        for i in range(num_k)
    ]

    if compute_mask:
        pad_mask = q[0].new_ones(
            *(1,) * len(q[0].shape[:-2]),
            n + q_pad_length,
            n + pad_left + pad_right,
            requires_grad=False,
        )
        pad_mask[..., :n, :pad_left] = 0
        pad_mask[..., :n, (pad_left + n) :] = 0
        pad_mask[..., n:, :] = 0

        concat_split_data = optimized_concat_split(pad_mask, n_queries)
        pad_mask_trunked = (
            concat_split_data.unfold(
                -1, n_keys, pad_mask.size(-1) + n_queries
            ).transpose(-2, -3)
        ).bool()
    else:
        pad_mask_trunked = None

    if not q_is_list:
        q_trunked = q_trunked[0]
    if not k_is_list:
        k_trunked = k_trunked[0]

    padding_info = {
        'mask_trunked': pad_mask_trunked,
        'q_pad': q_pad_length,
        'k_pad_left': pad_left,
        'k_pad_right': pad_right,
    }

    return q_trunked, k_trunked, padding_info


def optimized_concat_split(
    attn_bias: torch.Tensor, n_queries: int
) -> torch.Tensor:
    """Optimized concatenation and splitting of attention bias tensor.

    Args:
        attn_bias (torch.Tensor): The attention bias tensor.
            Shape: [..., D, E]
        n_queries (int): The number of queries in each split.

    Returns:
        torch.Tensor: The reshaped and permuted attention bias tensor.
            Shape: [..., n_queries, D // n_queries * E]
    """
    D = attn_bias.size(-2)
    E = attn_bias.size(-1)
    assert D % n_queries == 0
    num_splits = D // n_queries
    reshaped = attn_bias.reshape(
        *attn_bias.shape[:-2], num_splits, n_queries, E
    )
    permuted = reshaped.permute(*range(reshaped.dim() - 3), -2, -3, -1)
    output = permuted.reshape(*attn_bias.shape[:-2], n_queries, num_splits * E)
    return output


def rearrange_to_dense_trunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Rearrange q/k/v/bias into blocked tensors for local attention.

    Args:
        q (torch.Tensor): query tensor
            [..., n_q, d]
        k (torch.Tensor): key tensor
            [..., n_kv, d]
        v (torch.Tensor): value tensor
            [..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias
            [..., n_q, n_kv] or None
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.
        inf (float, optional): used for attention masking. Defaults to 1e10.

    Returns:
        tuple[Union[torch.Tensor, int]]:
            q_trunked
                [..., n_trunks, n_queries, d]
            k_trunked / v_trunked
                [..., n_trunks, n_keys, d]
            attn_bias_trunked:  padded position filled with -inf
                [..., n_trunks, n_queries, n_keys]
            q_pad_length: query padded dimension
    """
    assert n_keys >= n_queries
    assert n_queries & 0x01 == 0
    assert n_keys & 0x01 == 0

    n, d = q.shape[-2:]

    q_trunked, kv_trunked, padding_info = rearrange_qk_to_dense_trunk(
        q=q,
        k=[k, v],
        dim_q=-2,
        dim_k=[-2, -2],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=False,
    )
    q_pad_length, pad_left, pad_right = (
        padding_info['q_pad'],
        padding_info['k_pad_left'],
        padding_info['k_pad_right'],
    )

    # Padded_width = n + pad_left + pad_right
    if attn_bias is None:
        attn_bias = q.new_zeros(
            *(1,) * len(q.shape[:-2]),
            n + q_pad_length,
            n + pad_left + pad_right,
        )
        attn_bias[..., :n, 0:pad_left] = -inf
        attn_bias[..., :n, pad_left + n : :] = -inf
        attn_bias[..., n::, :] = -inf
    else:
        attn_bias = F.pad(
            attn_bias, (pad_left, pad_right, 0, q_pad_length), value=-inf
        )

    concat_split_data = optimized_concat_split(attn_bias, n_queries)
    attn_bias_trunked = concat_split_data.unfold(
        -1, n_keys, attn_bias.shape[-1] + n_queries
    ).transpose(-2, -3)
    return (
        q_trunked,
        kv_trunked[0],
        kv_trunked[1],
        attn_bias_trunked,
        q_pad_length,
    )


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    trunked_attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
    use_efficient_implementation: bool = False,
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Local attention

    Args:
        q (torch.Tensor): query tensor
            [..., Q, d]
        k (torch.Tensor): key tensor
            [..., K, d]
        v (torch.Tensor): value tensor
            [..., K, d]
        n_queries (int): local window size of query.
        n_keys (int): local window size of key/value.
        attn_bias (torch.Tensor, optional): the input biases for attention. Defaults to None.
            [..., Q, K]
        trunked_attn_bias (torch.Tensor, optional): the input biases where shape has been rearranged to dense trunks. Defaults to None.
            [..., n_trunks, n_queries, n_keys]
        inf (float): inf number used for attention bias. Defaults to 1e10.
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to False.
    Returns:
        torch.Tensor: standard attention output
            [..., Q, d]
    """
    assert (
        q.shape == k.shape == v.shape
    )  # local attention doesn't make sense if Q != K

    # Prepare for attention qkv, q: [..., n_trunks, n_queries, d], kv: [..., n_trunks, n_keys, d]

    # Rerrange to dense trunks
    # q: [*, n, d] -> [*, n_trunks, n_queries, d]
    # kv: [*, n, d] -> [*, n_trunks, n_keys, d]
    # attn_bias: [*, n, d] -> [*, n_trunks, n_queries, n_keys]
    q_trunked, k_trunked, v_trunked, attn_bias_trunked, q_pad_length = (
        rearrange_to_dense_trunk(
            q=q,
            k=k,
            v=v,
            n_queries=n_queries,
            n_keys=n_keys,
            attn_bias=attn_bias,
            inf=inf,
        )
    )

    # Apply attention
    # [..., n_trunks, n_queries, d]
    if trunked_attn_bias is not None:
        attn_bias_trunked = attn_bias_trunked + trunked_attn_bias

    if chunk_size is not None:
        attn_inputs = {
            'q': q_trunked,
            'k': k_trunked,
            'v': v_trunked,
            'attn_bias': attn_bias_trunked,
        }
        out = chunk_layer(
            partial(
                _attention,
                use_efficient_implementation=use_efficient_implementation,
                inplace_safe=inplace_safe,
            ),
            attn_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(attn_bias_trunked.shape[:-2]),
            _out=None,
        )
    else:
        out = _attention(
            q=q_trunked,
            k=k_trunked,
            v=v_trunked,
            attn_bias=attn_bias_trunked,
            use_efficient_implementation=use_efficient_implementation,
            inplace_safe=inplace_safe,
        )

    # Revert back to orignal shape and remove q_pad_length
    # [..., n_trunks, n_queries, d] ->  [..., n_trunks * n_queries, d] ->  [..., n, d]
    out = out.reshape(*out.shape[:-3], -1, out.shape[-1])
    if q_pad_length > 0:
        out = out[..., :-q_pad_length, :]
    return out


@torch.jit.ignore
def _tri_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
):
    """Compute attention using the custom TriAttention kernel."""
    from .kernel import tri_attention_run

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 2:
            return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]

    o = tri_attention_run(q, k, v, biases[0], biases[1])
    o = o.reshape(orig_shape)

    return o


def create_local_attn_bias(
    n: int,
    n_queries: int,
    n_keys: int,
    inf: float = 1e10,
    device: torch.device = None,
) -> torch.Tensor:
    """Create local attention bias based on query window n_queries and kv window n_keys.

    Args:
        n (int): the length of quiries
        n_queries (int): window size of quiries
        n_keys (int): window size of keys/values
        inf (float, optional): the inf to mask attention. Defaults to 1e10.
        device (torch.device, optional): cuda|cpu|None. Defaults to None.

    Returns:
        torch.Tensor: the diagonal-like global attention bias
    """
    n_trunks = int(math.ceil(n / n_queries))
    padded_n = n_trunks * n_queries
    attn_mask = torch.zeros(padded_n, padded_n, device=device)
    for block_index in range(0, n_trunks):
        i = block_index * n_queries
        j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
        j2 = n_queries * block_index + (n_queries + n_keys) // 2
        attn_mask[i : i + n_queries, j1:j2] = 1.0
    attn_bias = (1 - attn_mask) * -inf
    return attn_bias.to(device=device)[:n, :n]


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - self.shift
