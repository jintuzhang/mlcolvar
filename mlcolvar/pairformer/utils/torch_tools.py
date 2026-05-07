import torch
from typing import Tuple, Optional

"""
Helper functions for torch. These modules are taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/tools/torch_tools.py
https://github.com/ACEsuit/mace/blob/main/mace/tools/scatter.py
"""

__all__ = [
    'set_default_dtype',
    'get_centers',
    'get_distances',
    'scatter_sum',
    'scatter_mean',
]


def get_centers(
    positions: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """
    Get centers of a group of atoms in each batch.

    Parameters
    ----------
    position: torch.Tensor (shape: [n_atoms, 3])
        The position vector.
    indices: torch.Tensor (shape: [n_graphs, n_groups, n_atoms_in_group])
        Indices of the atoms of the group. Each group should be padded to the
        same size using negative numbers.

    Notes
    -----
    This function does not consider the PBC, which means, atoms inside the
    group should not be splitted by the simulation box.
    """
    device = positions.device
    n_graphs = indices.shape[0]
    indices = indices.clone()
    indices_all = indices.to(device).flatten(1)
    mask = indices >= 0
    mask_all = indices_all >= 0
    indices_all[~mask_all] = 0

    indices_all = indices_all.unsqueeze(-1).expand(-1, -1, 3)
    positions = positions.reshape((n_graphs, len(positions) // n_graphs, 3))
    positions_group = torch.gather(positions, dim=1, index=indices_all)
    positions_group = positions_group * mask_all.unsqueeze(-1)
    positions_group = positions_group.reshape(
        n_graphs, indices.shape[1], indices.shape[2], 3
    )

    centers = positions_group.sum(dim=2) / mask.sum(dim=2).unsqueeze(-1)
    centers = centers.reshape(n_graphs * indices.shape[1], 3)

    return centers


def get_distances(
    positions_1: torch.Tensor,
    positions_2: torch.Tensor,
    cells: torch.Tensor,
    n_graphs: int,
    normalize: bool = True,
    eps: float = 0.0,
    is_orthogonal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    s_1 = positions_1.shape[0] // n_graphs
    s_2 = positions_2.shape[0] // n_graphs

    positions_1_ = torch.repeat_interleave(
        positions_1, s_2, dim=0
    )
    positions_2_ = torch.repeat_interleave(
        positions_2.reshape(n_graphs, s_2, 3), s_1, dim=0
    ).reshape(
        s_1 * s_2 * n_graphs, 3
    )

    vectors = positions_2_ - positions_1_

    if cells.shape[1] == 3:

        cells = cells.reshape(n_graphs, 3, 3)
        cells_nodes = torch.repeat_interleave(
            cells.transpose(2, 1), s_1 * s_2, dim=0
        )

        if is_orthogonal:
            reciprocal = 1.0 / torch.diagonal(cells, dim1=-2, dim2=-1)
            cells_inv = torch.diag_embed(reciprocal)
        else:
            cells_inv = torch.linalg.pinv(cells.transpose(2, 1))
        cells_inv_nodes = torch.repeat_interleave(
            cells_inv, s_1 * s_2, dim=0
        )

        positions_1_s = torch.einsum(
            'bi,bij->bj', positions_1_, cells_inv_nodes
        )
        positions_2_s = torch.einsum(
            'bi,bij->bj', positions_2_, cells_inv_nodes
        )
        deltas = positions_1_s - positions_2_s
        unit_shifts = torch.round(deltas)
        shifts = torch.einsum('bi,bij->bj', unit_shifts, cells_nodes)

        vectors = vectors + shifts

    lengths = torch.linalg.norm(vectors + eps, dim=-1, keepdim=True)

    if normalize:
        vectors = torch.nan_to_num(torch.div(vectors, lengths))

    return vectors, lengths


def set_default_dtype(dtype: str) -> None:
    """
    Wrapper function of `torch.set_default_dtype`.

    Parameters
    ----------
    dtype: str
        The data type.
    """
    if not isinstance(dtype, str):
        raise TypeError('A string is required to set TORCH default dtype!')
    dtype = dtype.lower()
    if dtype in ['float', 'float32']:
        torch.set_default_dtype(torch.float32)
    elif dtype in ['double', 'float64']:
        torch.set_default_dtype(torch.float64)
    else:
        raise RuntimeError(
            'Unknown/Unsupported data type: "{:s}"!'.format(dtype)
        )


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """
    Helper function of scatter functions.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.script
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Basic `scatter_sum` operations from `torch_scatter`.

    Parameters
    ----------
    src: torch.Tensor
        The source tensor.
    index: torch.Tensor
        The indices of elements to scatter.
    dim: int
        The axis along which to index.
    out: Optional[torch.Tensor]
        The destination tensor.
    dim_size: int
        If out is not given, automatically create output with size dim_size at
        dimension dim. If dim_size is not given, a minimal sized output tensor
        according to index.max() + 1 is returned.
    """
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


@torch.jit.script
def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Basic `scatter_mean` operations from `torch_scatter`.

    Parameters
    ----------
    src: torch.Tensor
        The source tensor.
    index: torch.Tensor
        The indices of elements to scatter.
    dim: int
        The axis along which to index.
    out: Optional[torch.Tensor]
        The destination tensor.
    dim_size: int
        If out is not given, automatically create output with size dim_size at
        dimension dim. If dim_size is not given, a minimal sized output tensor
        according to index.max() + 1 is returned.
    """
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def test_get_centers() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    positions = torch.tensor([
        [6.554,  11.551, 14.302],
        [12.484, 13.391, 8.170],
        [16.167, 7.257,  6.487],
        [16.837, 7.288,  7.170],

        [3.276,  14.148, 3.005],
        [2.042,  14.498, 5.069],
        [11.280, 15.021, 8.510],
        [10.606, 14.539, 8.989],

        [1.230,  4.204,  10.364],
        [3.076,  4.812,  9.054],
        [7.902,  13.749, 1.631],
        [7.198, 13.280,  2.080],

        [10.753, 9.687,  11.751],
        [14.064, 14.303, 9.415],
        [8.404,  12.268, 4.013],
        [8.190,  12.918,  4.682],

        [7.576,  5.051,  15.288],
        [5.450,  6.744,  15.196],
        [11.791, 8.028,  11.087],
        [11.856, 8.952,  11.329],
    ])
    indices = torch.tensor(
        [[[0, 1, -1, -1, -1], [2, 3, -1, -1, -1], [0, 2, 3, -1, -1]],
         [[0, 1, -1, -1, -1], [2, 3, -1, -1, -1], [0, 2, 3, -1, -1]],
         [[0, 1, -1, -1, -1], [2, 3, -1, -1, -1], [0, 2, 3, -1, -1]],
         [[0, 1, -1, -1, -1], [2, 3, -1, -1, -1], [0, 2, 3, -1, -1]],
         [[0, 1, -1, -1, -1], [2, 3, -1, -1, -1], [0, 2, 3, -1, -1]]],
    )

    assert (
        get_centers(positions, indices) - torch.tensor([
            [9.5190000000000, 12.4710000000000, 11.2360000000000,],
            [16.5020000000000, 7.2725000000000, 6.8285000000000,],
            [13.1860000000000, 8.6986666666667, 9.3196666666667,],
            [2.6590000000000, 14.3230000000000, 4.0370000000000,],
            [10.9430000000000, 14.7800000000000, 8.7495000000000,],
            [8.3873333333333, 14.5693333333333,  6.8346666666667,],
            [2.1530000000000, 4.5080000000000, 9.7090000000000,],
            [7.5500000000000, 13.5145000000000, 1.8555000000000,],
            [5.4433333333333, 10.4110000000000, 4.6916666666667,],
            [12.4085000000000, 11.9950000000000, 10.5830000000000,],
            [8.2970000000000, 12.5930000000000, 4.3475000000000,],
            [9.1156666666667, 11.6243333333333, 6.8153333333333,],
            [6.5130000000000, 5.8975000000000, 15.2420000000000,],
            [11.8235000000000, 8.4900000000000, 11.2080000000000,],
            [10.4076666666667, 7.3436666666667, 12.5680000000000,],
        ]) < 1E-12
    ).all()

    torch.set_default_dtype(dtype)


def test_get_distances() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    positions_1 = torch.zeros((4, 3))
    positions_2 = torch.ones((6, 3))
    cells = torch.concat([
        torch.eye(3) * 0.2,
        torch.eye(3) * 1.1,
    ])

    out = get_distances(positions_1, positions_2, cells, 2)
    assert (
        out[1] - torch.tensor(
            [[0] * 6 + [torch.sqrt(torch.tensor(3.))] * 6],
        ).T < 1E-12
    ).all()

    cells = torch.zeros(2, 1)
    assert (
        out[1] - torch.tensor([[torch.sqrt(torch.tensor(3.))] * 12]).T < 1E-12
    ).all()

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=torch.float64
    )
    cells = torch.eye(3) * 0.2
    out = get_distances(positions, positions, cells, 1)
    assert ((
        out[1][[1, 2, 3, 5, 6, 7]]
        - torch.tensor([
            [0.09899494936611666],
            [0.09899494936611666],
            [0.09899494936611666],
            [0.06000000000000000],
            [0.09899494936611666],
            [0.06000000000000000],
        ])
    ) < 1E-12).all()

    torch.set_default_dtype(dtype)


def test_set_default_dtype() -> None:
    set_default_dtype('float64')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float64

    set_default_dtype('float32')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float32

    set_default_dtype('double')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float64

    set_default_dtype('float')
    t = torch.Tensor([1.0])
    assert t.dtype == torch.float32


def test_scatter() -> None:
    src = torch.ones((2, 6, 2), dtype=torch.long)
    index = torch.tensor([0, 1, 0, 1, 2, 1], dtype=torch.long)

    out = scatter_sum(src, index, dim=1)
    assert (
        out == torch.tensor(
            [[[2, 2], [3, 3], [1, 1]], [[2, 2], [3, 3], [1, 1]]],
            dtype=torch.long
        )
    ).all()

    out = scatter_mean(src, index, dim=1)
    assert (
        out == torch.tensor(
            [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]],
            dtype=torch.long
        )
    ).all()


if __name__ == '__main__':
    test_get_centers()
    test_get_distances()
    test_set_default_dtype()
    test_scatter()
