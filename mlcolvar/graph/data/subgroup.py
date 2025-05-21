import numpy as np
from typing import Optional, Tuple, List

__all__ = ['get_subgroup']

def get_subgroup(
        positions: np.ndarray,  # [num_positions, 3]
        cutoff: float,
        group1_indices: List[int], 
        group2_indices: List[int] = None, 
        pbc: Optional[Tuple[bool, bool, bool]] = None,
        cell: Optional[np.ndarray] = None,  # [3, 3]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the subgroup list of given group atoms
    
    Parameters
    ----------
    positions: numpy.ndarray( shape: [N, 3])
        The positions array.
    group1_indices: List[int]
    group2_indices: List[int]
    pbc: Tuple[bool, bool, bool] (shape: [3])
        If to enable PBC in the directions of the three lattice vectors.
    cell: numpy.ndarray (shape: [3, 3])
        The lattice vectors.
    
    """
    if group1_indices is None:
        raise ValueError("group1_indicesust be non-empty lists of integers.")
    
    if group1_indices is not None and group2_indices is not None:
        assert np.intersect1d(group1_indices, group2_indices).size == 0, "Groups must be disjoint"

    # Set periodic boundary conditions (default is no PBC)
    if pbc is None:
        pbc = (False, False, False)

    # Default to identity matrix if cell is not provided
    if cell is None or np.all(cell == 0):
        cell = np.identity(3, dtype=float)

    # Check PBC type and cell shape
    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to
    # be increased.
    expand_factor = 5 # TO DO
    if not pbc_x:
        cell[:, 0] = max_positions * expand_factor * identity[:, 0]
    if not pbc_y:
        cell[:, 1] = max_positions * expand_factor * identity[:, 1]
    if not pbc_z:
        cell[:, 2] = max_positions * expand_factor * identity[:, 2]

    sender, receiver, unit_shifts, distances = subgroup_list(
        pbc=np.array(pbc, dtype=bool),
        cell=cell,
        positions=positions,
        group1_indices=group1_indices,
        group2_indices=group2_indices,
        cutoff=cutoff
    )

    # Construct edge index
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # Compute shifts based on PBC
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts


def subgroup_list(pbc, cell, positions, cutoff, group1_indices, group2_indices=None):
    if positions is None:
        raise ValueError("Please provide a positions array")

    if cell is None:
        # Shrink wrapped cell
        rmin = np.min(positions, axis=0)
        rmax = np.max(positions, axis=0)
        cell = np.diag(rmax - rmin)

    if pbc is None:
        pbc = np.zeros(3, dtype=bool)

    sender = []
    receiver = []
    unit_shifts = []
    distances = []
    
    if group2_indices is None:
        # Fully connected within group1, excluding self-connections
        pairs = [(i,j) for i in group1_indices for j in group1_indices if i != j]
    else:
        # Fully connected between group1 and group2 (bidirectional)
        pairs = [(i, j) for i in group1_indices for j in group2_indices] + \
                [(j, i) for i in group1_indices for j in group2_indices]

    for i, j in pairs:
        pos1 = positions[i]
        pos2 = positions[j]
        delta = pos1 - pos2   

        if pbc.any():  # Apply PBC if necessary
            unit_shift = np.round(delta / np.diagonal(cell))
            delta_shifted = delta - unit_shift * np.diagonal(cell, axis1=-2, axis2=-1)
            dist = np.linalg.norm(delta_shifted)
        else:
            unit_shift = np.zeros(3)
            dist = np.linalg.norm(delta)

        if dist <= cutoff:
            sender.append(i)
            receiver.append(j)
            unit_shifts.append(unit_shift)
            distances.append(dist)

    return sender, receiver, unit_shifts, distances

def test_get_subgroup_1() -> None:

    positions = np.array(
        [[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=float
    )
    cell = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=float)
    
    n, s, u = get_subgroup(positions, 0, group1_indices=[0,1,2,3])
    assert (
        n == np.array(
            [[0,0,0,1,1,1,2,2,2,3,3,3], [1,2,3,0,2,3,0,1,3,0,1,2]]
        )
    ).all()
    n, s, u = get_subgroup(
        positions, 0, group1_indices=[0,1,2,3], pbc=[True] * 3, cell=cell,
    )
    assert (
        n == np.array(
            [[0,0,0,1,1,1,2,2,2,3,3,3], [1,2,3,0,2,3,0,1,3,0,1,2]], dtype=int
        )
    ).all()
    assert(
        s == np.array(
            [
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ -4., -4., -4.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 4.,  4.,  4.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ -1., -1., -1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=int
        )
    ).all()
    n, s, u = get_subgroup(
        positions, 2, group1_indices=[0,1,2,3], pbc=[True] * 3, cell=cell,
    )
    assert (
        n == np.array(
            [[0,1,2,3], [2,3,0,1]], dtype=int
        )
    ).all()
    assert(
        s == np.array(
            [
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=int
        )
    ).all()


def test_get_subgroup_2() -> None:

    positions = np.array(
        [[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=float
    )
    cell = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=float)
    
    n, s, u = get_subgroup(positions, cutoff=0, group1_indices=[0,1], group2_indices=[2,3],)
    assert (
        n == np.array(
            [[0, 0, 1, 1, 2, 3, 2, 3], [2, 3, 2, 3, 0, 0, 1, 1]]
        )
    ).all()
    n, s, u = get_subgroup(
        positions, cutoff=0, group1_indices=[0,1], group2_indices=[2,3], pbc=[True] * 3, cell=cell,
    )
    print(s)
    assert (
        n == np.array(
            [[0, 0, 1, 1, 2, 3, 2, 3], [2, 3, 2, 3, 0, 0, 1, 1]], dtype=int
        )
    ).all()
    assert(
        s == np.array(
            [
                [ 0.,  0.,  0.],
                [ -4.,  -4.,  -4.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 4.,  4.,  4.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [
                [ 0.,  0.,  0.],
                [ -1.,  -1.,  -1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=int
        )
    ).all()
    n, s, u = get_subgroup(
        positions, cutoff=2, group1_indices=[0,1], group2_indices=[2,3], pbc=[True] * 3, cell=cell,
    )
    assert (
        n == np.array(
            [[0, 1, 2, 3,], [2, 3, 0, 1]], dtype=int
        )
    ).all()
    assert(
        s == np.array(
            [
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
            ],
            dtype=float
        )
    ).all()
    assert (
        u == np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            dtype=int
        )
    ).all()

def test_subgroup_list():
    positions = np.array(
        [[0,0,0], [1,1,1], [2,2,2], [3,3,3]], dtype=float
    )
    group1_indices = np.array([0, 1])
    group2_indices = np.array([2,3])
    cell = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=float)
    pbc = np.array([False, False, False])
    cutoff = 2.0

    sender, receiver, unit_shifts, distances = subgroup_list(pbc, cell, positions, cutoff, group1_indices, group2_indices)
    assert(sender == np.array([0, 0, 1, 2, 3, 3])).all()
    assert(receiver == np.array([2, 3, 3, 0, 0, 1])).all()
    assert(unit_shifts == np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])).all()

    pbc = np.array([True,True,True])
    sender, receiver, unit_shifts, distances = subgroup_list(pbc, cell, positions, cutoff, group1_indices=[0,1,2,3])
    assert(sender == np.array([0, 1, 2, 3])).all()
    assert(receiver == np.array([2, 3, 0, 1])).all()
    assert(unit_shifts == np.array([[0.,0.,0.],[0.,0.,0.],[0,0,0],[0,0,0]])).all()

    sender, receiver, unit_shifts, distances = subgroup_list(pbc, cell, positions, cutoff, group1_indices, group2_indices)
    assert(sender == np.array([0, 1, 2, 3])).all()
    assert(receiver == np.array([2, 3, 0, 1])).all()
    assert(unit_shifts == np.array([[0.,0.,0.],[0.,0.,0.],[0,0,0],[0,0,0]])).all()

if __name__ == "__main__":
    test_subgroup_list()
    test_get_subgroup_1()
    test_get_subgroup_2()
