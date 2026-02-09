import numpy as np
from matscipy.neighbours import neighbour_list
from typing import Optional, Tuple, List

"""
The neighbor list function. This module is taken from MACE directly:
https://github.com/ACEsuit/mace/blob/main/mace/data/neighborhood.py
"""

__all__ = ['get_neighborhood_of_centers']


def get_neighborhood_of_centers(
    positions: np.ndarray,  # [num_positions, 3]
    centers: np.ndarray,  # [n_centers, 3]
    cutoff: float,
    environment_indices: List[int],
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get neighbor environment atoms of given centers.

    Parameters
    ----------
    positions: numpy.ndarray (shape: [N, 3])
        The positions array.
    centers: numpy.ndarray (shape: [N_c, 3])
        The centers array.
    cutoff: float
        The cutoff radius.
    environment_indices: List[int]
        Indices of the environment atoms.
    pbc: Tuple[bool, bool, bool] (shape: [3])
        If enable PBC in the directions of the three lattice vectors.
    cell: numpy.ndarray (shape: [3, 3])
        The lattice vectors.

    Returns
    -------
    neighbors: numpy.ndarray (shape: [2, n_neighbors])
        Neighbor of the centers.

    Notes
    -----
    Arguments `system_indices` and `environment_indices` must presnet at the
    same time. Only edges formed between [the systems atoms] and
    [environment atoms within the cutoff radius of the systems atoms] will
    be kept. Besides, these two lists could not contain common atoms.
    """

    environment_indices = np.array(environment_indices)
    center_indices = np.arange(len(positions), len(positions) + len(centers))
    positions = np.vstack([positions, centers])

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

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
    if not pbc_x:
        cell[:, 0] = max_positions * 5 * cutoff * identity[:, 0]
    if not pbc_y:
        cell[:, 1] = max_positions * 5 * cutoff * identity[:, 1]
    if not pbc_z:
        cell[:, 2] = max_positions * 5 * cutoff * identity[:, 2]

    sender, receiver, unit_shifts, distances = neighbour_list(
        quantities='ijSd',
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=float(cutoff),
    )

    # Get environment atoms that are neighbors of the system.
    keep_edge_r = np.where(np.in1d(receiver, center_indices))[0]
    keep_sender = np.intersect1d(sender[keep_edge_r], environment_indices)
    keep_edge_s = np.where(np.in1d(sender, np.unique(keep_sender)))[0]
    keep_edge = np.intersect1d(keep_edge_r, keep_edge_s)

    return np.unique(sender[keep_edge])


def test_get_neighborhood_of_centers() -> None:

    positions = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float
    )
    cell = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=float)

    n = get_neighborhood_of_centers(
        positions,
        np.array([[1.5, 1.5, 1.5]], dtype=float),
        cutoff=2.6,
        pbc=[True] * 3,
        cell=cell,
        environment_indices=[0, 1, 2, 3],
    )
    assert (n == np.array([0, 1, 2, 3], dtype=int)).all()

    n = get_neighborhood_of_centers(
        positions,
        np.array([[1.5, 1.5, 1.5]], dtype=float),
        cutoff=2.6,
        pbc=[True] * 3,
        cell=cell,
        environment_indices=[1, 2],
    )
    assert (n == np.array([1, 2], dtype=int)).all()

    n = get_neighborhood_of_centers(
        positions,
        np.array([[2, 2, 2]], dtype=float),
        cutoff=2,
        pbc=[True] * 3,
        cell=cell,
        environment_indices=[0, 1, 2, 3],
    )
    assert (n == np.array([1, 2, 3], dtype=int)).all()

    n = get_neighborhood_of_centers(
        positions,
        np.array([[2, 2, 2]], dtype=float),
        cutoff=1.1,
        pbc=[True] * 3,
        cell=cell,
        environment_indices=[1, 2],
    )
    assert (n == np.array([2], dtype=int)).all()

    n = get_neighborhood_of_centers(
        positions,
        np.array([[2, 2, 2]], dtype=float),
        cutoff=1.1,
        pbc=[True] * 3,
        cell=cell,
        environment_indices=[1],
    )
    assert len(n) == 0


if __name__ == "__main__":
    test_get_neighborhood_of_centers()
