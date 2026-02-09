import numpy as np
import mdtraj as md
from dataclasses import dataclass
from typing import List, Iterable, Optional, Dict

"""
The helper functions for atomic data.
"""

__all__ = ['GenericMappingTable', 'Configuration', 'Configurations']


class GenericMappingTable:
    """
    A generic mapping table that maps string to index.

    Parameters
    ----------
    name_list: List[int]
        The mapping names in this table.
    """

    def __init__(self, name_list: List[str]):
        self.name_list = name_list

    def __len__(self) -> int:
        return len(self.name_list)

    def index_to_name(self, index: int) -> int:
        return self.name_list[index]

    def name_to_index(self, name: str) -> int:
        return self.name_list.index(name)

    def names_to_indices(self, names: List[str]) -> np.ndarray:
        to_index_fn = np.vectorize(self.name_to_index)
        return to_index_fn(np.array(names))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> 'GenericMappingTable':
        """
        Build the table from an array of names.

        Parameters
        ----------
        names: Iterable[str]
            The mapping names.
        """
        name_set = set()
        for name in names:
            name_set.add(name)
        return cls(sorted(list(name_set)))


def get_masses(names: Iterable[str]) -> List[float]:
    """
    Get atomic masses from atomic numbers.

    Parameters
    ----------
    names: Iterable[str]
        The atomic names.
    """
    return [md.element.Element.getBySymbol(n).mass for n in names]


@dataclass
class Configuration:
    """
    Internal helper class that describe a given configuration of the system.
    """
    positions: np.ndarray               # shape: [n_atoms, 3], units: Ang
    cell: np.ndarray                    # shape: [n_atoms, 3], units: Ang
    pbc: Optional[tuple]                # shape: [3]
    node_labels: Optional[np.ndarray]   # shape: [n_atoms, n_node_labels]
    graph_labels: Optional[np.ndarray]  # shape: [n_graph_labels, 1]
    weight: Optional[float] = 1.0       # shape: []
    system: Optional[np.ndarray] = None         # shape: [n_system_atoms]
    environment: Optional[np.ndarray] = None    # shape: [n_environment_atoms]
    centers: Optional[List[np.ndarray]] = None  # shape: [n_centers, *]
    node_attrs: Optional[Dict[str, List[str]]] = None  # shape: [n_atoms]


Configurations = List[Configuration]


if __name__ == '__main__':
    pass
