import copy
import torch
import torch_geometric as tg
import numpy as np
from typing import List, Union, Dict, Optional

from mlcolvar.pairformer.data import atomic
from mlcolvar.pairformer.data.neighborhood import get_neighborhood_of_centers
from mlcolvar.pairformer.utils import torch_tools
from mlcolvar.pairformer.utils import progress

"""
Build the Pairformer data from a configuration. This module is taken from MACE:
https://github.com/ACEsuit/mace/blob/main/mace/data/atomic_data.py
"""

__all__ = [
    'PairDataSet',
    'create_dataset_from_configurations',
    'save_dataset',
    'save_dataset_as_exyz',
    'load_dataset',
    'cat_dataset',
]

# WARN: do not change order of this list!
__implemented_embeddings__ = tuple(['atom_names', 'residue_names'])


class PairDataSet(list):
    """
    A very simple Pairformer dataset class.

    Parameters
    ----------
    data: List[torch_geometric.data.Data]
        The data.
    mapping_names: Dict[str, List[str]]
        The node embedding mapping name lists.
        E.g., {'atom_names': ['C', 'H'], 'residue_names': ['ALA', 'NME']}.
    n_atoms_padded: int
        Number of atoms after padding.
    cutoff: float
        The cutoff radius for truncating the system.
    """

    def __init__(
        self,
        data: List[tg.data.Data],
        mapping_names: Dict[str, List[str]],
        n_atoms_padded: int,
        cutoff: float = -1.0,
    ) -> None:
        super().__init__()
        self.extend(data)
        self.__mapping_names = mapping_names
        self.__n_atoms_padded = n_atoms_padded
        self.__cutoff = cutoff

    def __getitem__(
        self,
        index: Union[int, slice, list, range, np.ndarray]
    ) -> Union['PairDataSet', tg.data.Data]:
        """
        Build sub-dataset from the dataset.

        Parameters
        ----------
        index : int, slice or list
            Indices of the data.
        """
        if type(index) in [slice, list, np.ndarray, range]:
            if isinstance(index, slice):
                index = list(range(len(self)))[index]
            data = [super(PairDataSet, self).__getitem__(i) for i in index]
            return PairDataSet(data, self.mapping_names, self.cutoff)
        elif np.issubdtype(type(index), np.integer):
            return super(PairDataSet, self).__getitem__(index)
        else:
            raise RuntimeError(
                'Could only indexing a PairDataSet by an int, slice or list!'
            )

    def __repr__(self) -> str:
        result = 'PAIRDATASET [ '

        data_string = '\033[32m{:d}\033[0m\033[36m 󰡷 \033[0m'
        result = result + data_string.format(len(self))
        if self.cutoff > 0:
            result = result + '| '
            data_string = '\033[32m{:f}\033[0m\033[36m 󰳁 \033[0m'
            result = result + data_string.format(self.cutoff)
        result = result + ']'

        return result

    @property
    def cutoff(self) -> float:
        """
        The cutoff radius for truncating the system.
        """
        return self.__cutoff

    @property
    def mapping_names(self) -> Dict[str, List[str]]:
        """
        The mapping name lists.
        """
        return copy.deepcopy(self.__mapping_names)

    @property
    def n_atoms_padded(self) -> int:
        """
        Number of system atoms after padding.
        """
        return self.__n_atoms_padded


def _create_dataset_from_configuration(
    config: atomic.Configuration,
    mapping_tables: Dict[str, atomic.GenericMappingTable],
    n_atoms_padded: int,
    cutoff: float = -1.0,
) -> tg.data.Data:
    """
    Build the Pairformer data object from a configuration.

    Parameters
    ----------
    config: mlcolvar.pairformer.utils.atomic.Configuration
        The configuration.
    n_atoms_padded: int
        Number of atoms after padding.
    mapping_tables: Dict[str, atomic.GenericMappingTable]
        The node embedding mapping tables.
    cutoff: float
        The cutoff radius for truncating the system.
    """

    assert config.graph_labels is None or len(config.graph_labels.shape) == 2

    # NOTE: positions tensor layout:
    # no neighbors:
    #  [[x, y, x], [x, y, x], ..., [0, 0, 0], [0, 0, 0]]
    #   |     n_system     |                          |
    #   |               n_atoms_padded                |
    # with neighbots:
    #  [[x_s, y_s, x_s] ..., [0, 0, 0], ... [x_e, y_e, x_e],]
    #   |  n_system   |              |                    |
    #   |       n_atoms_padded       |     n_neighbors    |
    # thus, the first `n_atoms_padded` elements of this tensor could be safely
    # used in Pairformer calculations.

    if cutoff > 0:
        assert config.system is not None and config.environment is not None, (
            'The `cutoff` option argument requires the `system_selection`'
            '`environment_selection` options to be defined!'
        )
        assert config.centers is not None, (
            'The `cutoff` option argument requires centers to be given!'
        )
        for i, c in enumerate(config.centers):
            assert (set(c) == set(c).intersection(set(config.system))), (
                f'Not all atoms in center group {i} are system atoms!'
            )

        n_atoms_in_center_max = max([len(c) for c in config.centers])
        centers = -torch.ones(
            (len(config.centers), n_atoms_in_center_max), dtype=torch.long
        )
        for i, c in enumerate(config.centers):
            centers[i, :len(c)] = torch.tensor(c, dtype=torch.long)

        positions_tmp = torch.tensor(config.positions)
        positions_centers = torch_tools.get_centers(
            positions_tmp,
            centers,
            torch.tensor([0, len(positions_tmp)], dtype=torch.long)
        )
        positions_centers = positions_centers.detach().numpy()
        neighbors = get_neighborhood_of_centers(
            positions=positions_tmp,
            centers=positions_centers,
            environment_indices=config.environment,
            cutoff=cutoff,
            pbc=config.pbc,
            cell=config.cell,
        )

        positions_system = torch.tensor(
            config.positions[config.system], dtype=torch.get_default_dtype()
        )

        # use local indices as centers
        for i, c in enumerate(config.centers):
            c = [
                np.where(config.system == x)[0][0]
                if x in config.system else -1 for x in c
            ]
            centers[i, :len(c)] = torch.tensor(c, dtype=torch.long)

    else:
        positions_system = torch.tensor(
            config.positions, dtype=torch.get_default_dtype()
        )
        neighbors = np.array([])
        centers = None

    assert len(positions_system) <= n_atoms_padded, (
        'Number of nodes {:d} is larger than the padding size {:d}'.format(
            len(positions_system), n_atoms_padded
        )
    )
    positions = torch.zeros(
        (n_atoms_padded + len(neighbors), 3), dtype=torch.get_default_dtype()
    )
    positions[:len(positions_system), :] = positions_system
    if len(neighbors) > 0:
        positions[-len(neighbors):, :] = torch.tensor(
            config.positions[neighbors], dtype=torch.get_default_dtype()
        )

    cell = torch.tensor(config.cell, dtype=torch.get_default_dtype())
    if (cell < 1E-7).all():
        cell = torch.zeros((1, 1), dtype=torch.get_default_dtype())

    node_attrs_list = []
    if 'atom_names' in mapping_tables.keys():
        table = mapping_tables['atom_names']
        atom_names = config.node_attrs.get('atom_names', None)
        if atom_names is None:
            raise AttributeError(
                'Can not read the `atom_names` list from configuration!'
            )
        if config.system is not None:
            atom_names = [atom_names[i] for i in config.system]
        atomic_numbers = table.names_to_indices(atom_names)
        node_attrs_list.append(
            torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(-1)
        )
    else:
        raise AttributeError(
            'Can not find the `atom_names` embedding, which is always '
            + 'required, from the embedding tables!'
        )
    if 'residue_names' in mapping_tables.keys():
        table = mapping_tables['residue_names']
        residue_names = config.node_attrs.get('residue_names', None)
        if residue_names is None:
            raise AttributeError(
                'Can not read the `residue_name` list from configuration!'
            )
        if config.system is not None:
            residue_names = [residue_names[i] for i in config.system]
        residue_indices = table.names_to_indices(residue_names)
        node_attrs_list.append(
            torch.tensor(residue_indices, dtype=torch.long).unsqueeze(-1)
        )

    node_attrs = torch.zeros(
        (n_atoms_padded + len(neighbors), len(mapping_tables.keys())),
        dtype=torch.long,
    )
    node_attrs[:len(node_attrs_list[0]), :] = torch.hstack(node_attrs_list)

    graph_labels = (
        torch.tensor(config.graph_labels, dtype=torch.get_default_dtype())
        if config.graph_labels is not None
        else None
    )

    weight = (
        torch.tensor(config.weight, dtype=torch.get_default_dtype())
        if config.weight is not None
        else 1
    )

    n_system = torch.tensor(
        [[positions_system.shape[0]]], dtype=torch.get_default_dtype()
    )
    n_system_padded = torch.tensor(
        [[n_atoms_padded]], dtype=torch.get_default_dtype()
    )

    pair_masks = torch.zeros(
        (n_atoms_padded, n_atoms_padded), dtype=torch.long
    )
    pair_masks[:len(positions_system), :len(positions_system)] = 1

    system_masks_padded = torch.zeros((len(positions), 1), dtype=torch.bool)
    system_masks_padded[:n_atoms_padded, 0] = 1

    return tg.data.Data(
        positions=positions,               # [n_atoms_padded + n_neighbors, 3]
        cell=cell,                         # [3, 3]
        node_attrs=node_attrs,             # [n_atoms_padded + n_neighbors, 3]
        graph_labels=graph_labels,         # [1, 1]
        n_system=n_system,                 # [1, 1]
        n_system_padded=n_system_padded,   # [1, 1]
        weight=weight,                     # [1]
        pair_masks=pair_masks,             # [n_atoms_padded, n_atoms_padded]
        centers=centers,                   # [n_centers, n_atoms_in_center_max]
        system_masks_padded=system_masks_padded,
                                           # [n_atoms_padded + n_neighbors, 1]
    )


def create_dataset_from_configurations(
    config: atomic.Configurations,
    mapping_tables: Dict[str, atomic.GenericMappingTable],
    cutoff: float = -1.0,
    n_atoms_padded: int = 0,
    show_progress: bool = True
) -> PairDataSet:
    """
    Build Pairformer data objects from configurations.

    Parameters
    ----------
    config: mlcolvar.pairformer.utils.atomic.Configurations
        The configurations.
    mapping_tables: Dict[str, atomic.GenericMappingTable]
        The node embedding mapping tables.
    cutoff: float
        The cutoff radius for truncating the system.
    n_atoms_padded: int
        Number of nodes after padding. This is only used by Pairformer.
    show_progress: bool
        If show the progress bar.
    """
    if show_progress:
        items = progress.pbar(config, frequency=0.0001, prefix='Making pairs')
    else:
        items = config

    data_list = [
        _create_dataset_from_configuration(
            c, mapping_tables, n_atoms_padded, cutoff
        ) for c in items
    ]

    mapping_names = {
        k: mapping_tables[k].name_list for k in mapping_tables.keys()
    }
    dataset = PairDataSet(data_list, mapping_names, n_atoms_padded, cutoff)

    return dataset


def cat_dataset(datasets: List[PairDataSet]) -> PairDataSet:
    """
    Concatenate multiple datasets.

    Parameters
    ----------
    datasets: List[PairDataSet]
        The datasets.
    """
    d0 = datasets[0]
    same_cutoffs = all(d.cutoff == d0.cutoff for d in datasets)
    same_mapping_names = all(
        d.mapping_names == d0.mapping_names for d in datasets
    )
    same_n_atoms_paddeds = all(
        d.n_atoms_padded == d0.n_atoms_padded for d in datasets
    )

    assert same_cutoffs, (
        'Cutoff radii are different in different datasets!'
    )
    assert same_mapping_names, (
        'Mapping names are different in different datasets!'
    )
    assert same_n_atoms_paddeds, (
        'Padding sizes are different in different datasets!'
    )

    return PairDataSet(
        [dd for d in datasets for dd in d],
        mapping_names=d0.mapping_names,
        n_atoms_padded=d0.n_atoms_padded,
        cutoff=d0.cutoff,
    )


def save_dataset(dataset: PairDataSet, file_name: str) -> None:
    """
    Save a dataset to disk.

    Parameters
    ----------
    dataset: PairDataSet
        The dataset.
    file_name: str
        The filename.
    """
    assert isinstance(dataset, PairDataSet)

    torch.save(dataset, file_name)  # super torch magic go brrrrrrrrr


def load_dataset(file_name: str) -> PairDataSet:
    """
    Load a dataset from disk.

    Parameters
    ----------
    file_name: str
        The filename.
    """
    dataset = torch.load(file_name, weights_only=False)

    assert isinstance(dataset, PairDataSet)

    return dataset


def save_dataset_as_exyz(dataset: PairDataSet, file_name: str) -> None:
    """
    Save a dataset to disk in the extxyz format.

    Parameters
    ----------
    dataset: PairDataSet
        The dataset.
    file_name: str
        The filename.
    """
    fp = open(file_name, 'w')

    assert 'atom_names' in dataset.mapping_names, (
        'Dataset does not contain atomic names!'
    )
    z_table = atomic.GenericMappingTable(dataset.mapping_names['atom_names'])

    for d in dataset:
        print(len(d['positions']), file=fp)
        line = (
            'Lattice="{:s}" '.format((r'{:.5f} ' * 9).strip())
            + 'Properties=species:S:1:pos:R:3 pbc="T T T"'
        )
        cell = [c.item() for c in d['cell'].flatten()]
        print(line.format(*cell), file=fp)
        for i in range(0, len(d['positions'])):
            s = z_table.index_to_name(d['node_attrs'][i, 0].item())
            print('{:2s}'.format(s), file=fp, end=' ')
            positions = [p.item() for p in d['positions'][i]]
            print('{:10.5f} {:10.5f} {:10.5f}'.format(*positions), file=fp)

    fp.close()


def test_from_configurations() -> None:

    torch_tools.set_default_dtype('float64')

    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2

    atom_names = ['O', 'H', 'H']
    residue_names = [['H2O'] * 3] * 5 + [['H3O'] * 3] * 5
    config = [atomic.Configuration(
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        graph_labels=np.array([[i]]),
        node_labels=None,
        node_attrs={
            'atom_names': atom_names, 'residue_names': residue_names[i]
        }
    ) for i in range(0, 10)]
    dataset = create_dataset_from_configurations(
        config,
        mapping_tables={
            'atom_names': atomic.GenericMappingTable.from_names(
                atom_names
            ),
            'residue_names': atomic.GenericMappingTable.from_names(
                ['H2O', 'H3O']
            )
        },
        n_atoms_padded=3,
        show_progress=False,
    )
    for i in range(5):
        assert (
            dataset[i]['node_attrs'] == torch.tensor([[1, 0], [0, 0], [0, 0]])
        ).all()
    for i in range(5, 10):
        assert (
            dataset[i]['node_attrs'] == torch.tensor([[1, 1], [0, 1], [0, 1]])
        ).all()
    for i in range(10):
        assert (
            dataset[i]['positions'] == torch.tensor([
                [0.0, 0.0, 0.0],
                [0.07, 0.07, 0.0],
                [0.07, -0.07, 0.0],
            ])
        ).all()

    dataset_1 = dataset[range(0, 5, 2)]
    assert (dataset_1[0]['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset_1[1]['graph_labels'] == torch.tensor([[2.0]])).all()
    assert (dataset_1[2]['graph_labels'] == torch.tensor([[4.0]])).all()

    dataset_1 = dataset[np.array([0, -1])]
    assert (dataset_1[0]['graph_labels'] == torch.tensor([[0.0]])).all()
    assert (dataset_1[1]['graph_labels'] == torch.tensor([[9.0]])).all()

    config = [atomic.Configuration(
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        graph_labels=np.array([[i]]),
        node_labels=None,
        node_attrs={
            'atom_names': atom_names, 'residue_names': residue_names[i]
        },
        system=[0],
        environment=[1, 2],
        centers=[[0]],
    ) for i in range(0, 10)]
    dataset = create_dataset_from_configurations(
        config,
        mapping_tables={
            'atom_names': atomic.GenericMappingTable.from_names(
                atom_names
            ),
            'residue_names': atomic.GenericMappingTable.from_names(
                ['H2O', 'H3O']
            ),
        },
        cutoff=0.1,
        n_atoms_padded=3,
        show_progress=False,
    )
    for i in range(10):
        assert (
            dataset[i]['positions'] == torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.07, 0.07, 0.0],
                [0.07, -0.07, 0.0],
            ])
        ).all()

    config = [atomic.Configuration(
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        graph_labels=np.array([[i]]),
        node_labels=None,
        node_attrs={
            'atom_names': atom_names, 'residue_names': residue_names[i]
        },
        system=[1],
        environment=[2],
        centers=[[1]],
    ) for i in range(0, 10)]
    dataset = create_dataset_from_configurations(
        config,
        mapping_tables={
            'atom_names': atomic.GenericMappingTable.from_names(
                atom_names
            ),
            'residue_names': atomic.GenericMappingTable.from_names(
                ['H2O', 'H3O']
            ),
        },
        cutoff=0.1,
        n_atoms_padded=3,
        show_progress=False,
    )
    for i in range(10):
        assert (
            dataset[i]['positions'] == torch.tensor([
                [0.07, 0.07, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.07, -0.07, 0.0],
            ])
        ).all()
        assert (
            dataset[i]['cell'] == torch.tensor([
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.2],
            ])
        ).all()


def test_cat_dataset() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = [np.array([[1]]) * i for i in range(6)]
    node_labels = np.array([[0], [1], [1]])
    atom_names = ['O', 'H', 'H']
    residue_names = [['H2O'] * 3] * 5 + [['H3O'] * 3] * 5

    config = [atomic.Configuration(
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        graph_labels=graph_labels[i],
        node_labels=node_labels,
        node_attrs={
            'atom_names': atom_names, 'residue_names': residue_names[i]
        },
        system=[0],
        environment=[1, 2],
        centers=[[0]],
    ) for i in range(6)]
    dataset = create_dataset_from_configurations(
        config,
        mapping_tables={
            'atom_names': atomic.GenericMappingTable.from_names(
                atom_names
            ),
            'residue_names': atomic.GenericMappingTable.from_names(
                ['H2O', 'H3O']
            ),
        },
        n_atoms_padded=3,
        show_progress=False,
    )

    dataset = cat_dataset([dataset, dataset])
    assert [d.graph_labels[0, 0] for d in dataset] == [0, 1, 2, 3, 4, 5] * 2

    graph_labels = [np.array([[1]]) * (i + 6) for i in range(6)]
    config = [atomic.Configuration(
        positions=positions,
        cell=cell,
        pbc=[True] * 3,
        graph_labels=graph_labels[i],
        node_labels=node_labels,
        node_attrs={
            'atom_names': atom_names, 'residue_names': residue_names[i]
        },
        system=[0],
        environment=[1, 2],
        centers=[[0]],
    ) for i in range(6)]
    dataset_1 = create_dataset_from_configurations(
        config,
        mapping_tables={
            'atom_names': atomic.GenericMappingTable.from_names(
                atom_names
            ),
            'residue_names': atomic.GenericMappingTable.from_names(
                ['H2O', 'H3O']
            ),
        },
        n_atoms_padded=3,
        show_progress=False,
    )

    dataset = cat_dataset([dataset, dataset_1, dataset])
    assert [d.graph_labels[0, 0] for d in dataset] == (
            [0, 1, 2, 3, 4, 5] * 2
            + [6, 7, 8, 9, 10, 11]
            + [0, 1, 2, 3, 4, 5] * 2
    )

    dataset_1 = create_dataset_from_configurations(
        config,
        mapping_tables={
            'atom_names': atomic.GenericMappingTable.from_names(
                atom_names
            ),
            'residue_names': atomic.GenericMappingTable.from_names(
                ['H2O', 'H3O']
            ),
        },
        cutoff=0.2,
        n_atoms_padded=3,
        show_progress=False,
    )
    try:
        dataset = cat_dataset([dataset, dataset_1])
    except Exception:
        pass
    else:
        raise Exception()


if __name__ == '__main__':
    test_from_configurations()
    test_cat_dataset()
