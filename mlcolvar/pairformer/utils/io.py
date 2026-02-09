import os
import torch
import numpy as np
import mdtraj as md
import multiprocessing as mp
import warnings
import collections
from typing import Union, List, Tuple

from mlcolvar.pairformer import data as pdata
from mlcolvar.pairformer.utils import progress

"""
Some I/O things.
"""

__all__ = ['create_dataset_from_trajectories']


def create_dataset_from_trajectories(
    trajectories: Union[List[List[str]], List[str], str],
    top: Union[List[List[str]], List[str], str],
    node_embeddings: List[str] = ['atom_names'],
    element_symbol_as_name: bool = True,
    cutoff: float = -1.0,
    folder: str = None,
    create_labels: bool = True,
    system_selection: str = None,
    environment_selection: str = None,
    center_selections: List[str] = [],
    n_atoms_padded: int = 0,
    return_trajectories: bool = False,
    no_pbc: bool = False,
    show_progress: bool = True,
    n_workers: int = 1,
) -> Union[
    pdata.PairDataSet,
    Tuple[
        pdata.PairDataSet,
        Union[List[List[md.Trajectory]], List[md.Trajectory]]
    ]
]:
    """
    Create a dataset from a set of trajectory files.

    Parameters
    ----------
    trajectories: Union[List[List[str]], List[str], str]
        Names of trajectories files.
    top: Union[List[List[str]], List[str], str]
        Names of topology files.
    cutoff: float (units: Ang)
        The cutoff radius for truncating the system.
    node_embeddings: List[str]
        One-hot node embedding passing to the model.
    element_symbol_as_name: bool
        If use element symbol as atom names.
    folder: str
        Common path for the files to be imported. If set, filenames become
        `folder/file_name`.
    create_labels: bool
        Assign a label to each file according to the total number of files.
        If False, labels of all files will be `-1`.
    system_selection: str
        MDTraj style atom selections [1] of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    environment_selection: str
        MDTraj style atom selections [1] of the environment atoms. If given,
        only the system atoms and [the environment atoms within the cutoff
        radius of the system atoms] will be kept in the graph.
    center_selections: List[str]
        MDTraj style atom selections [1] of the pocket atoms.
    n_atoms_padded: int
        Number of nodes after padding. This is only used by Pairformer.
    return_trajectories: bool
        If also return the loaded trajectory objects.
    no_pbc: bool
        Ignore the PBC.
    show_progress: bool
        If show the progress bar.

    Returns
    -------
    dataset: mlcolvar.graph.data.GraphDataSet
        The graph dataset.
    trajectories: Union[List[List[mdtraj.Trajectory]], List[mdtraj.Trajectory]]
        The loaded trajectory objects.

    Notes
    -----
    The login behind this method is like the follows:
        1. If only `system_selection` is given, the method will only load atoms
        selected by this selection, from the trajectories.
        2. If both `system_selection` and `environment_selection` are given,
        the method will lead the atoms select by both selections, but will
        build graphs using [the system atoms] and [the environment atoms within
        the cutoff radius of the system atoms].

    References
    ----------
    .. [1] https://www.mdtraj.org/1.9.8.dev0/atom_selection.html
    """

    if environment_selection is not None:
        assert system_selection is not None, (
            'the `environment_selection` argument requires the'
            + '`system_selection` argument to be defined!'
        )
        assert len(center_selections) > 0, (
            'the `environment_selection` argument requires the'
            + '`center_selections` argument to be defined!'
        )
        assert (
            type(center_selections) is list
            and all([type(c) is str for c in center_selections])
        ), (
            'the `center_selections` argument should be a list of strings!'
        )
        selection = '({:s}) or ({:s})'.format(
            system_selection, environment_selection
        )
    elif system_selection is not None:
        selection = system_selection
    else:
        selection = None

    # fmt: off
    assert type(trajectories) is type(top), (
        'The `trajectories` and `top` parameters should have the same type!'
    )
    if isinstance(trajectories, str):
        trajectories = [trajectories]
        top = [top]
    assert len(trajectories) == len(top), (
        'Numbers of trajectories and topology files should be the same!'
    )
    # fmt: on

    assert 'atom_names' in node_embeddings, (
        'The `atom_names` embedding is always required!'
    )
    for embedding in node_embeddings:
        if embedding not in pdata.dataset.__implemented_embeddings__:
            raise NotImplementedError(
                'Embedding name {:s} is not implemented!'.format(embedding)
                + 'Implemented embedding includes: {}'.format(
                    pdata.dataset.__implemented_embeddings__
                )
            )

    assert n_workers > 0, 'Number of workers should be positive!'

    for i in range(len(trajectories)):
        assert type(trajectories[i]) is type(top[i]), (
            'Each element of `trajectories` and `top` parameters '
            + 'should have the same type!'
        )
        if isinstance(trajectories[i], list):
            assert len(trajectories[i]) == len(top[i]), (
                'Numbers of trajectories and topology files should be '
                + 'the same!'
            )
            for j in range(len(trajectories[i])):
                if folder is not None:
                    trajectories[i][j] = folder + '/' + trajectories[i][j]
                    top[i][j] = folder + '/' + top[i][j]
                assert isinstance(trajectories[i][j], str)
                assert isinstance(top[i][j], str)
        else:
            if folder is not None:
                trajectories[i] = folder + '/' + trajectories[i]
                top[i] = folder + '/' + top[i]
            assert isinstance(trajectories[i], str)
            assert isinstance(top[i], str)

    topologies = []
    trajectories_in_memory = []

    for i in range(len(trajectories)):
        if isinstance(trajectories[i], list):
            traj = [
                md.load(trajectories[i][j], top=top[i][j])
                for j in range(len(trajectories[i]))
            ]
            for j, t in enumerate(traj):
                t.top = md.core.trajectory.load_topology(top[i][j])
            if selection is not None:
                for j in range(len(traj)):
                    subset = traj[j].top.select(selection)
                    assert len(subset) > 0, (
                        'No atoms will be selected with selection string '
                        + '"{:s}"!'.format(selection)
                    )
                    traj[j] = traj[j].atom_slice(subset)
            trajectories_in_memory.append(traj)
            topologies.extend([t.top for t in traj])
        else:
            traj = md.load(trajectories[i], top=top[i])
            traj.top = md.core.trajectory.load_topology(top[i])
            if selection is not None:
                subset = traj.top.select(selection)
                assert len(subset) > 0, (
                    'No atoms will be selected with selection string '
                    + '"{:s}"!'.format(selection)
                )
                traj = traj.atom_slice(subset)
            trajectories_in_memory.append(traj)
            topologies.append(traj.top)

    mapping_tables = {}
    if 'atom_names' in node_embeddings:
        z_table = _z_table_from_top(topologies, element_symbol_as_name)
        mapping_tables['atom_names'] = z_table
    if 'residue_names' in node_embeddings:
        r_table = _r_table_from_top(topologies)
        mapping_tables['residue_names'] = r_table

    configurations = []
    for i in range(len(trajectories_in_memory)):
        if isinstance(trajectories_in_memory[i], list):
            for j in range(len(trajectories_in_memory[i])):
                configuration = _configures_from_trajectory(
                    trajectories_in_memory[i][j],
                    i if create_labels else -1,  # NOTE: all these configurations have a label `i`
                    node_embeddings,
                    system_selection,
                    environment_selection,
                    center_selections,
                    no_pbc,
                    element_symbol_as_name,
                )
                configurations.extend(configuration)
        else:
            configuration = _configures_from_trajectory(
                trajectories_in_memory[i],
                i if create_labels else -1,
                node_embeddings,
                system_selection,
                environment_selection,
                center_selections,
                no_pbc,
                element_symbol_as_name,
            )
            configurations.extend(configuration)

    # NOTE: we set padding size to the maximum number of the system atoms.
    if n_atoms_padded <= 0:
        n_atoms = [
            (len(c.system) if c.system is not None else len(c.positions))
            for c in configurations
        ]
        n_atoms_padded = np.max(n_atoms)

    if n_workers > 1:
        pool = mp.Pool(processes=n_workers)
        indices = np.array_split(range(len(configurations)), n_workers)
        pool.map(
            _create_dataset_from_configurations_wrapper,
            zip(
                [
                    (
                        [configurations[ii] for ii in i],
                        mapping_tables,
                        cutoff,
                        n_atoms_padded,
                        show_progress,
                    )
                    for i in indices
                ],
                list(range(n_workers))
            )
        )
        pool.close()

        if show_progress:
            items = progress.pbar(
                range(n_workers), frequency=0.0001, prefix='Merging dataset'
            )
        else:
            items = range(n_workers)
        dataset = pdata.cat_dataset([
            torch.load('.MGTEMP.{:d}.pt'.format(i), weights_only=False)
            for i in items
        ])
        for i in range(n_workers):
            os.remove('.MGTEMP.{:d}.pt'.format(i))
    else:
        dataset = pdata.create_dataset_from_configurations(
            configurations,
            mapping_tables,
            cutoff,
            n_atoms_padded,
            show_progress,
        )

    if return_trajectories:
        return dataset, trajectories_in_memory
    else:
        return dataset


def _z_table_from_top(
    top: List[md.Topology], element_symbol_as_name: bool = True
) -> pdata.atomic.GenericMappingTable:
    """
    Create an atomic name table from the topologies.

    Parameters
    ----------
    top: List[mdtraj.Topology]
        The topology objects.
    """
    atom_names = []
    for t in top:
        if element_symbol_as_name:
            atom_names.extend([a.element.symbol for a in t.atoms])
        else:
            atom_names.extend([a.name for a in t.atoms])
    r_table = pdata.atomic.GenericMappingTable.from_names(atom_names)
    return r_table


def _r_table_from_top(
    top: List[md.Topology]
) -> pdata.atomic.GenericMappingTable:
    """
    Create a residue name table from the topologies.

    Parameters
    ----------
    top: List[mdtraj.Topology]
        The topology objects.
    """
    residue_names = []
    for t in top:
        residue_names.extend([r.name for r in t.residues])
    r_table = pdata.atomic.GenericMappingTable.from_names(residue_names)
    return r_table


def _configures_from_trajectory(
    trajectory: md.Trajectory,
    label: int = None,
    node_embeddings: List[str] = [],
    system_selection: str = None,
    environment_selection: str = None,
    center_selections: List[str] = None,
    no_pbc: bool = False,
    element_symbol_as_name: bool = True,
) -> pdata.atomic.Configurations:
    """
    Create configurations from one trajectory.

    Parameters
    ----------
    trajectory: mdtraj.Trajectory
        The MDTraj Trajectory object.
    label: int
        The graph label.
    node_embeddings: List[str]
        One-hot node embedding passing to the model.
    system_selection: str
        MDTraj style atom selections of the system atoms. If given, only
        selected atoms will be loaded from the trajectories. This option may
        increase the speed of building graphs.
    environment_selection: str
        MDTraj style atom selections of the environment atoms. If given,
        only the system atoms and [the environment atoms within the cutoff
        radius of the system atoms] will be kept in the graph.
    center_selections: List[str]
        MDTraj style atom selections [1] of the pocket atoms.
    no_pbc: bool
        Ignore the PBC.
    element_symbol_as_name: bool
        If use element symbol as atom names.
    """
    if label is not None:
        label = np.array([[label]])

    if system_selection is not None and environment_selection is not None:
        system_atoms = trajectory.top.select(system_selection)
        assert len(system_atoms) > 0, (
            'No atoms will be selected with `system_selection`: '
            + '"{:s}"!'.format(system_selection)
        )
        environment_atoms = trajectory.top.select(environment_selection)
        assert len(environment_atoms) > 0, (
            'No atoms will be selected with `environment_selection`: '
            + '"{:s}"!'.format(environment_selection)
        )
    else:
        system_atoms = None
        environment_atoms = None

    center_atoms_list = []
    if center_selections is not None:
        for c in center_selections:
            center_atoms = trajectory.top.select(c)
            center_atoms_list.append(center_atoms)
            assert len(center_atoms) > 0, (
                'No atoms will be selected with `center_selections`: '
                + '"{:s}"!'.format(c)
            )
            if (
                system_selection is not None
                and environment_selection is not None
            ):
                c_1 = collections.Counter(system_atoms)
                c_2 = collections.Counter(center_atoms)
                assert c_1 >= c_2, (
                    "All atoms selected by `center_selections` should also be "
                    + "selected by `system_selection`!"
                )

    if no_pbc and (trajectory.unitcell_vectors is not None):
        warnings.warn(
            'Your trajectory files contain box information, however '
            + 'the `no_pbc` option is enabled. You need to understand what '
            + 'you are doing!'
        )
    if (not no_pbc) and (trajectory.unitcell_vectors is None):
        raise RuntimeError(
            'Your trajectory files does not contain box information, however '
            + 'the `no_pbc` option is not enabled!'
        )

    if (not no_pbc) and (trajectory.unitcell_vectors is not None):
        pbc = [True] * 3
        cell = trajectory.unitcell_vectors
    else:
        pbc = [False] * 3
        cell = [np.zeros((3, 3), dtype=float) for _ in range(len(trajectory))]

    node_attrs = {}
    if 'atom_names' in node_embeddings:
        if element_symbol_as_name:
            atom_names = [a.element.symbol for a in trajectory.top.atoms]
        else:
            atom_names = [a.name for a in trajectory.top.atoms]
        node_attrs['atom_names'] = atom_names
    if 'residue_names' in node_embeddings:
        residue_names = [a.residue.name for a in trajectory.top.atoms]
        node_attrs['residue_names'] = residue_names

    configurations = []
    for i in range(len(trajectory)):
        configuration = pdata.atomic.Configuration(
            positions=trajectory.xyz[i] * 10,
            cell=cell[i] * 10,
            pbc=pbc,
            node_labels=None,  # TODO: Add supports for per-node labels.
            graph_labels=label,
            system=system_atoms,
            environment=environment_atoms,
            centers=center_atoms_list,
            node_attrs=node_attrs,
        )
        configurations.append(configuration)

    return configurations


def _create_dataset_from_configurations_wrapper(args) -> None:
    """
    Wrapper function to `create_dataset_from_configurations`.
    """
    d = pdata.create_dataset_from_configurations(*(args[0]))
    pdata.save_dataset(d, '.MGTEMP.{:d}.pt'.format(args[1]))


def test_create_dataset_from_trajectories(
    text: str, system_selection: str,
) -> None:

    node_embeddings = ['atom_names', 'residue_names']

    with open('test_dataset.pdb', 'w') as fp:
        print(text, file=fp)

    dataset, trajectories = create_dataset_from_trajectories(
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        node_embeddings,
        element_symbol_as_name=False,
        system_selection=system_selection,
        return_trajectories=True,
        show_progress=False,
    )

    assert len(dataset) == 6
    assert len(trajectories) == 2
    assert len(trajectories[0]) == 2
    assert len(trajectories[1]) == 2
    assert len(trajectories[1][0]) == 2
    assert len(trajectories[1][1]) == 2

    assert dataset[0]['graph_labels'] == torch.tensor([[0.0]])
    assert dataset[1]['graph_labels'] == torch.tensor([[0.0]])
    assert dataset[2]['graph_labels'] == torch.tensor([[1.0]])
    assert dataset[3]['graph_labels'] == torch.tensor([[1.0]])
    assert dataset[4]['graph_labels'] == torch.tensor([[1.0]])
    assert dataset[5]['graph_labels'] == torch.tensor([[1.0]])

    if node_embeddings == ['residue_names']:
        assert (dataset[0]['node_labels'] == torch.tensor([[0]] * 3)).all()
        assert (dataset[1]['node_labels'] == torch.tensor([[0]] * 3)).all()
        assert (dataset[2]['node_labels'] == torch.tensor([[0]] * 3)).all()
        assert (dataset[3]['node_labels'] == torch.tensor([[0]] * 3)).all()
        assert (dataset[4]['node_labels'] == torch.tensor([[0]] * 3)).all()
        assert (dataset[5]['node_labels'] == torch.tensor([[0]] * 3)).all()

    dataset, trajectories = create_dataset_from_trajectories(
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
        node_embeddings,
        element_symbol_as_name=False,
        create_labels=False,
        system_selection=system_selection,
        return_trajectories=True,
        show_progress=False,
    )

    assert dataset[0]['graph_labels'] == torch.tensor([[-1.0]])
    assert dataset[1]['graph_labels'] == torch.tensor([[-1.0]])
    assert dataset[2]['graph_labels'] == torch.tensor([[-1.0]])
    assert dataset[3]['graph_labels'] == torch.tensor([[-1.0]])
    assert dataset[4]['graph_labels'] == torch.tensor([[-1.0]])
    assert dataset[5]['graph_labels'] == torch.tensor([[-1.0]])

    def check_data_1(data) -> None:
        if len(data['positions']) == 3:
            assert (
                data['positions'] == torch.tensor([
                    [0.0, 0.0, 0.0],
                    [0.7, 0.7, 0.0],
                    [0.7, -0.7, 0.0],
                ])
            ).all()
        else:
            assert (
                data['positions'] == torch.tensor([
                    [0.0, 0.0, 0.0],
                ])
            ).all()
        assert (
            data['cell'] == torch.tensor([
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ])
        ).all()
        if len(data['positions']) == 3 and dataset.n_atoms_padded == 3:
            assert (
                data['node_attrs'] == torch.tensor([
                    [2.0, 0.0], [0.0, 0.0], [1.0, 0.0]
                ])
            ).all()
        else:
            assert (
                data['node_attrs'] == torch.tensor([
                    [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]
                ])
            ).all()

    for i in range(6):
        check_data_1(dataset[i])

    if system_selection is not None:

        dataset = create_dataset_from_trajectories(
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            node_embeddings,
            element_symbol_as_name=False,
            cutoff=1.0,
            system_selection='type O and {:s}'.format(system_selection),
            environment_selection='type H and {:s}'.format(system_selection),
            center_selections=['index 0'],
            show_progress=False,
        )

        for i in range(6):
            check_data_1(dataset[i])

        dataset = create_dataset_from_trajectories(
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            ['test_dataset.pdb', ['test_dataset.pdb', 'test_dataset.pdb']],
            node_embeddings,
            element_symbol_as_name=False,
            cutoff=1.0,
            system_selection='name H2 and {:s}'.format(system_selection),
            environment_selection='name H1 and {:s}'.format(system_selection),
            center_selections=['name H2 and {:s}'.format(system_selection)],
            show_progress=False,
        )

        def check_data_2(data) -> None:
            assert (
                data['positions'] == torch.tensor([
                    [0.7, -0.7, 0.0], [0.7, 0.7, 0.0],
                ])
            ).all()
            assert (
                data['cell'] == torch.tensor([
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ])
            ).all()
            assert (
                data['node_attrs'] == torch.tensor([
                    [1.0, 0.0], [0.0, 0.0]
                ])
            ).all()

        for i in range(6):
            check_data_2(dataset[i])

    __import__('os').remove('test_dataset.pdb')


if __name__ == '__main__':
    text = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
END
"""
    test_create_dataset_from_trajectories(text, None)

    text = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ATOM      4  OH2 XXXXW   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      5  H1  XXXXW   2       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  XXXXW   2       0.300  -0.300   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 TIP3W   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  H1  TIP3W   1       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      3  H2  TIP3W   1       0.700  -0.700   0.000  1.00  0.00      WT1  H
ATOM      4  OH2 XXXXW   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      5  H1  XXXXW   2       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  XXXXW   2       0.300  -0.300   0.000  1.00  0.00      WT1  H
END
"""
    test_create_dataset_from_trajectories(text, 'not resname XXXX')

    text = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1           1
ATOM      1  OH2 XXXXW   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  OH2 TIP3W   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      3  H1  XXXXW   1       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      4  H1  TIP3W   2       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      5  H2  XXXXW   1       0.300  -0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  TIP3W   2       0.700  -0.700   0.000  1.00  0.00      WT1  H
ENDMODEL
ATOM      1  OH2 XXXXW   1       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      2  OH2 TIP3W   2       0.000   0.000   0.000  1.00  0.00      WT1  O
ATOM      3  H1  XXXXW   1       0.300   0.300   0.000  1.00  0.00      WT1  H
ATOM      4  H1  TIP3W   2       0.700   0.700   0.000  1.00  0.00      WT1  H
ATOM      5  H2  XXXXW   1       0.300  -0.300   0.000  1.00  0.00      WT1  H
ATOM      6  H2  TIP3W   2       0.700  -0.700   0.000  1.00  0.00      WT1  H
END
"""
    test_create_dataset_from_trajectories(text, 'not resname XXXX')
