import torch
from torch import nn
import numpy as np
import torch_geometric as tg
from typing import List, Dict, Tuple

from mlcolvar.graph import data as gdata
from mlcolvar.graph.core.nn import radial
from mlcolvar.graph.core.nn import schnet
from mlcolvar.graph.core.nn import gvp_layer
from mlcolvar.graph.utils import torch_tools

"""
GNN models.
"""

__all__ = ['BaseModel', 'GVPModel', 'SchNetModel']


class BaseModel(nn.Module):
    """
    The commen GNN interface for mlcolvar.

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    n_bases: int
        Size of the basis set.
    n_polynomials: bool
        Order of the polynomials in the basis functions.
    basis_type: str
        Type of the basis function.
    """

    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        n_bases: int = 6,
        n_polynomials: int = 6,
        basis_type: str = 'bessel'
    ) -> None:
        super().__init__()
        self._n_out = n_out
        self._radial_embedding = radial.RadialEmbeddingBlock(
            cutoff, n_bases, n_polynomials, basis_type
        )
        self.register_buffer(
            'n_out', torch.tensor(n_out, dtype=torch.int64)
        )
        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.int64)
        )

    def embed_edge(
        self, data: Dict[str, torch.Tensor], normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the edge embedding.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        normalize: bool
            If return the normalized distance vectors.

        Returns
        -------
        edge_lengths: torch.Tensor (shape: [n_edges, 1])
            The edge lengths.
        edge_length_embeddings: torch.Tensor (shape: [n_edges, n_bases])
            The edge length embeddings.
        edge_unit_vectors: torch.Tensor (shape: [n_edges, 3])
            The normalized edge vectors.
        """
        vectors, lengths = torch_tools.get_edge_vectors_and_lengths(
            positions=data['positions'],
            edge_index=data['edge_index'],
            shifts=data['shifts'],
            normalize=normalize,
        )
        return lengths, self._radial_embedding(lengths), vectors


class GVPModel(BaseModel):
    """
    The Geometric Vector Perceptron (GVP) model [1, 2] with vector gate [2].

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    n_bases: int
        Size of the basis set.
    n_polynomials: bool
        Order of the polynomials in the basis functions.
    n_layers: int
        Number of the graph convolution layers.
    n_messages: int
        Number of GVPs to use in the message functions.
    n_feedforwards: int
        Number of GVPs to use in the feedforward functions.
    n_scalars_node: int
        Size of the scalar channel of the node embedding in hidden layers.
    n_vectors_node: int
        Size of the vector channel of the node embedding in hidden layers.
    n_scalars_edge: int
        Size of the scalar channel of the edge embedding in hidden layers.
    drop_rate: int
        Drop probability in all dropout layers.
    activation: str
        Name of the activation function to use in the GVPs (case sensitive).
    basis_type: str
        Type of the basis function.

    References
    ----------
    .. [1] Jing, Bowen, et al.
           "Learning from protein structure with geometric vector perceptrons."
           International Conference on Learning Representations. 2020.
    .. [2] Jing, Bowen, et al.
           "Equivariant graph neural networks for 3d macromolecular structure."
           arXiv preprint arXiv:2106.03843 (2021).
    """
    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        n_bases: int = 8,
        n_polynomials: int = 6,
        n_layers: int = 1,
        n_messages: int = 2,
        n_feedforwards: int = 2,
        n_scalars_node: int = 8,
        n_vectors_node: int = 8,
        n_scalars_edge: int = 8,
        drop_rate: int = 0.1,
        activation: str = 'SiLU',
        basis_type: str = 'bessel'
    ) -> None:
        super().__init__(
            n_out, cutoff, atomic_numbers, n_bases, n_polynomials, basis_type
        )

        self.W_e = nn.ModuleList([
            gvp_layer.LayerNorm((n_bases, 1)),
            gvp_layer.GVP(
                (n_bases, 1),
                (n_scalars_edge, 1),
                activations=(None, None),
                vector_gate=True,
            )
        ])

        self.W_v = nn.ModuleList([
            gvp_layer.LayerNorm((len(atomic_numbers), 0)),
            gvp_layer.GVP(
                (len(atomic_numbers), 0),
                (n_scalars_node, n_vectors_node),
                activations=(None, None),
                vector_gate=True,
            )
        ])

        self.layers = nn.ModuleList(
            gvp_layer.GVPConvLayer(
                (n_scalars_node, n_vectors_node),
                (n_scalars_edge, 1),
                n_message=n_messages,
                n_feedforward=n_feedforwards,
                drop_rate=drop_rate,
                activations=(eval(f'torch.nn.{activation}')(), None),
                vector_gate=True,
            )
            for _ in range(n_layers)
        )

        self.W_out = nn.ModuleList([
            gvp_layer.LayerNorm((n_scalars_node, n_vectors_node)),
            gvp_layer.GVP(
                (n_scalars_node, n_vectors_node),
                (n_out, 0),
                activations=(None, None),
                vector_gate=True,
            ),
        ])

    def forward(
        self, data: Dict[str, torch.Tensor], scatter_mean: bool = True
    ) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        scatter_mean: bool
            If perform the scatter mean to the model output.
        """
        h_V = (data['node_attrs'], None)
        for w in self.W_v:
            h_V = w(h_V)
        h_V_1, h_V_2 = h_V
        assert h_V_2 is not None
        h_V = (h_V_1, h_V_2)

        h_E = self.embed_edge(data)
        h_E = (h_E[1], h_E[2].unsqueeze(-2))
        for w in self.W_e:
            h_E = w(h_E)
        h_E_1, h_E_2 = h_E
        assert h_E_2 is not None
        h_E = (h_E_1, h_E_2)

        batch_id = data['batch']

        for layer in self.layers:
            h_V = layer(h_V, data['edge_index'], h_E)

        for w in self.W_out:
            h_V = w(h_V)
        out = h_V[0]

        if scatter_mean:
            if 'system_masks' not in data.keys():
                out = torch_tools.scatter_mean(out, batch_id, dim=0)
            else:
                out = out * data['system_masks']
                out = torch_tools.scatter_sum(out, batch_id, dim=0)
                out = out / data['n_system']

        return out


class SchNetModel(BaseModel):
    """
    The SchNet [1] model. This implementation is taken from torch_geometric:
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    n_bases: int
        Size of the basis set.
    n_layers: int
        Number of the graph convolution layers.
    n_filters: int
        Number of filters.
    n_hidden_channels: int
        Size of hidden embeddings.

    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Schnet–a deep learning architecture for
        molecules and materials." The Journal of Chemical Physics 148.24
        (2018).
    """

    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        n_bases: int = 16,
        n_layers: int = 2,
        n_filters: int = 16,
        n_hidden_channels: int = 16,
        drop_rate: int = 0
    ) -> None:

        super().__init__(
            n_out, cutoff, atomic_numbers, n_bases, 0, 'gaussian'
        )

        self.W_v = nn.Linear(
            len(atomic_numbers), n_hidden_channels, bias=False
        )

        self.layers = nn.ModuleList([
            schnet.InteractionBlock(
                n_hidden_channels, n_bases, n_filters, cutoff
            )
            for _ in range(n_layers)
        ])

        self.W_out = nn.ModuleList([
            nn.Linear(n_hidden_channels, n_hidden_channels // 2),
            schnet.ShiftedSoftplus(),
            nn.Linear(n_hidden_channels // 2, n_out)
        ])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets all learnable parameters of the module.
        """
        self.W_v.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

        nn.init.xavier_uniform_(self.W_out[0].weight)
        self.W_out[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_out[2].weight)
        self.W_out[2].bias.data.fill_(0)

    def forward(
        self, data: Dict[str, torch.Tensor], scatter_mean: bool = True
    ) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        scatter_mean: bool
            If perform the scatter mean to the model output.
        """

        h_E = self.embed_edge(data)
        h_V = self.W_v(data['node_attrs'])

        batch_id = data['batch']

        for layer in self.layers:
            h_V = h_V + layer(h_V, data['edge_index'], h_E[0], h_E[1])

        for w in self.W_out:
            h_V = w(h_V)
        out = h_V

        if scatter_mean:
            if 'system_masks' not in data.keys():
                out = torch_tools.scatter_mean(out, batch_id, dim=0)
            else:
                out = out * data['system_masks']
                out = torch_tools.scatter_sum(out, batch_id, dim=0)
                out = out / data['n_system']

        return out


def test_get_data() -> tg.data.Batch:
    # TODO: This is not a real test, but a helper function for other tests.
    # Maybe should change its name.
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    numbers = [8, 1, 1]
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
            [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07]],
            [[0.0, 0.0, 0.0], [0.07, 0.0, 0.07], [-0.07, 0.0, 0.07]],
            [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1], [0.17, -0.07, 1.1]],
        ],
        dtype=np.float64
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1]])
    node_labels = np.array([[0], [1], [1]])
    z_table = gdata.atomic.AtomicNumberTable.from_zs(numbers)

    config = [
        gdata.atomic.Configuration(
            atomic_numbers=numbers,
            positions=p,
            cell=cell,
            pbc=[True] * 3,
            node_labels=node_labels,
            graph_labels=graph_labels,
        ) for p in positions
    ]
    dataset = gdata.create_dataset_from_configurations(
        config, z_table, 0.1, show_progress=False
    )

    loader = gdata.GraphDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=10,
        shuffle=False,
    )
    loader.setup()

    return next(iter(loader.train_dataloader()))


def test_gvp() -> None:
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    model = GVPModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=1,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation='SiLU',
    )

    data = test_get_data().to_dict()
    assert (
        torch.abs(
            model(data) -
            torch.tensor([[0.6100070244145421, -0.2559670171962067]] * 6)
        ) < 1E-12
    ).all()


def test_schnet() -> None:
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    model = SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16
    )

    data = test_get_data().to_dict()
    assert (
        torch.abs(
            model(data) -
            torch.tensor([[0.40384621527953063, -0.1257513365138969]] * 6)
        ) < 1E-12
    ).all()


if __name__ == '__main__':
    test_gvp()
    test_schnet()
