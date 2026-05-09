import torch
from torch import nn
import numpy as np
import torch_geometric as tg
from typing import List, Dict, Optional, Any, Tuple, Union

from mlcolvar.pairformer import data as pdata
from mlcolvar.pairformer.core.nn import radial
from mlcolvar.pairformer.core.nn import pairformer
from mlcolvar.pairformer.utils import torch_tools

"""
PairFormer models.
"""

__all__ = ['BaseModel', 'PairFormerModel']


class BaseModel(nn.Module):
    """
    The common PairFormer interface for mlcolvar.

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    cutoff: float
        Cutoff radius of the basis functions.
    n_bases: int
        Size of the basis set.
    n_polynomials: int
        Order of the polynomials in the basis functions.
    basis_type: str
        Type of the basis function.
    """

    def __init__(
        self,
        n_out: int,
        cutoff: float,
        n_bases: int = 6,
        n_polynomials: int = 6,
        basis_type: str = 'gaussian'
    ) -> None:
        super().__init__()
        self._n_out = n_out

        if cutoff > 0:
            self._radial_embedding = radial.RadialEmbeddingBlock(
                cutoff, -1.0, n_bases, n_polynomials, basis_type
            )
        else:
            self._radial_embedding = None

        self.register_buffer(
            'n_out', torch.tensor(n_out, dtype=torch.int64)
        )
        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )


class FFNNModel(BaseModel):
    """
    Fully connected distances matrix + feedforward network.

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    n_distances: int
        Number of input distances, should be equal to square of atom numbers.
    mapping_names: Dict[str, List[str]]
        Placeholder.
    cutoff: float
        Placeholder.
    layers: List[int]
        Number of interaction layers.
    drop_rate: float
        Placeholder.
    activation: str
        Name of the activation function (case sensitive).
    """

    def __init__(
        self,
        n_out: int,
        n_distances: int,
        mapping_names: Dict[str, List[str]] = {},
        cutoff: float = -1.0,
        hidden_layers: List[int] = [8],
        drop_rate: float = 0.0,
        activation: str = 'ELU',
    ) -> None:

        super().__init__(n_out, -1.0, 2, 0)

        self.n_distances = n_distances
        hidden_layers = [n_distances, *hidden_layers, n_out]

        layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(eval(f'torch.nn.{activation}')())
        self.layers = layers
        self.cn_layer = None

        self._mapping_names = mapping_names

    def reset_parameters(self) -> None:

        for m in self.layers:
            if m.__class__.__name__ == 'Linear':
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        scatter_mean: bool = True,
        return_lengths: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        scatter_mean: bool
            If perform the scatter mean to the model output.
        return_lengths: bool
            If return distances for gradient calculations.
        """

        cell = data['cell']
        node_attrs = data['node_attrs']
        n_graphs = data['ptr'].numel() - 1
        n_atoms = len(data['positions']) // n_graphs

        assert n_atoms * n_atoms == self.n_distances, (
            'Number of the input disntances does not equal to shape of the '
            + 'first NN layer!'
        )

        node_attrs = node_attrs.reshape(n_graphs, n_atoms, node_attrs.shape[1])

        _, pair_lengths = torch_tools.get_distances(
            positions_1=data['positions'],
            positions_2=data['positions'],
            cells=cell,
            n_graphs=n_graphs,
            normalize=False,
            eps=1E-7,
        )

        if return_lengths:
            pair_lengths_ = pair_lengths

        h = pair_lengths.reshape((n_graphs, n_atoms * n_atoms))

        for layer in self.layers:
            h = layer(h)

        if return_lengths:
            return h, pair_lengths_
        else:
            return h


class PairFormerModel(BaseModel):
    """
    The PairFormer model used in AlphaFold 3 [1]. This implementation is taken
    from Protinix: https://github.com/bytedance/Protenix by Zichang Jin.

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    mapping_names: Dict[str, List[str]]
        The node embedding mapping name lists, e.g. the `mapping_names`
        attribute of a `mlcolvar.pairformer.data.PairDataSet` instance.
    cutoff: float
        Cutoff radius of the basis functions. If a negative value is given,
        will not use radial basis functions to expand distances.
    constant_d: float
        Constant used in distance basis: 1.0 / (constant_d + d ** 2).
    n_bases: int
        Size of the basis set.
    n_layers: int
        Number of interaction layers.
    n_heads_apb: int
        Number of attention heads in AttentionPairBias.
    n_heads_pair: int
        Number of attention heads in TriangleAttention.
    n_embedding_pair: int
        Size of the pair embedding array.
    n_hidden_channels_mul: int
        Size of the hidden channels in TriangleMultiplicationOutgoing.
    n_hidden_channels_pair: int
        Size of the hidden channels in TriangleAttention.
    drop_rate: float
        Drop probability in all dropout layers.
    triangle_attention: str
        Type of the triangle attention implementation. Valid options are:
        - 'triattention': Optimized tri-attention module
        - 'torch': PyTorch native implementation
        Note that when applying the model in Committor tasks, this option HAS
        to be 'torch'.
    triangle_multiplicative: Triangle multiplicative implementation type.
        - 'torch': PyTorch native implementation
        - None: Disable triangle update
    pair_transition: bool
        If apply pair transition.
    n_polynomials: int
        Order of the polynomials in the basis functions.

    References
    ----------
    .. [1] Abramson, Josh, et al. "Accurate structure prediction of
        biomolecular interactions with AlphaFold 3."
        Nature 630.8016 (2024): 493-500.
    """

    def __init__(
        self,
        n_out: int,
        mapping_names: Dict[str, List[str]],
        cutoff: float = -1.0,
        constant_d: float = 1.0,
        n_bases: int = 0,
        n_layers: int = 1,
        n_heads_apb: int = 1,
        n_heads_pair: int = 1,
        n_embedding_pair: int = 8,
        n_hidden_channels_mul: int = 16,
        n_hidden_channels_pair: int = 16,
        drop_rate: float = 0.0,
        triangle_attention: str = 'torch',
        triangle_multiplicative: str = 'torch',
        pair_transition: bool = True,
        n_polynomials: int = 0,
        cn_options: Optional[Dict[str, Any]] = None,
    ) -> None:

        if n_bases <= 0:
            n_bases = n_embedding_pair

        super().__init__(n_out, cutoff, n_bases, n_polynomials, 'gaussian')

        n_embedding = n_embedding_pair // 2
        n_embedders = len(mapping_names.keys())

        self.embedders = torch.nn.ModuleList([])
        for emb in pdata.dataset.__implemented_embeddings__:
            if emb in mapping_names.keys():
                self.embedders.append(
                    torch.nn.Embedding(len(mapping_names[emb]), n_embedding)
                )

        self.W_p = torch.nn.Linear(
            n_embedders * n_embedding * 2 + n_embedding_pair, n_embedding_pair
        )
        if cutoff < 0:
            self.W_x = torch.nn.Linear(1, n_embedding_pair, bias=False)
        else:
            self.W_x = None
        if cutoff > 0 and n_bases != n_embedding_pair:
            self.W_b = torch.nn.Linear(n_bases, n_embedding_pair, bias=False)
        else:
            self.W_b = None

        self.layers = torch.nn.ModuleList([
            pairformer.PairformerBlock(
                n_heads=n_heads_apb,
                c_z=n_embedding_pair,
                c_s=0,
                c_hidden_mul=n_hidden_channels_mul,
                c_hidden_pair_att=n_hidden_channels_pair,
                no_heads_pair=n_heads_pair,
                dropout=drop_rate,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                pair_transition=pair_transition,
            ) for _ in range(n_layers)
        ])

        if cn_options is not None:
            n_out_w_c = cn_options.pop('n_centers') * 2
            self.cn_layer = CNModel(**cn_options)
            self.W_c = torch.nn.Linear(n_out_w_c // 2, n_out_w_c)
        else:
            n_out_w_c = 0
            self.cn_layer = None
            self.W_c = None

        n_in_w_out = n_embedding_pair + n_out_w_c
        self.W_out = nn.Sequential(*[
            nn.Linear(n_in_w_out, n_in_w_out // 2),
            pairformer.utils.ShiftedSoftplus(),
            nn.Linear(n_in_w_out // 2, n_out)
        ])

        self._mapping_names = mapping_names
        self._n_embedding_pair = n_embedding_pair
        self._n_centers = n_out_w_c // 2
        self._c_d = constant_d

        self.reset_parameters()

    def reset_parameters(self) -> None:

        for m in self.layers:
            m.reset_parameters()

        nn.init.xavier_uniform_(self.W_out[0].weight)
        self.W_out[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_out[2].weight)
        self.W_out[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_p.weight)
        self.W_p.bias.data.fill_(0)
        if self.W_x is not None:
            nn.init.xavier_uniform_(self.W_x.weight)
        if self.W_b is not None:
            nn.init.xavier_uniform_(self.W_b.weight)
        if self.W_c is not None:
            nn.init.xavier_uniform_(self.W_c.weight)
            self.W_c.bias.data.fill_(0)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        scatter_mean: bool = True,
        return_lengths: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        scatter_mean: bool
            If perform the scatter mean to the model output.
        return_lengths: bool
            If return distances for gradient calculations.
        """

        cell = data['cell']
        pair_masks = data['pair_masks']
        system_masks_padded = data['system_masks_padded'].flatten()
        node_attrs = data['node_attrs'][system_masks_padded]
        positions = data['positions'][system_masks_padded]

        n_graphs = data['ptr'].numel() - 1
        n_atoms = len(data['pair_masks']) // n_graphs

        pair_masks = pair_masks.reshape(n_graphs, n_atoms, n_atoms)
        node_attrs = node_attrs.reshape(n_graphs, n_atoms, node_attrs.shape[1])

        _, pair_lengths = torch_tools.get_distances(
            positions_1=positions,
            positions_2=positions,
            cells=cell,
            n_graphs=n_graphs,
            normalize=False,
            eps=1E-7,
        )
        if return_lengths:
            pair_lengths_ = pair_lengths
        if self._radial_embedding is not None:
            pair_lengths = self._radial_embedding(pair_lengths)
            if self.W_b is not None:
                pair_lengths = self.W_b(pair_lengths)
            pair_lengths = pair_lengths.reshape(
                (n_graphs, n_atoms, n_atoms, self._n_embedding_pair)
            )
        else:
            pair_lengths = pair_lengths.reshape(
                (n_graphs, n_atoms, n_atoms)
            ).unsqueeze(-1)
            pair_lengths = self.W_x(1.0 / (self._c_d + pair_lengths ** 2))

        # layer one: distances only
        _, embedding_pair = self.layers[0](
            s=None, z=pair_lengths, pair_mask=pair_masks
        )

        # inject node type information
        embedding_node_list = []
        for i, embedder in enumerate(self.embedders):
            embedding_node_list.append(embedder(node_attrs[..., i]))
        embedding_node = torch.cat(embedding_node_list, dim=-1)
        embedding_node = embedding_node.reshape(
            n_graphs, n_atoms, embedding_node.shape[-1]
        )

        embedding_pair = self.W_p(torch.cat(
            [
                embedding_node.unsqueeze(1).expand(-1, n_atoms, -1, -1),
                embedding_pair,
                embedding_node.unsqueeze(2).expand(-1, -1, n_atoms, -1),
            ],
            dim=-1,
        ))

        # other layers: distance + node type
        for layer in self.layers[1:]:
            _, embedding_pair = layer(
                s=None, z=embedding_pair, pair_mask=pair_masks
            )

        n_values = pair_masks.sum(dim=(1, 2))
        out = (embedding_pair * pair_masks.unsqueeze(-1)).sum(dim=(1, 2))
        out = out / n_values.unsqueeze(-1)

        if self.cn_layer is not None:
            cn = self.W_c(self.cn_layer(data))
            out = torch.hstack([out, cn])

        if return_lengths:
            return self.W_out(out), pair_lengths_
        else:
            return self.W_out(out)


class CNModel(nn.Module):
    """
    A trival coordination number calculator.

    Parameters
    ----------
    n: int
        The n parameter of the switching function.
    m: int
        The m parameter of the switching function.
    r_0: float
        The r_0 parameter of the switching function.
    d_0: float
        The d_0 parameter of the switching function.
    d_max: float
        The d_max parameter of the switching function.
    """

    def __init__(
        self, n: int, m: int, r_0: float, d_0: float, d_max: float,
    ) -> None:

        super().__init__()

        self.register_buffer(
            'n', torch.tensor(n, dtype=torch.long)
        )
        self.register_buffer(
            'm', torch.tensor(m, dtype=torch.long)
        )
        self.register_buffer(
            'r_0', torch.tensor(r_0, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'd_0', torch.tensor(d_0, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'd_max', torch.tensor(d_max, dtype=torch.get_default_dtype())
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        """

        cell = data['cell']
        environment_masks = data['environment_masks'].view(-1)
        system_masks_padded = ~data['system_masks_padded'].view(-1)

        n_graphs = data['ptr'].numel() - 1
        n_centers = data['centers'].shape[1]

        environment_masks_repeat = torch.repeat_interleave(
            environment_masks,
            torch.ones(
                len(environment_masks), dtype=torch.long, device=cell.device
            ) * n_centers,
            dim=0,
        )

        # center positions
        positions_center = torch_tools.get_centers(
            data['positions'], data['centers']
        )

        # distances
        _, lengths = torch_tools.get_distances(
            positions_1=data['positions'][system_masks_padded],
            positions_2=positions_center,
            cells=cell,
            n_graphs=n_graphs,
            eps=1E-7,
        )

        # decay
        lengths = lengths.flatten()
        distance_masks = lengths > self.d_max
        c = (lengths - self.d_0) / (self.r_0)
        lengths = torch.div(
            (1 - torch.pow(c, self.n) + 1E-12),
            (1 - torch.pow(c, self.m) + 2E-12),
        )
        c = (self.d_max - self.d_0) / (self.r_0)
        lengths_max = torch.div(
            (1 - torch.pow(c, self.n) + 1E-12),
            (1 - torch.pow(c, self.m) + 2E-12),
        )
        lengths = torch.div((lengths - lengths_max), (1 - lengths_max))

        # filter padding atoms/atoms beyond d_max
        lengths = lengths * environment_masks_repeat
        lengths[distance_masks] = 0

        # sum
        results = lengths.reshape(
            n_graphs, lengths.shape[0] // (n_centers * n_graphs), n_centers
        )
        results = results.sum(dim=1)

        return results

    @property
    def device(self) -> torch.device:
        return self.r_0.device


def test_get_data(c: bool = False) -> Tuple[tg.data.Batch, Dict[str, Any]]:
    # TODO: This is not a real test, but a helper function for other tests.
    # Maybe should change its name.
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

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
    atom_names = ['O', 'H', 'H']
    residue_names = ['H2O'] * 3
    if not c:
        config = [
            pdata.atomic.Configuration(
                positions=p,
                cell=cell,
                pbc=[True] * 3,
                graph_labels=graph_labels,
                node_labels=None,
                node_attrs={
                    'atom_names': atom_names, 'residue_names': residue_names
                },
            ) for p in positions
        ]
        dataset = pdata.create_dataset_from_configurations(
            config,
            mapping_tables={
                'atom_names': pdata.atomic.GenericMappingTable.from_names(
                    atom_names
                ),
                'residue_names': pdata.atomic.GenericMappingTable.from_names(
                    residue_names
                )
            },
            n_atoms_padded=3,
            show_progress=False,
        )
    else:
        config = [
            pdata.atomic.Configuration(
                positions=p,
                cell=cell,
                pbc=[True] * 3,
                graph_labels=graph_labels,
                node_labels=None,
                node_attrs={
                    'atom_names': atom_names, 'residue_names': residue_names
                },
                system=np.array([0]),
                environment=np.array([1, 2]),
                centers=np.array([[0]]),
            ) for p in positions
        ]
        dataset = pdata.create_dataset_from_configurations(
            config,
            mapping_tables={
                'atom_names': pdata.atomic.GenericMappingTable.from_names(
                    atom_names
                ),
                'residue_names': pdata.atomic.GenericMappingTable.from_names(
                    residue_names
                )
            },
            cutoff=0.1,
            n_atoms_padded=3,
            show_progress=False,
            n_atoms_padded_environment=3,
        )

    loader = pdata.PairDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=10,
        shuffle=False,
    )
    loader.setup()

    return next(iter(loader.train_dataloader())), dataset.mapping_names


def test_pairformer() -> None:
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    data, mapping_names = test_get_data()

    model = PairFormerModel(2, mapping_names)

    assert (
        torch.abs(
            model(data) -
            torch.tensor([[0.771122634223133, -0.2714238083585388]] * 6)
        ) < 1E-12
    ).all()

    data['cell'] = torch.zeros((6, 1), dtype=float)
    assert (
        torch.abs(
            model(data) -
            torch.tensor([[0.7699681542284031, -0.27189165318942404]] * 6)
        ) < 1E-12
    ).all()

    model = PairFormerModel(2, mapping_names, n_layers=2)
    assert (
        torch.abs(
            model(data) -
            torch.tensor([[-0.1672441388337443, 0.20963137884834385]] * 6)
        ) < 1E-12
    ).all()

    data, mapping_names = test_get_data(True)

    model = PairFormerModel(
        2,
        mapping_names,
        cn_options={
            'n': 6,
            'm': 12,
            'r_0': 0.09,
            'd_0': 0,
            'd_max': 0.1,
            'n_centers': 1,
        },
    )
    assert (
        torch.abs(
            model(data) -
            torch.tensor([[-0.05064218647162956, 0.49252906877217784]] * 6)
        ) < 1E-12
    ).all()


def test_cn() -> None:
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    data, mapping_names = test_get_data(True)
    model = CNModel(6, 12, 0.09, 0.0, 0.1)

    assert (
        torch.abs(
            model(data) -
            torch.tensor([[0.042448066282396]] * 6)
        ) < 1E-12
    ).all()


if __name__ == '__main__':
    test_pairformer()
    test_cn()
