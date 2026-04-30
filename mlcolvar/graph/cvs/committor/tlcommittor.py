import torch
import torch_geometric as tg
from typing import Dict, Any, List

from mlcolvar.graph.cvs import GraphBaseCV
from mlcolvar.graph.cvs.cv import test_get_data
from mlcolvar.graph.utils import torch_tools

from mlcolvar.core.nn.utils import Custom_Sigmoid

"""
Data-driven learning of committor function, based on Graph Neural Networks
(GNNs) and using a time-lagged estimator.
"""

__all__ = ['GraphTimeLaggedCommittor']


class GraphTimeLaggedCommittor(GraphBaseCV):
    """
    Data-driven learning of committor function, based on GNNs and a
    time-lagged estimator.

    Parameters
    ----------
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    atomic_masses : List[float]
        List of masses of all the atoms we are using.
    cutoff_l: float
        The lone graph cutoff radius between subsystem atoms.
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options.
    extra_loss_options: Dict[Any, Any]
        Extra loss function options.
    optimizer_options: Dict[Any, Any]
        Optimizer options.
    sync_dist: bool
        If reduces the metric across devices. Use with care as this may lead to
        a significant communication overhead.

    References
    ----------
    .. [1] Megías A, Arredondo SC, Chen CG, Tang C, Roux B, Chipot C.
        "Iterative variational learning of committor-consistent transition
        pathways using artificial neural networks", 2024, arXiv preprint,
        arXiv:2412.01947.

    See also
    --------
    mlcolvar.graph.cvs.committor.Committor
        The variational principle based GNN committor module.
    mlcolvar.graph.cvs.committor.utils.GraphCommittorLoss
        Kolmogorov's variational optimization of committor and imposition of
        boundary conditions.
    mlcolvar.graph.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    """

    def __init__(
        self,
        cutoff: float,
        atomic_numbers: List[int],
        cutoff_l: float = -1.0,
        model_name: str = 'GVPModel',
        model_options: Dict[Any, Any] = {},
        extra_loss_options: Dict[Any, Any] = {
            'alpha': 10000.0,
            'sigmoid_p': 3.0,
            'penalty_weight': 10.0,
            'z_threshold': 10.0,
        },
        optimizer_options: Dict[Any, Any] = {},
        sync_dist: bool = True,
        **kwargs,
    ) -> None:
        if model_options.pop('n_out', None) is not None:
            raise RuntimeError(
                'The `n_out` key of parameter `model_options` will be ignored!'
            )
        model_options['n_out'] = 1

        if optimizer_options != {}:
            kwargs['optimizer_options'] = optimizer_options

        super().__init__(
            2,
            cutoff,
            atomic_numbers,
            cutoff_l,
            model_name,
            model_options,
            **kwargs
        )

        self._alpha = float(
            extra_loss_options.get('alpha', 10000.0)
        )
        self._z_threshold = float(
            extra_loss_options.get('z_threshold', 10.0)
        )
        self._penalty_weight = float(
            extra_loss_options.get('penalty_weight', 10.0)
        )
        self.sigmoid = Custom_Sigmoid(extra_loss_options.get('sigmoid_p', 3.0))
        self.register_buffer('is_committor', torch.tensor(1, dtype=int))
        self._sync_dist = sync_dist

    def forward_nn(
        self,
        data: Dict[str, torch.Tensor],
        token: bool = False
    ) -> torch.Tensor:
        """
        The forward pass for the NN.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        token: bool
            To be used.
        """

        if not self._exporting:
            data['positions'].requires_grad_(True)
            data['node_attrs'].requires_grad_(True)

        return self._model(data)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        token: bool = False
    ) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        token: bool
            To be used.
        """
        z = self.forward_nn(data)
        q = self.sigmoid(z)

        return torch.hstack([z, q])

    def training_step(
        self,
        train_batch: Dict[str, tg.data.Batch],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute and return the training loss and record metrics.

        Parameters
        ----------
        train_batch: Tuple[Dict[str, torch_geometric.data.Batch], int, int]
            The data batch.
        """
        data_t = train_batch['dataset_1'].to_dict()
        data_lag = train_batch['dataset_2'].to_dict()

        labels_t = data_t['graph_labels'].long().squeeze()
        mask_a_t = labels_t == 0
        mask_b_t = labels_t == 1
        labels_lag = data_lag['graph_labels'].long().squeeze()
        mask_a_lag = labels_lag == 0
        mask_b_lag = labels_lag == 1

        z_t = self.forward_nn(data_t)
        z_lag = self.forward_nn(data_lag)
        q_t = self.sigmoid(z_t)
        q_lag = self.sigmoid(z_lag)

        loss_v = torch.mean(torch.pow((q_t - q_lag), 2)).log()
        loss_a = (
            torch.mean(torch.pow((q_t[mask_a_t]), 2))
            + torch.mean(torch.pow((q_lag[mask_a_lag]), 2))
        ) * self._alpha
        loss_b = (
            torch.mean(torch.pow((q_t[mask_b_t] - 1), 2))
            + torch.mean(torch.pow((q_lag[mask_b_lag] - 1), 2))
        ) * self._alpha

        loss_z_range = 0
        over_threshold = torch.relu(z_t.abs() - self._z_threshold)
        loss_z_range += self._penalty_weight * torch.mean(
            over_threshold.pow(2)
        )
        over_threshold = torch.relu(z_lag.abs() - self._z_threshold)
        loss_z_range += self._penalty_weight * torch.mean(
            over_threshold.pow(2)
        )

        loss = loss_v + loss_a + loss_b + loss_z_range

        name = 'train' if self.training else 'valid'
        self.log(
            f'{name}_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=self._sync_dist,
            batch_size=z_t.shape[0],
        )
        self.log(
            f'{name}_loss_variational',
            loss_v,
            on_step=False,
            on_epoch=True,
            sync_dist=self._sync_dist,
            batch_size=z_t.shape[0],
        )
        self.log(
            f'{name}_loss_boundary_A',
            loss_a,
            on_step=False,
            on_epoch=True,
            sync_dist=self._sync_dist,
            batch_size=z_t.shape[0],
        )
        self.log(
            f'{name}_loss_boundary_B',
            loss_b,
            on_step=False,
            on_epoch=True,
            sync_dist=self._sync_dist,
            batch_size=z_t.shape[0],
        )
        self.log(
            f'{name}_loss_z_range',
            loss_z_range,
            on_step=False,
            on_epoch=True,
            sync_dist=self._sync_dist,
            batch_size=z_t.shape[0],
        )
        return loss
