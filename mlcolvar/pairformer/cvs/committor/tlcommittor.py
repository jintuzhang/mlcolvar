import torch
import torch_geometric as tg
from typing import Dict, Any, List

from mlcolvar.pairformer.cvs import PairBaseCV
from mlcolvar.pairformer.cvs.cv import test_get_data
from mlcolvar.pairformer.utils import torch_tools

from mlcolvar.core.nn.utils import Custom_Sigmoid

"""
Data-driven learning of committor function, based on Pairformer and using a
time-lagged estimator.
"""

__all__ = ['PairTimeLaggedCommittor']


class PairTimeLaggedCommittor(PairBaseCV):
    """
    Data-driven learning of committor function, based on GNNs and a
    time-lagged estimator.

    Parameters
    ----------
    mapping_names: Dict[str, List[str]]
        The node embedding mapping name lists, e.g. the `mapping_names`
        attribute of a `mlcolvar.pairformer.data.PairDataSet` instance.
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options.
    extra_loss_options: Dict[Any, Any]
        Extra loss function options.
    optimizer_options: Dict[Any, Any]
        Optimizer options.

    References
    ----------
    .. [1] Megías A, Arredondo SC, Chen CG, Tang C, Roux B, Chipot C.
        "Iterative variational learning of committor-consistent transition
        pathways using artificial neural networks", 2024, arXiv preprint,
        arXiv:2412.01947.

    See also
    --------
    mlcolvar.pairformer.cvs.committor.Committor
        The variational principle based GNN committor module.
    mlcolvar.pairformer.cvs.committor.utils.PairCommittorLoss
        Kolmogorov's variational optimization of committor and imposition of
        boundary conditions.
    mlcolvar.pairformer.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    """

    def __init__(
        self,
        mapping_names: Dict[str, List[str]],
        model_name: str = 'PairFormerModel',
        model_options: Dict[Any, Any] = {},
        extra_loss_options: Dict[Any, Any] = {
            'alpha': 10000.0,
            'sigmoid_p': 3.0,
            'penalty_weight': 10.0,
            'z_threshold': 10.0,
        },
        optimizer_options: Dict[Any, Any] = {},
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
            mapping_names,
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

        # avoid nan
        loss_a = torch.nan_to_num(loss_a)
        loss_b = torch.nan_to_num(loss_b)
        loss_v = torch.nan_to_num(loss_v, posinf=torch.inf, neginf=-torch.inf)

        loss = loss_v + loss_a + loss_b + loss_z_range

        name = 'train' if self.training else 'valid'
        self.log(f'{name}_loss', loss, on_epoch=True)
        self.log(f'{name}_loss_variational', loss_v, on_epoch=True)
        self.log(f'{name}_loss_boundary_A', loss_a, on_epoch=True)
        self.log(f'{name}_loss_boundary_B', loss_b, on_epoch=True)
        self.log(f'{name}_loss_z_range', loss_z_range, on_epoch=True)
        return loss


def test_tlcommittor():
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    data, mapping_names = test_get_data()

    cv = PairTimeLaggedCommittor(mapping_names)

    assert (
        torch.abs(
            cv(data)
            - torch.tensor([[0.10873606283495335, 0.58083648568948]] * 6)
        ) < 1E-12
    ).all()

    assert torch.isinf(
        cv.training_step({'dataset_1': data, 'dataset_2': data})
    )


if __name__ == '__main__':
    test_tlcommittor()
