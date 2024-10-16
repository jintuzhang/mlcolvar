import torch
import torch_geometric as tg
from typing import Dict, Any, List

from mlcolvar.core.stats import TICA
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from mlcolvar.graph.cvs import GraphBaseCV
from mlcolvar.graph.cvs.cv import test_get_data
from mlcolvar.graph.utils import torch_tools

"""
The Deep time-lagged independent component analysis (Deep-TICA) CV based on
Graph Neural Networks (GNN).
"""

__all__ = ['GraphDeepTICA']


class GraphDeepTICA(GraphBaseCV):
    """
    Graph neural network-based time-lagged independent component analysis
    (Deep-TICA).

    It is a non-linear generalization of TICA in which a feature map is learned
    by a neural network optimized as to maximize the eigenvalues of the
    transfer operator, approximated by TICA. The method is described in [1]_.
    Note that from the point of view of the architecture DeepTICA is similar to
    the SRV [2]_ method.

    Parameters
    ----------
    n_cvs: int
        Number of components of the CV.
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    n_cvs : int
        Number of collective variables to be trained
    model_name: str
        Name of the GNN model.
    model_options: Dict[Any, Any]
        Model options. Note that the `n_out` key of this dict is REQUIRED,
        which stands for the dimension of the output of the network.
    extra_loss_options: Dict[Any, Any]
        Extra loss function options.
    optimizer_options: Dict[Any, Any]
        Optimizer options.

    References
    ----------
    .. [1] L. Bonati, G. Piccini, and M. Parrinello, "Deep learning the slow
        modes for rare events sampling." PNAS USA 118, e2113533118 (2021)
    .. [2] W. Chen, H. Sidky, and A. L. Ferguson, "Nonlinear discovery of slow
        molecular modes using state-free reversible vampnets."
        JCP 150, 214114 (2019).

    See also
    --------
    mlcolvar.core.stats.TICA
        Time Lagged Indipendent Component Analysis
    mlcolvar.core.loss.ReduceEigenvalueLoss
        Eigenvalue reduction to a scalar quantity
    mlcolvar.utils.timelagged.create_timelagged_dataset
        Create dataset of time-lagged data.
    """
    def __init__(
        self,
        n_cvs: int,
        cutoff: float,
        atomic_numbers: List[int],
        model_name: str = 'GVPModel',
        model_options: Dict[Any, Any] = {'n_out': 6},
        extra_loss_options: Dict[Any, Any] = {'mode': 'sum2', 'n_eig': 0},
        optimizer_options: Dict[Any, Any] = {},
        **kwargs,
    ) -> None:
        if 'n_out' not in model_options.keys():
            raise RuntimeError(
                'The `n_out` key of parameter `model_options` is required!'
            )
        model_options['drop_rate'] = 0.0
        n_out = model_options['n_out']

        if optimizer_options != {}:
            kwargs['optimizer_options'] = optimizer_options

        super().__init__(
            n_cvs, cutoff, atomic_numbers, model_name, model_options, **kwargs
        )

        self.loss_fn = ReduceEigenvaluesLoss(**extra_loss_options)

        self.tica = TICA(n_out, n_cvs)

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
        nn_outputs = self.forward_nn(data)
        outputs = self.tica(nn_outputs)

        return outputs

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

        nn_outputs_t = self.forward_nn(data_t)
        nn_outputs_lag = self.forward_nn(data_lag)

        eigvals, _ = self.tica.compute(
            data=[nn_outputs_t, nn_outputs_lag],
            weights=[data_t['weight'], data_lag['weight']],
            save_params=True
        )

        loss = self.loss_fn(eigvals)
        name = 'train' if self.training else 'valid'
        loss_dict = {f'{name}_loss': loss}
        eig_dict = {
            f'{name}_eigval_{i+1}': eigvals[i] for i in range(len(eigvals))
        }
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        return loss

    def set_regularization(self, c0_reg=1e-6) -> None:
        """
        Add identity matrix multiplied by `c0_reg` to correlation matrix C(0)
        to avoid instabilities in performin Cholesky.

        Parameters
        ----------
        c0_reg : float
            Regularization value for C_0.
        """
        self.tica.reg_C_0 = c0_reg


def test_deep_tica():
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    cv = GraphDeepTICA(
        2,
        0.1,
        [1, 8],
        model_options={
            'n_out': 6,
            'n_bases': 6,
            'n_polynomials': 6,
            'n_layers': 2,
            'n_messages': 2,
            'n_feedforwards': 1,
            'n_scalars_node': 16,
            'n_vectors_node': 8,
            'n_scalars_edge': 16,
            'drop_rate': 0,
            'activation': 'Tanh',
        }
    )

    data = test_get_data()

    assert (
        torch.abs(
            cv(data)
            - torch.tensor([[0.4301124873647384, -0.3866279366944752]] * 6)
        ) < 1E-12
    ).all()

    assert torch.abs(
        cv.training_step({'dataset_1': data, 'dataset_2': data})
        - torch.tensor(0)
    ) < 1E-12


if __name__ == '__main__':
    test_deep_tica()
