import torch
import warnings
import torch_geometric as tg
from typing import Dict, Any, List, Union, Tuple

from mlcolvar.core.stats import TICA
from mlcolvar.core.nn.utils import Custom_Sigmoid
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from mlcolvar.pairformer.cvs import PairBaseCV
from mlcolvar.pairformer.cvs.cv import test_get_data
from mlcolvar.pairformer.utils import torch_tools

"""
The Deep time-lagged independent component analysis (Deep-TICA) CV based on
Pairformer.
"""

__all__ = ['PairDeepTICA']


class PairDeepTICA(PairBaseCV):
    """
    Pairformer-based time-lagged independent component analysis (Deep-TICA).

    It is a non-linear generalization of TICA in which a feature map is learned
    by a neural network optimized as to maximize the eigenvalues of the
    transfer operator, approximated by TICA. The method is described in [1]_.
    Note that from the point of view of the architecture DeepTICA is similar to
    the SRV [2]_ method.

    Parameters
    ----------
    n_cvs: int
        Number of components of the CV.
    mapping_names: Dict[str, List[str]]
        The node embedding mapping name lists, e.g. the `mapping_names`
        attribute of a `mlcolvar.pairformer.data.PairDataSet` instance.
    model_name: str
        Name of the model.
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
        mapping_names: Dict[str, List[str]],
        model_name: str = 'PairFormerModel',
        model_options: Dict[Any, Any] = {'n_out': 6},
        extra_loss_options: Dict[Any, Any] = {
            'mode': 'sum2', 'n_eig': 0, 'use_sigmoid': False,
        },
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
            n_cvs,
            mapping_names,
            model_name,
            model_options,
            **kwargs
        )

        self._use_sigmoid = extra_loss_options.pop('use_sigmoid', False)
        if self._use_sigmoid:
            self.sigmoid = Custom_Sigmoid(p=3)
            warnings.warn(
                '\n\nThe `use_sigmoid` functionality is for testing only '
                + 'and may lead to wrong results!\n'
            )

        self.loss_fn = ReduceEigenvaluesLoss(**extra_loss_options)

        self.tica = TICA(n_out, n_cvs)

    def forward_nn(
        self,
        data: Dict[str, torch.Tensor],
        return_lengths: bool = False
    ) -> torch.Tensor:
        """
        The forward pass for the NN.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        return_lengths: bool
            If return distances for gradient calculations.
        """

        if not self._exporting:
            data['positions'].requires_grad_(True)

        return self._model(data, return_lengths=return_lengths)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        return_lengths: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        return_lengths: bool
            If return distances for gradient calculations.
        """
        nn_outputs = self.forward_nn(data, return_lengths)

        if return_lengths:
            outputs = self.tica(nn_outputs[0])
            return outputs, nn_outputs[1]
        else:
            outputs = self.tica(nn_outputs)
            return outputs

    def forward_eigenfunctions(
        self,
        data: Dict[str, torch.Tensor],
        return_lengths: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        The forward pass to get the eigenfunctions. This is identical to the
        `forword` method when `use_sigmoid` is not enabled.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        return_lengths: bool
            If return distances for gradient calculations.
        """
        nn_outputs = self.forward_nn(data)
        if self._use_sigmoid:
            nn_outputs = self.sigmoid(nn_outputs)
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

        if self._use_sigmoid:
            nn_outputs_t = self.sigmoid(nn_outputs_t)
            nn_outputs_lag = self.sigmoid(nn_outputs_lag)

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

    data, mapping_names = test_get_data()

    cv = PairDeepTICA(2, mapping_names)

    assert (
        torch.abs(
            cv(data)
            - torch.tensor([[-1.2534413015122814, 1.1660525500707786]] * 6)
        ) < 1E-12
    ).all()

    assert torch.abs(
        cv.training_step({'dataset_1': data, 'dataset_2': data})
        - torch.tensor(0)
    ) < 1E-12


if __name__ == '__main__':
    test_deep_tica()
