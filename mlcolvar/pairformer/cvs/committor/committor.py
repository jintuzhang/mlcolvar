import torch
import torch_geometric as tg
from typing import Dict, Any, List, Union, Tuple

from mlcolvar.core.nn.utils import Custom_Sigmoid

from mlcolvar.pairformer.cvs import PairBaseCV
from mlcolvar.pairformer.cvs.cv import test_get_data
from mlcolvar.pairformer.cvs.committor.utils import PairCommittorLoss
from mlcolvar.pairformer.utils import torch_tools

"""
Data-driven learning of committor function, based on Pairformer.
"""

__all__ = ['PairCommittor']


class PairCommittor(PairBaseCV):
    """
    Data-driven learning of committor function, based on GNNs.

    The committor function q is expressed as the output of a neural network
    optimized with a self-consistent approach based on the Kolmogorov's
    variational principle for the committor and on the imposition of its
    boundary conditions.

    Parameters
    ----------
    mapping_names: Dict[str, List[str]]
        The node embedding mapping name lists, e.g. the `mapping_names`
        attribute of a `mlcolvar.pairformer.data.PairDataSet` instance.
    atomic_masses : List[float]
        List of masses of all the atoms we are using.
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
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor
        using the committor to study the transition state ensemble",
        Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0

    Notes
    -----
    The atomic masses are mapped by the `atom_names` embedding.

    See also
    --------
    mlcolvar.cvs.committor.Committor
        The feedforward NN based ML committor module.
    mlcolvar.pairformer.cvs.committor.utils.PairCommittorLoss
        Kolmogorov's variational optimization of committor and imposition of
        boundary conditions.
    mlcolvar.pairformer.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    """

    def __init__(
        self,
        mapping_names: Dict[str, List[str]],
        atomic_masses: List[float],
        model_name: str = 'PairFormerModel',
        model_options: Dict[Any, Any] = {},
        extra_loss_options: Dict[Any, Any] = {
            'alpha': 1.0,
            'gamma': 100.0,
            'delta_f': 0.0,
            'sigmoid_p': 3.0,
            'penalty_weight': 1.0,
            'z_threshold': 10.0,
            'n_bootstrap': 5,
            'exclude_boundary_in_loss_v': False
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

        if type(atomic_masses) is torch.Tensor:
            atomic_masses = atomic_masses.detach().clone().to(
                torch.get_default_dtype()
            )
        else:
            atomic_masses = torch.tensor(
                atomic_masses, dtype=torch.get_default_dtype()
            )
        self.register_buffer('atomic_masses', atomic_masses)
        self.register_buffer('is_committor', torch.tensor(1, dtype=int))

        self.sigmoid = Custom_Sigmoid(extra_loss_options.get('sigmoid_p', 3.0))

        self.loss_fn = PairCommittorLoss(
            atomic_masses,
            alpha=float(extra_loss_options.get('alpha', 1.0)),
            gamma=float(extra_loss_options.get('gamma', 10000.0)),
            delta_f=float(extra_loss_options.get('delta_f', 0.0)),
            exclude_boundary_in_loss_v=bool(
                extra_loss_options.get('exclude_boundary_in_loss_v', False)
            ),
        )
        self._z_threshold = float(
            extra_loss_options.get('z_threshold', 10.0)
        )
        self._penalty_weight = float(
            extra_loss_options.get('penalty_weight', 10.0)
        )
        self._n_bootstrap = int(
            extra_loss_options.get('n_bootstrap', 5)
        )

    def forward_nn(
        self,
        data: Dict[str, torch.Tensor],
        return_lengths: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        z = self.forward_nn(data, return_lengths)

        if return_lengths:
            q = self.sigmoid(z[0])
            return torch.hstack([z[0], q]), z[1]
        else:
            q = self.sigmoid(z)
            return torch.hstack([z, q])

    def training_step(
        self, train_batch: tg.data.Batch, *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute and return the training loss and record metrics.

        Parameters
        ----------
        train_batch: torch_geometric.data.Batch
            The data batch.
        """
        torch.set_grad_enabled(True)

        batch_dict = train_batch.to_dict()
        z = self.forward_nn(batch_dict)
        q = self.sigmoid(z)

        loss, loss_var, loss_bound_A, loss_bound_B = self.loss_fn(
            batch_dict, q
        )

        if self.current_epoch < self._n_bootstrap:
            loss = loss_bound_A + loss_bound_B

        over_threshold = torch.relu(z.abs() - self._z_threshold)
        over_threshold = over_threshold[over_threshold > 0]
        if over_threshold.numel() > 0:
            loss_z_range = torch.mean(over_threshold.pow(2))
            loss_z_range = self._penalty_weight * loss_z_range
        else:
            loss_z_range = 0.0
        loss = loss + loss_z_range

        name = 'train' if self.training else 'valid'
        self.log(f'{name}_loss', loss, on_epoch=True)
        self.log(f'{name}_loss_variational', loss_var, on_epoch=True)
        self.log(f'{name}_loss_boundary_A', loss_bound_A, on_epoch=True)
        self.log(f'{name}_loss_boundary_B', loss_bound_B, on_epoch=True)
        self.log(f'{name}_loss_z_range', loss_z_range, on_epoch=True)
        return loss


def test_committor():
    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    data, mapping_names = test_get_data()

    cv = PairCommittor(mapping_names, [1, 2])

    assert (
        torch.abs(
            cv(data)
            - torch.tensor([[0.10873606283495335, 0.58083648568948]] * 6)
        ) < 1E-12
    ).all()

    assert torch.abs(
        cv.training_step(data) - torch.tensor(17.569805172914553)
    ) < 1E-12


if __name__ == '__main__':
    test_committor()
