import os
import warnings
import torch
import torch._inductor.package
import torch_geometric as tg
from lightning import LightningModule
from typing import Dict, Tuple, Optional, Any, List
from torch.fx.experimental.proxy_tensor import make_fx

from mlcolvar.graph.utils import torch_tools

"""
Helper functions for `torch.export` a model.
"""

__all__ = ['export', 'load_exported']


class ExportableCV(torch.nn.Module):

    def __init__(
        self, model: LightningModule, calculate_gradients: bool = True,
    ) -> None:
        super().__init__()
        self._model = model
        self._calculate_gradients = calculate_gradients

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        token: bool = False
    ) -> Dict[str, torch.Tensor]:

        outputs = self._model(data)

        if outputs.shape[1] > 1:

            def fake_func_wrapper(positions: torch.Tensor) -> torch.Tensor:
                data['positions'] = positions
                outputs = self._model(data)
                return outputs

            gradients = torch.autograd.functional.jacobian(
                fake_func_wrapper,
                data['positions'],
                create_graph=False,
                strict=False,
                vectorize=False,
            )[0]

        else:

            grad_outputs: Optional[List[Optional[torch.Tensor]]] = [
                torch.ones(1, device=outputs.device)
            ]
            gradients = torch.autograd.grad(
                [outputs[0]],
                [data['positions']],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )[0]
            gradients = gradients.unsqueeze(0)

        results = {
            'values': outputs,
            'gradients': gradients if self._calculate_gradients else None,
        }

        return results


def _scatter_sum_static(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:

    return torch.sum(src, dim=dim, keepdim=True)


def _scatter_mean_static(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:

    return torch.mean(src, dim=dim, keepdim=True)


def _get_input_and_shapes(
    data: tg.data.Data,
    n_nodes_max: Optional[int] = None,
    n_edges_max: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Dict[str, Dict[int, torch.export.Dim]]],
]:

    loader = tg.loader.DataLoader(
        [data], batch_size=1, shuffle=False,
    )
    dd = next(iter(loader)).to(device).to_dict()
    dd['positions'].requires_grad_(True)

    dim_node = torch.export.Dim('node', max=n_nodes_max)
    dim_edge = torch.export.Dim('edge', max=n_edges_max)

    shapes = {
        'edge_index': {1: dim_edge},
        'shifts': {0: dim_edge},
        'unit_shifts': {0: dim_edge},
        'positions': {0: dim_node},
        'node_attrs': {0: dim_node},
        'batch': {0: dim_node},
        'weight': {0: torch.export.Dim.STATIC},
        'graph_labels': {0: torch.export.Dim.STATIC},
        'cell': {0: torch.export.Dim.STATIC},
        'ptr': {0: torch.export.Dim.STATIC},
        'n_system': {0: torch.export.Dim.STATIC},
    }
    if 'system_masks' in dd.keys():
        shapes['system_masks'] = {0: dim_node}
    if 'subsystem_masks' in dd.keys():
        shapes['subsystem_masks'] = {0: dim_node}
    if 'edge_masks_le' in dd.keys():
        shapes['edge_masks_le'] = {0: dim_edge}

    return dd, {'data': shapes}


def export(
    model: LightningModule,
    example_inputs: tg.data.Data,
    file_name: str = 'model.pt2',
    calculate_gradients: bool = True,
    n_nodes_max: Optional[int] = None,
    n_edges_max: Optional[int] = None,
) -> str:

    torch._dynamo.allow_in_graph(torch.autograd.grad)
    torch._dynamo.allow_in_graph(torch.autograd.functional.jacobian)

    inputs, input_shapes = _get_input_and_shapes(
        example_inputs, n_nodes_max, n_edges_max, model.device
    )

    # I hate monkey patch ...
    scatter_sum = torch_tools.scatter_sum
    scatter_mean = torch_tools.scatter_mean
    torch_tools.scatter_sum = _scatter_sum_static
    torch_tools.scatter_mean = _scatter_mean_static

    exportable = ExportableCV(model, calculate_gradients)

    # taken from: https://depyf.readthedocs.io/en/latest/walk_through.html
    def forward_and_backward(
        _inputs: Dict[str, torch.Tensor], kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        return exportable(_inputs, False)

    wrapped_function = make_fx(
        forward_and_backward,
        tracing_mode='symbolic',
        _allow_non_fake_inputs=True,
    )
    joint_graph = wrapped_function(inputs, {})
    aot_files = torch._inductor.aot_compile(
        joint_graph, inputs, options={'aot_inductor.package': True}
    )

    if file_name[-4:] != '.pt2':
        tmp = os.path.splitext(file_name)[0] + '.pt2'
        warnings.warn(
            'renamed file name "{:s}" to "{:s}"!'.format(file_name, tmp)
        )
        file_name = tmp

    output_path = torch._inductor.package.package_aoti(file_name, aot_files)

    torch_tools.scatter_sum = scatter_sum
    torch_tools.scatter_mean = scatter_mean

    return output_path


def load_exported(
    file_name: str,
) -> torch._inductor.package.package.AOTICompiledModel:

    model = torch._inductor.aoti_load_package(file_name)

    return model


def test_export_schnet() -> None:

    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    model = __import__('mlcolvar').graph.core.nn.models.SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16,
        aggr='min',
        w_out_after_sum=True
    )
    model.device = 'cpu'

    batch = __import__('mlcolvar').graph.core.nn.models.test_get_data()
    dataset = batch.to_data_list()[0]
    loader = tg.loader.DataLoader(
        [dataset], batch_size=1, shuffle=False,
    )
    data_dict = next(iter(loader)).to_dict()

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.no_grad.pt2',
        calculate_gradients=False,
    )
    model_c = load_exported('model.no_grad.pt2')

    assert (
        torch.abs(
            model_c(data_dict)['values']
            - torch.tensor([[0.3654537816221449, -0.0748265132499575]])
        ) < 1E-12
    ).all()

    os.remove('model.no_grad.pt2')

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.pt2',
    )
    model_c = load_exported('model.pt2')

    data_dict['positions'].requires_grad_(True)
    outputs = model(data_dict)
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [
        torch.tensor(1, device=outputs.device)
    ]
    gradients_1 = torch.autograd.grad(
        [outputs[0, 0]],
        [data_dict['positions']],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=False,
    )[0]
    gradients_2 = torch.autograd.grad(
        [outputs[0, 1]],
        [data_dict['positions']],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=False,
    )[0]

    assert (
        torch.abs(
            model_c(data_dict)['gradients'][0] - gradients_1
        ) < 1E-12
    ).all()
    assert (
        torch.abs(
            model_c(data_dict)['gradients'][1] - gradients_2
        ) < 1E-12
    ).all()

    os.remove('model.pt2')


def test_export_painn() -> None:

    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    model = __import__('mlcolvar').graph.core.nn.models.PaiNNModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_hidden_channels=12,
        w_out_after_sum=True,
        basis_type='gaussian',
    )
    model.device = 'cpu'

    batch = __import__('mlcolvar').graph.core.nn.models.test_get_data()
    dataset = batch.to_data_list()[0]
    loader = tg.loader.DataLoader(
        [dataset], batch_size=1, shuffle=False,
    )
    data_dict = next(iter(loader)).to_dict()

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.no_grad.pt2',
        calculate_gradients=False,
    )
    model_c = load_exported('model.no_grad.pt2')

    assert (
        torch.abs(
            model_c(data_dict)['values']
            - torch.tensor([[0.012601337298479546, -0.0032668391572678087]])
        ) < 1E-12
    ).all()

    os.remove('model.no_grad.pt2')

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.pt2',
    )
    model_c = load_exported('model.pt2')

    data_dict['positions'].requires_grad_(True)
    outputs = model(data_dict)
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [
        torch.tensor(1, device=outputs.device)
    ]
    gradients_1 = torch.autograd.grad(
        [outputs[0, 0]],
        [data_dict['positions']],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=False,
    )[0]
    gradients_2 = torch.autograd.grad(
        [outputs[0, 1]],
        [data_dict['positions']],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=False,
    )[0]

    assert (
        torch.abs(
            model_c(data_dict)['gradients'][0] - gradients_1
        ) < 1E-12
    ).all()
    assert (
        torch.abs(
            model_c(data_dict)['gradients'][1] - gradients_2
        ) < 1E-12
    ).all()

    os.remove('model.pt2')


if __name__ == '__main__':
    test_export_schnet()
    test_export_painn()
