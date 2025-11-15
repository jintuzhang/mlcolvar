import os
import json
import uuid
import warnings
import zipfile

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
        data: Tuple[torch.Tensor],
        token: bool = False
    ) -> Tuple[torch.Tensor]:

        data = _tensors_to_dict(data)

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
                retain_graph=False,
                create_graph=False,
            )[0]
            gradients = gradients.unsqueeze(0)

        results = (
            outputs,
            gradients if self._calculate_gradients else torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            )
        )

        return results


class ExportableCommittor(torch.nn.Module):

    def __init__(
        self,
        model: LightningModule,
        calculate_gradients: bool = True,
        calculate_k_bias: bool = False,
        kb_epsilon: float = 1E-14,
        kb_lambda: float = -1.0,
        kb_truncated: bool = False,
        kb_weighted: bool = False,
    ) -> None:

        super().__init__()
        self._model = model
        self._calculate_gradients = calculate_gradients
        self._calculate_k_bias = calculate_k_bias
        self._kb_truncated = kb_truncated
        self._kb_weighted = kb_weighted
        self._kb_epsilon = torch.tensor(
            kb_epsilon, dtype=torch.get_default_dtype()
        )
        self._kb_lambda = torch.tensor(
            kb_lambda, dtype=torch.get_default_dtype()
        )
        self._kb_sigmoid_p = torch.tensor(
            self._model.sigmoid.p, dtype=torch.get_default_dtype()
        )

        if calculate_k_bias and not calculate_gradients:
            raise RuntimeError(
                'Can not calculate k_bias without calculating gradients!'
            )

    def forward(
        self,
        data: Tuple[torch.Tensor],
        token: bool = False
    ) -> Tuple[torch.Tensor]:

        data = _tensors_to_dict(data)

        outputs = self._model(data)

        dtype = outputs.dtype
        device = outputs.device

        z = outputs[0][0]
        q = outputs[0][1]
        lambd = self._kb_lambda.to(device)
        epsilon = self._kb_epsilon.to(device)
        sigmoid_p = self._kb_sigmoid_p.to(device)

        grad_outputs: Optional[List[Optional[torch.Tensor]]] = [
            torch.tensor(1, device=outputs.device)
        ]
        gradients_z = torch.autograd.grad(
            [outputs[0, 0]],
            [data['positions']],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
        )[0]

        gradients_z_2 = torch.pow(gradients_z, 2)

        if self._kb_weighted:
            atomic_masses = self._model.atomic_masses.to(dtype).to(device)
            node_types = torch.where(data['node_attrs'])[1]
            node_masses = atomic_masses[node_types].unsqueeze(-1)
            gradients_z_2 = gradients_z_2 / node_masses

        gradients_z_sum = torch.sum(gradients_z_2)

        if not self._kb_truncated:
            k_bias_value = lambd * (
                torch.log(gradients_z_sum + epsilon)
                - 4.0 * torch.log(1.0 + torch.exp(-sigmoid_p * z))
                - 2.0 * sigmoid_p * z
                - torch.log(epsilon)
            )
        else:
            k_bias_value = lambd * (
                torch.log(
                    gradients_z_sum * torch.pow(q * (1 - q), 2) + epsilon
                )
                - torch.log(epsilon)
            )

        gradients_b = torch.autograd.grad(
            [k_bias_value],
            [data['positions']],
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
        )[0]

        gradients_z = gradients_z.unsqueeze(0)
        gradients_b = gradients_b.unsqueeze(0)

        results = (
            outputs,
            gradients_z if self._calculate_gradients else torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            ),
            k_bias_value if self._calculate_k_bias else torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            ),
            gradients_b if self._calculate_k_bias else torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            ),
        )

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


def _get_inputs(
    data: tg.data.Data,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:

    loader = tg.loader.DataLoader(
        [data], batch_size=1, shuffle=False,
    )
    dd = next(iter(loader)).to(device).to_dict()
    dd['positions'].requires_grad_(True)

    return dd


def _dict_to_tensors(inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:

    dtype = inputs['positions'].dtype
    device = inputs['positions'].device

    outputs = (
        inputs['edge_index'],
        inputs['shifts'],
        inputs['unit_shifts'],
        inputs['positions'],
        inputs['node_attrs'],
        inputs['batch'],
        inputs['weight'],
        inputs['graph_labels'],
        inputs['cell'],
        inputs['ptr'],
        inputs['n_system'],
        (
            inputs['system_masks']
            if 'system_masks' in inputs.keys() else torch.tensor(
                0, device=device, dtype=dtype
            )
        ),
        (
            inputs['subsystem_masks']
            if 'subsystem_masks' in inputs.keys() else torch.tensor(
                0, device=device, dtype=dtype
            )
        ),
        (
            inputs['edge_masks_le']
            if 'edge_masks_le' in inputs.keys() else torch.tensor(
                0, device=device, dtype=dtype
            )
        ),
    )

    return outputs


def _tensors_to_dict(inputs: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:

    outputs = {
        'edge_index': inputs[0],
        'shifts': inputs[1],
        'unit_shifts': inputs[2],
        'positions': inputs[3],
        'node_attrs': inputs[4],
        'batch': inputs[5],
        'weight': inputs[6],
        'graph_labels': inputs[7],
        'cell': inputs[8],
        'ptr': inputs[9],
        'n_system': inputs[10],
    }
    if len(inputs[11].shape) != 0:
        outputs['system_masks'] = inputs[11]
    if len(inputs[12].shape) != 0:
        outputs['subsystem_masks'] = inputs[12]
    if len(inputs[13].shape) != 0:
        outputs['edge_masks_le'] = inputs[13]

    return outputs


def _get_model_summary(
    model_name: str, module: torch.nn.Module, level_max: int, level: int
) -> str:

    result = "  " * (level + 1) + "(" + model_name + "): "

    model_type = str(module.__class__.__name__)
    if model_type in ['Linear', 'TICA']:
        result = result + str(module)
    else:
        result = result + model_type

    if (len(list(module.named_children())) != 0):
        if (level <= level_max):
            result = result + " {\n"
            for s in module.named_children():
                result = result + _get_model_summary(
                    s[0], s[1], level_max, level + 1
                )
            result = result + "  " * (level + 1) + "}\n"
        else:
            result = result + " { ... }\n"
    else:
        result = result + "\n"

    return result


def _get_model_metadata(
    model: LightningModule,
    calculate_gradients: bool,
    k_bias_options: Optional[Dict[str, Any]] = None,
    model_summary_level: int = 3,
) -> Dict[str, str]:

    is_committor = hasattr(model, 'is_committor') and model.is_committor == 1

    metadata = {
        'n_cvs': str(model.n_cvs.item()),
        'cutoff': str(model.cutoff.item()),
        'cutoff_l': str(model.cutoff_l.item()),
        'n_atom_types': str(len(model.atomic_numbers)),
        'float_dtype': str(model.dtype)[-2:],
    }
    for i in range(len(model.atomic_numbers)):
        metadata[
            'atomic_number_{:d}'.format(i)
        ] = str(model.atomic_numbers[i].item())

    metadata['training_time'] = (
        'UTC{:+d} {:d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'
    ).format(
        *(model.training_time.cpu().numpy().tolist()),
    )
    metadata['model_summary'] = _get_model_summary(
        'CV', model, model_summary_level, 0
    )
    metadata['n_parameters'] = str(sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    if is_committor:
        metadata['is_committor'] = str(True)
        for i in range(len(model.atomic_numbers)):
            metadata[
                'atomic_masses_{:d}'.format(i)
            ] = str(model.atomic_masses[i].item())
    if is_committor and k_bias_options is not None:
        metadata_c = {
            'calculate_k_bias': False,
            'kb_epsilon': 1E-14 if model.dtype == torch.float64 else 1E-7,
            'kb_lambda': -1.0,
            'kb_truncated': False,
            'kb_weighted': False,
        }
        metadata_c.update(k_bias_options)
        for k in metadata_c.keys():
            metadata_c[k] = str(metadata_c[k])
        metadata.update(metadata_c)
    else:
        metadata['is_committor'] = str(False)

    metadata['calculate_gradients'] = str(calculate_gradients)

    return metadata


def _update_package_metadata(file_name: str, data: Dict[str, str]) -> None:

    tmp_file_name = str(uuid.uuid4())

    with (
        zipfile.ZipFile(file_name, 'r') as f_in,
        zipfile.ZipFile(tmp_file_name, 'w') as f_out
    ):
        for item in f_in.infolist():
            if len(item.filename.split('metadata')) == 2:
                metadata = json.loads(f_in.read(item.filename))
                metadata.update(data)
                f_out.writestr(item.filename, json.dumps(metadata))
            else:
                f_out.writestr(item.filename, f_in.read(item.filename))

    os.remove(file_name)
    os.rename(tmp_file_name, file_name)


def export(
    model: LightningModule,
    example_inputs: tg.data.Data,
    file_name: str = 'model.pt2',
    calculate_gradients: bool = True,
    k_bias_options: Optional[Dict[str, Any]] = {},
    model_summary_level: int = 3,
) -> str:

    torch._dynamo.allow_in_graph(torch.autograd.grad)
    torch._dynamo.allow_in_graph(torch.autograd.functional.jacobian)

    inputs = _get_inputs(example_inputs, model.device)
    inputs = _dict_to_tensors(inputs)

    # I hate monkey patch ...
    scatter_sum = torch_tools.scatter_sum
    scatter_mean = torch_tools.scatter_mean
    torch_tools.scatter_sum = _scatter_sum_static
    torch_tools.scatter_mean = _scatter_mean_static

    is_committor = hasattr(model, 'is_committor') and model.is_committor == 1

    if is_committor:
        for k in k_bias_options.keys():
            if k in [
                'calculate_k_bias',
                'kb_truncated',
                'kb_weighted',
            ]:
                k_bias_options[k] = bool(k_bias_options[k])
            if k in [
                'kb_epsilon',
                'kb_lambda',
            ]:
                k_bias_options[k] = float(k_bias_options[k])
        exportable = ExportableCommittor(
            model, calculate_gradients, **k_bias_options
        )
    else:
        exportable = ExportableCV(model, calculate_gradients)

    # taken from: https://depyf.readthedocs.io/en/latest/walk_through.html
    def forward_and_backward(
        _inputs: Tuple[torch.Tensor], kwargs: Dict[str, Any] = {}
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

    metadata = _get_model_metadata(
        model, calculate_gradients, k_bias_options, model_summary_level
    )
    output_path = torch._inductor.package.package_aoti(file_name, aot_files)

    _update_package_metadata(file_name, metadata)

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
    model.n_cvs = model.n_out
    model.training_time = torch.zeros(7, dtype=int)
    model.dtype = torch.float64
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
            model_c(_dict_to_tensors(data_dict))[0]
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
            model_c(_dict_to_tensors(data_dict))[1][0] - gradients_1
        ) < 1E-12
    ).all()
    assert (
        torch.abs(
            model_c(_dict_to_tensors(data_dict))[1][1] - gradients_2
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
    model.n_cvs = model.n_out
    model.training_time = torch.zeros(7, dtype=int)
    model.dtype = torch.float64
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
            model_c(_dict_to_tensors(data_dict))[0]
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
            model_c(_dict_to_tensors(data_dict))[1][0] - gradients_1
        ) < 1E-12
    ).all()
    assert (
        torch.abs(
            model_c(_dict_to_tensors(data_dict))[1][1] - gradients_2
        ) < 1E-12
    ).all()

    os.remove('model.pt2')


if __name__ == '__main__':
    test_export_schnet()
    test_export_painn()
