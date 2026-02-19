import os
import json
import uuid
import zipfile
import warnings

import torch
import torch._inductor.package
import torch._inductor.compile_fx
import torch_geometric as tg
from lightning import LightningModule
from torch.fx.experimental.proxy_tensor import make_fx

from typing import Dict, Tuple, Optional, Any, List, Union, Callable

from mlcolvar.pairformer.utils import torch_tools
from mlcolvar.pairformer.data.dataset import __implemented_embeddings__

if os.environ.get('MLCOLVAR_EXPORT_MAXIMUM_OPT') == '1':

    torch._inductor.config.freezing = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm = True
    torch._inductor.config.cuda.compile_opt_level = '-O3'
    if hasattr(
        torch._inductor.config.aot_inductor, 'compile_wrapper_opt_level'
    ):
        torch._inductor.config.aot_inductor.compile_wrapper_opt_level = 'O3'

    os.environ['MLCOLVAR_EXPORT_FLOAT_TOL'] = '1E-4'

# TF32 is very inaccurate for our tasks
torch._inductor.compile_fx._warn_tf32_disabled = lambda *args: None

"""
Helper functions for exporting a model.
"""

__all__ = ['export', 'load_exported']


_MODEL_INPUT_TYPE = eval(
    'Tuple[{}]'.format(''.join(['torch.Tensor,' for _ in range(12)]))
)
_MODEL_OUTPUT_TYPE = eval(
    'Tuple[{}]'.format(''.join(['torch.Tensor,' for _ in range(4)]))
)


class ExportableCV(torch.nn.Module):

    def __init__(
        self,
        model: LightningModule,
        tensors_to_dict_function: Callable,
        calculate_gradients: bool = True,
    ) -> None:

        super().__init__()
        self._model = model
        self._calculate_gradients = calculate_gradients
        self._tensors_to_dict = tensors_to_dict_function

    def forward(
        self,
        data: _MODEL_INPUT_TYPE,
        token: bool = False,
    ) -> _MODEL_OUTPUT_TYPE:

        data = self._tensors_to_dict(data)

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
                allow_unused=True,
                materialize_grads=True,
            )[0]
            gradients = gradients.unsqueeze(0)

        results = (
            outputs,
            gradients if self._calculate_gradients else torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            ),
            torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            ),
            torch.tensor(
                0, device=outputs.device, dtype=outputs.dtype
            ),
        )

        return results


class ExportableCommittor(torch.nn.Module):

    def __init__(
        self,
        model: LightningModule,
        tensors_to_dict_function: Callable,
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

        self._tensors_to_dict = tensors_to_dict_function

    def forward(
        self,
        data: _MODEL_INPUT_TYPE,
        token: bool = False,
    ) -> _MODEL_OUTPUT_TYPE:

        data = self._tensors_to_dict(data)

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
            allow_unused=True,
            materialize_grads=True,
        )[0]

        gradients_z_2 = torch.pow(gradients_z, 2)

        if self._kb_weighted:
            atomic_masses = self._model.atomic_masses.to(dtype).to(device)
            node_types = data['node_attrs'][:, 0]
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
            allow_unused=True,
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


def _get_mic_distances_static(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    cells: torch.Tensor,
    n_edges: torch.Tensor,
    normalize: bool = True,
    eps: float = 1E-15,
) -> Tuple[torch.Tensor, torch.Tensor]:

    n_edges = edge_index.shape[1]
    cells_inv = torch.linalg.pinv(cells.transpose(2, 1))
    cells_inv_nodes = torch.repeat_interleave(cells_inv, n_edges, dim=0)
    cells_nodes = torch.repeat_interleave(
        cells.transpose(2, 1), n_edges, dim=0
    )

    positions_1 = positions[edge_index[0]]
    positions_2 = positions[edge_index[1]]
    positions_1_s = torch.einsum('bi,bij->bj', positions_1, cells_inv_nodes)
    positions_2_s = torch.einsum('bi,bij->bj', positions_2, cells_inv_nodes)
    deltas = positions_1_s - positions_2_s
    unit_shifts = torch.round(deltas)
    shifts = torch.einsum('bi,bij->bj', unit_shifts, cells_nodes)

    vectors = positions_2 - positions_1 + shifts + eps
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

    if normalize:
        vectors = torch.nan_to_num(torch.div(vectors, lengths))

    return vectors, lengths


def _ptr_to_edge_index_fc_static(
    ptr: torch.Tensor, n_nodes_max: int = None
) -> torch.Tensor:

    ptr = ptr.long().flatten()

    idx = torch.arange(n_nodes_max, dtype=torch.long, device=ptr.device)
    src = idx.repeat_interleave(n_nodes_max)
    dst = idx.repeat(n_nodes_max)
    block = torch.stack([src, dst], dim=0)

    return block


def _set_check_dtype_device_cxx(
    model: LightningModule,
) -> LightningModule:

    if model.device.type == 'cpu':
        if torch._inductor.config.cpp.cxx != 'clang++':
            torch._inductor.config.cpp.cxx = 'clang++'
            warnings.warn(
                'You are exporting a PairFormer model on CPU, this will '
                + 'require the clang++ compiler. On Debian-like systems '
                + 'you may install it with command: '
                + '`sudo apt install clang libomp-dev`.'
            )
        if model.dtype != torch.float64:
            raise RuntimeError(
                'You are exporting a PairFormer model on CPU, this will '
                + 'require the model dtype to be float64. You can apply the '
                + '`mlcolvar.pairformer.torch_tools.set_default_dtype` method '
                + 'at the begining of your export script.'
            )

    return model


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

    outputs = (
        inputs['positions'],
        inputs['node_attrs'],
        inputs['cell'],
        inputs['weight'],
        inputs['graph_labels'],
        inputs['ptr'],
        inputs['pair_masks'],
        inputs['n_system_padded'],
        inputs['system_masks_padded'],
        inputs['n_environment_padded'],
        inputs['environment_masks'],
        inputs['centers'],
    )

    return outputs


def _tensors_to_dict(inputs: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:

    outputs = {
        'positions': inputs[0],
        'node_attrs': inputs[1],
        'cell': inputs[2],
        'weight': inputs[3],
        'graph_labels': inputs[4],
        'ptr': inputs[5],
        'pair_masks': inputs[6],
        'n_system_padded': inputs[7],
        'system_masks_padded': inputs[8],
        'n_environment_padded': inputs[9],
        'environment_masks': inputs[10],
        'centers': inputs[11],
    }

    return outputs


def _get_model_summary(
    model_name: str, module: torch.nn.Module, level_max: int, level: int
) -> str:

    result = "  " * (level + 1) + "(" + model_name + "): "

    model_type = str(module.__class__.__name__)
    if model_type in ['Linear', 'TICA', 'Embedding']:
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
    n_atoms_padded: int,
    n_atoms_padded_environment: int,
    calculate_gradients: bool,
    k_bias_options: Optional[Dict[str, Any]] = None,
    model_summary_level: int = 3,
) -> Dict[str, str]:

    is_committor = hasattr(model, 'is_committor') and model.is_committor == 1

    metadata = {
        'n_cvs': str(model.n_cvs.item()),
        'float_dtype': str(model.dtype)[-2:],
    }

    metadata['training_time'] = (
        'UTC{:+d} {:d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'
    ).format(
        *(model.training_time.cpu().numpy().tolist()),
    )
    metadata['exporting_time'] = (
        'UTC{:+d} {:d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'
    ).format(
        *(model._get_time()),
    )
    metadata['model_summary'] = _get_model_summary(
        'CV', model, model_summary_level, 0
    )
    metadata['n_parameters'] = str(sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    mapping = model._model._mapping_names
    metadata['n_embeddings'] = str(len(mapping.keys()))
    metadata['n_atoms_padded'] = str(n_atoms_padded)

    count = 0
    for emb in __implemented_embeddings__:
        if emb in mapping.keys():
            metadata['embedding_{:d}'.format(count)] = emb
            metadata['n_{:s}'.format(emb)] = str(len(mapping[emb]))
            for j in range(len(mapping[emb])):
                metadata['{:s}_{:d}'.format(emb, j)] = mapping[emb][j]
            count += 1

    if is_committor:
        metadata['is_committor'] = str(True)
        for i in range(len(model._mapping_names['atom_names'])):
            metadata[
                'atomic_masses_{:d}'.format(i)
            ] = str(model.atomic_masses[i].item())
        metadata_c = {}
        for k in k_bias_options.keys():
            metadata_c[k] = str(k_bias_options[k])
        metadata.update(metadata_c)
    else:
        metadata['is_committor'] = str(False)

    if model._model.cn_layer is not None:
        assert n_atoms_padded_environment > 0, (
            'the CN model requires a positive `n_atoms_padded_environment` '
            + 'value! You may use the `DUMPATOMS` functionality of PLUMED to '
            + 'estimate this value.'
        )

        metadata['d_max'] = str(model._model.cn_layer.d_max.item())
        metadata['n_centers'] = str(model._model._n_centers)
        metadata['n_atoms_padded_environment'] = str(
            n_atoms_padded_environment
        )
    else:
        metadata['d_max'] = str(0.0)
        metadata['n_centers'] = str(0)
        metadata['n_atoms_padded_environment'] = str(0)

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


def _regularize_k_bias_options(
    model: LightningModule,
    k_bias_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    results = {
        'calculate_k_bias': False,
        'kb_epsilon': 1E-14 if model.dtype == torch.float64 else 1E-7,
        'kb_lambda': -1.0,
        'kb_truncated': False,
        'kb_weighted': False,
    }

    if k_bias_options is not None:
        for k in k_bias_options.keys():
            if k in ['calculate_k_bias', 'kb_truncated', 'kb_weighted']:
                results[k] = bool(k_bias_options[k])
            if k in ['kb_epsilon', 'kb_lambda']:
                results[k] = float(k_bias_options[k])

    return results


def _check_exported_model_outputs(
    file_name: str,
    model: Union[ExportableCV, ExportableCommittor],
    example_inputs: _MODEL_INPUT_TYPE,
) -> None:

    print('Export precision check:')

    def check_mae(x: float, dtype: str, prefix: str) -> bool:
        if dtype == '32':
            tol = eval(os.environ.get('MLCOLVAR_EXPORT_FLOAT_TOL', '1E-6'))
        elif dtype == '64':
            tol = eval(os.environ.get('MLCOLVAR_EXPORT_FLOAT_TOL', '1E-12'))
        else:
            raise RuntimeError('Unknown dtype ' + dtype)

        if x > tol:
            raise RuntimeError(
                'MAE ({:e}) of {:s} is larger than '.format(mae, prefix)
                + '{:e} for a float{:s} model!'.format(tol, dtype)
            )
        else:
            print('  MAE of {:s}: {:e}'.format(prefix, mae))

    aot_model = load_exported(file_name)
    metadata = aot_model.get_metadata()
    float_dtype = metadata['float_dtype']

    model_outputs = model(example_inputs)
    aot_model_outputs = aot_model(example_inputs)

    delta = model_outputs[0] - aot_model_outputs[0]
    mae = delta.abs().max().item()
    check_mae(mae, float_dtype, 'CV values')

    if (
        (not eval(metadata['is_committor']))
        and eval(metadata['calculate_gradients'])
    ):

        for i in range(eval(metadata['n_cvs'])):
            delta_i = model_outputs[1][i] - aot_model_outputs[1][i]
            mae = delta_i.abs().max().item()
            check_mae(mae, float_dtype, 'CV gradients {:d}'.format(i))

    elif eval(metadata['is_committor']):

        delta_z = model_outputs[1][0] - aot_model_outputs[1][0]
        mae = delta_z.abs().max().item()
        check_mae(mae, float_dtype, 'CV gradients')

        if eval(metadata['calculate_k_bias']):

            delta_k = model_outputs[2] - aot_model_outputs[2]
            mae = delta_k.abs().max().item()
            check_mae(mae, float_dtype, 'KBias')

            delta_k = model_outputs[3][0] - aot_model_outputs[3][0]
            mae = delta_k.abs().max().item()
            check_mae(mae, float_dtype, 'KBias gradients')


def export(
    model: LightningModule,
    example_inputs: tg.data.Data,
    file_name: str = 'model.pt2',
    calculate_gradients: bool = True,
    k_bias_options: Optional[Dict[str, Any]] = None,
    n_atoms_padded: int = 0,
    n_atoms_padded_environment: int = 0,
    model_summary_level: int = 3,
) -> str:
    """
    Export a CV model using symbolic tracing and Ahead-Of-Time (AOT)
    compilation. By now, this is the only method to export Pairformer models.

    Parameters
    ----------
    model: lightning.LightningModule
        The CV model.
    example_inputs: torch_geometric.data.Data
        Example input data.
    file_name: str
        Name of the exported model file. Note that the name should contain a
        `.pt2` extension.
    calculate_gradients: bool
        If include gradient calculations in the exported model. Do NOT disable
        this option if you do not know that you are doing!
    k_bias_options: Dict[str, Any]
        Options for calculating the Kolmogorov bias ($V_K) for a committor
        model. Available fields are:
        - 'calculate_k_bias': bool
            If calculate the Kolmogorov bias.
        - 'kb_epsilon': float
            The epsilon value for calculating the Kolmogorov bias.
        - 'kb_lambda': float,
            The lambda value for calculating the Kolmogorov bias.
        - 'kb_truncated': False,
            If calculate the truncated (twisted) Kolmogorov bias.
        - 'kb_weighted': False,
            If calculate the mass-weighted (exact) Kolmogorov bias.
    n_atoms_padded: int
        Number of nodes after padding.
    n_atoms_padded_environment: int
        Number of environment nodes after padding.

    Notes
    -----
    A few remarks are in order:

    1. dtype and running device of the model will be fixed after export, which
    means that one can not exporting a model stored on CPU and then inference
    on GPU. To ensure that the exported model will has the desired dtype and/or
    running device, one should move the model before the export:

    ```python
    model = mlcolvar.pairformer.cvs.PairDeepTICA(...)
    model = model.to(troch.float64).to('CUDA')
    dataset = mlcolvar.pairformer.data.load_dataet(...)
    mlcolvar.pairformer.utils.export.export(cv, example_inputs=dataset[0])
    ```

    2. The exported models are generally MUCH FASTER when running on GPUs. Thus
    it is strongly recommended to compile your `PLUMED` interface using the
    CUDA version of LibTorch.

    3. Unlike the JIT compilation, where the gradient calculation of the model
    is done by `torch.autograd.grad` calls in a on-the-fly manner, computation
    graph of the gradient calculation in exported models are statically
    compiled. As a result, one can not change the Kolmogorov bias calculation
    parameters at run time. Instead, these parameters should be given at
    compilation time:

    ```python
    model = mlcolvar.pairformer.cvs.PairCommittor(...)
    dataset = mlcolvar.pairformer.data.load_dataet(...)
    mlcolvar.pairformer.utils.export.export(
        cv,
        example_inputs=dataset[0],
        file_name='model.k_bisa_lambda_1.0.pt2',
        k_bias_options={'calculate_k_bias': True, 'kb_lambda': -1.0},
    )
    ```

    If one would like the change these parameters, the model should be
    re-exported using the updated parameters.

    4. When using CUDA, the export operation requires the CUDA library. Make
    sure that PyTorch is able to find the library, e.g., setting the following
    environment variables before doing the export:

    ```bash
    export CUDA_HOME=/usr/local/cuda-12.9
    export PATH=$PATH:/usr/local/cuda-12.9/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.9/lib64
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/cuda-12.9/include
    ```

    5. The exported models are NOT portable, they will not run on machines
    other than the one where there were exported.

    6. If exporting on CPU is desired, a `clang++` compiler should be
    installed. On Debian-like systems, this compiler can be installed with
    command:
    ```bash
    sudo apt install clang libomp-dev
    ```
    Besides, the model precision has to be float64.
    """

    model = _set_check_dtype_device_cxx(model)

    torch._dynamo.allow_in_graph(torch.autograd.grad)
    torch._dynamo.allow_in_graph(torch.autograd.functional.jacobian)

    is_committor = hasattr(model, 'is_committor') and model.is_committor == 1

    inputs = _get_inputs(example_inputs, model.device)
    inputs = _dict_to_tensors(inputs)

    if n_atoms_padded <= 0:
        warnings.warn(
            'Exporting a pairformer model without exact `n_atoms_padded` '
            + 'given! Will estimate this parameter from example inputs! '
            + 'This might lead to wrong results!'
        )
        n_atoms_padded = inputs[0].shape[0]

    # I hate monkey patch ...
    model._exporting = True
    scatter_sum = torch_tools.scatter_sum
    scatter_mean = torch_tools.scatter_mean
    get_mic_distances = torch_tools.get_mic_distances
    ptr_to_edge_index_fc = torch_tools.ptr_to_edge_index_fc
    torch_tools.scatter_sum = _scatter_sum_static
    torch_tools.scatter_mean = _scatter_mean_static
    torch_tools.get_mic_distances = _get_mic_distances_static
    torch_tools.ptr_to_edge_index_fc = _ptr_to_edge_index_fc_static

    if is_committor:
        k_bias_options = _regularize_k_bias_options(model, k_bias_options)
        exportable = ExportableCommittor(
            model, _tensors_to_dict, calculate_gradients, **k_bias_options
        )
    else:
        exportable = ExportableCV(model, _tensors_to_dict, calculate_gradients)

    # taken from: https://depyf.readthedocs.io/en/latest/walk_through.html
    def forward_and_backward(
        _inputs: _MODEL_INPUT_TYPE, kwargs: Dict[str, Any] = {}
    ) -> _MODEL_OUTPUT_TYPE:
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

    metadata = _get_model_metadata(
        model,
        n_atoms_padded,
        n_atoms_padded_environment,
        calculate_gradients,
        k_bias_options,
        model_summary_level,
    )
    _update_package_metadata(file_name, metadata)
    _check_exported_model_outputs(file_name, exportable, inputs)

    torch_tools.scatter_sum = scatter_sum
    torch_tools.scatter_mean = scatter_mean
    torch_tools.get_mic_distances = get_mic_distances
    torch_tools.ptr_to_edge_index_fc = ptr_to_edge_index_fc
    model._exporting = False

    return output_path


def load_exported(
    file_name: str,
) -> torch._inductor.package.package.AOTICompiledModel:
    """
    Load an exported CV model.

    Parameters
    ----------
    file_name: str
        Name of the `.pt2` file.
    """

    model = torch._inductor.aoti_load_package(file_name)

    return model


def test_export_1() -> None:

    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')

    import mlcolvar.pairformer as mpair

    batch, mapping_names = mpair.core.nn.models.test_get_data()

    model = mpair.cvs.PairBaseCV(2, mapping_names)

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
            - torch.tensor([[0.771122634223133, -0.2714238083585388]])
        ) < 1E-12
    ).all()

    os.remove('model.no_grad.pt2')

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.pt2',
        n_atoms_padded=10,
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
            model_c(_dict_to_tensors(data_dict))[0]
            - torch.tensor([[0.771122634223133, -0.2714238083585388]])
        ) < 1E-12
    ).all()
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


def test_export_2() -> None:

    torch.manual_seed(0)
    torch_tools.set_default_dtype('float64')
    os.environ['MLCOLVAR_EXPORT_FLOAT_TOL'] = '1E-10'

    import mlcolvar.pairformer as mpair

    batch, mapping_names = mpair.core.nn.models.test_get_data(True)

    model = mpair.cvs.PairBaseCV(
        2,
        mapping_names,
        model_options={
            'cn_options': {
                'n': 6,
                'm': 12,
                'r_0': 0.09,
                'd_0': 0,
                'd_max': 0.1,
                'n_centers': 1,
            },
        }
    )

    dataset = batch.to_data_list()[0]
    loader = tg.loader.DataLoader(
        [dataset], batch_size=1, shuffle=False,
    )
    data_dict = next(iter(loader)).to_dict()

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.pt2',
        n_atoms_padded=3,
        n_atoms_padded_environment=3,
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
            model_c(_dict_to_tensors(data_dict))[0]
            - torch.tensor([[-0.05064218647162956, 0.49252906877217784]])
        ) < 1E-12
    ).all()
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
    assert (
        model_c(_dict_to_tensors(data_dict))[1][0][1:3, :] == 0.0
    ).all()
    assert (
        model_c(_dict_to_tensors(data_dict))[1][0][-1, :] == 0.0
    ).all()

    os.remove('model.pt2')

    model = mpair.cvs.PairCommittor(
        mapping_names,
        [1.0, 1.0],
        model_options={
            'cn_options': {
                'n': 6,
                'm': 12,
                'r_0': 0.09,
                'd_0': 0,
                'd_max': 0.1,
                'n_centers': 1,
            },
        }
    )

    dataset = batch.to_data_list()[0]
    loader = tg.loader.DataLoader(
        [dataset], batch_size=1, shuffle=False,
    )
    data_dict = next(iter(loader)).to_dict()

    export(
        model,
        example_inputs=batch.to_data_list()[0],
        file_name='model.pt2',
        k_bias_options={
            'calculate_k_bias': True,
            'kb_epsilon': 1E-10,
            'kb_lambda': -3.0,
            'kb_weighted': True,
            'kb_truncated': False,
        },
        n_atoms_padded=3,
        n_atoms_padded_environment=3,
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

    assert (
        torch.abs(
            model_c(_dict_to_tensors(data_dict))[1][0] - gradients_1
        ) < 1E-12
    ).all()
    assert (
        model_c(_dict_to_tensors(data_dict))[1][0][1:3, :] == 0.0
    ).all()
    assert (
        model_c(_dict_to_tensors(data_dict))[1][0][-1, :] == 0.0
    ).all()
    assert (
        model_c(_dict_to_tensors(data_dict))[3][0][1:3, :] == 0.0
    ).all()
    assert (
        model_c(_dict_to_tensors(data_dict))[3][0][-1, :] == 0.0
    ).all()


if __name__ == '__main__':
    test_export_1()
    test_export_2()
