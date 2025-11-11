import os
import warnings
import torch
import torch_geometric as tg
from lightning import LightningModule
from typing import Dict, Tuple, Optional, Any

from mlcolvar.graph.utils import torch_tools

"""
Helper functions for `torch.export` a model.
"""

__all__ = ['export', 'save_exported', 'load_exported']


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
    Tuple[Dict[str, torch.Tensor]],
    Dict[str, Dict[str, Dict[int, torch.export.Dim]]],
]:

    loader = tg.loader.DataLoader(
        [data], batch_size=1, shuffle=False,
    )
    dd = next(iter(loader)).to(device).to_dict()

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

    return (dd,), {'data': shapes}


def export(
    model: LightningModule,
    example_inputs: tg.data.Data,
    n_nodes_max: Optional[int] = None,
    n_edges_max: Optional[int] = None,
) -> torch.export.ExportedProgram:

    inputs, input_shapes = _get_input_and_shapes(
        example_inputs, n_nodes_max, n_edges_max, model.device
    )

    # I hate monkey patch ...
    scatter_sum = torch_tools.scatter_sum
    scatter_mean = torch_tools.scatter_mean
    torch_tools.scatter_sum = _scatter_sum_static
    torch_tools.scatter_mean = _scatter_mean_static

    exported = torch.export.export(
        model, inputs, dynamic_shapes=input_shapes, strict=False
    )

    torch_tools.scatter_sum = scatter_sum
    torch_tools.scatter_mean = scatter_mean

    return exported


def save_exported(
    exported: torch.export.ExportedProgram,
    file_name: str = 'model.pt2',
) -> str:

    if file_name[-4:] != '.pt2':
        tmp = os.path.splitext(file_name)[0] + '.pt2'
        warnings.warn(
            'renamed file name "{:s}" to "{:s}"!'.format(file_name, tmp)
        )
        file_name = tmp

    output_path = torch._inductor.aoti_compile_and_package(
        exported, package_path=file_name
    )

    return output_path


def load_exported(file_name: str) -> Any:

    model = torch._inductor.aoti_load_package(file_name)

    return model
