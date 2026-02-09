import torch
import numpy as np
import typing as tp

from mlcolvar.pairformer import cvs as pcvs
from mlcolvar.pairformer import data as pdata
from mlcolvar.pairformer import utils as putils

__all__ = ['get_dataset_cv_values', 'get_dataset_cv_gradients']

"""
Analysis utils.
"""


def get_dataset_cv_values(
    model: pcvs.PairBaseCV,
    dataset: pdata.PairDataSet,
    device: str = 'cpu',
    batch_size: int = None,
    show_progress: bool = True,
    progress_prefix: str = 'Calculating CV values'
) -> np.ndarray:
    """
    Get CV values of a given dataset. The calculation will run on the device
    where the model is on.

    Parameters
    ----------
    model: mlcolvar.pairformer.cvs.PairBaseCV
        Collective variable model.
    dataset: mlcovar.pairformer.data.PairDataSet
        Dataset on which to compute the sensitivity analysis.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.
    """
    datamodule = pdata.PairDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    cv_values = []

    if show_progress:
        items = putils.progress.pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()

    with torch.no_grad():
        for batchs in items:
            outputs = model(batchs.to(device).to_dict())
            outputs = outputs.cpu().numpy()
            cv_values.append(outputs)

    return np.concatenate(cv_values)


def get_dataset_cv_gradients(
    model: pcvs.PairBaseCV,
    dataset: pdata.PairDataSet,
    component: int = 0,
    device: str = 'cpu',
    batch_size: int = None,
    show_progress: bool = True,
    progress_prefix: str = 'Calculating CV gradients'
) -> tp.List[np.ndarray]:
    """
    Get gradients of the CV w.r.t. node positions in a given dataset. The
    calculation will run on the device where the model is on.

    Parameters
    ----------
    model: mlcolvar.pairformer.cvs.PairBaseCV
        Collective variable model.
    dataset: mlcovar.pairformer.data.PairDataSet
        Dataset on which to compute the sensitivity analysis.
    component: int
        Component of the CV to analysis.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.
    """
    datamodule = pdata.PairDataModule(
        dataset,
        lengths=(1.0,),
        batch_size=batch_size,
        random_split=False,
        shuffle=False
    )
    datamodule.setup()

    cv_value_gradients = []

    if show_progress:
        items = putils.progress.pbar(
            datamodule.train_dataloader(),
            frequency=0.001,
            prefix=progress_prefix
        )
    else:
        items = datamodule.train_dataloader()

    for batchs in items:
        batch_dict = batchs.to(device).to_dict()
        cv_values = model(batch_dict)
        cv_values = cv_values[:, component]
        grad_outputs = [torch.ones_like(cv_values, device=device)]
        gradients = torch.autograd.grad(
            outputs=[cv_values],
            inputs=[batch_dict['positions']],
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
        )
        graph_sizes = batch_dict['ptr'][1:] - batch_dict['ptr'][:-1]
        gradients = torch.split(
            gradients[0].detach(), graph_sizes.cpu().numpy().tolist()
        )
        gradients = [g.cpu().numpy() for g in gradients]
        cv_value_gradients.extend(gradients)

    return cv_value_gradients
