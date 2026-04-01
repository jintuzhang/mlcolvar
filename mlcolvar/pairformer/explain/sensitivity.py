import warnings
import numpy as np
from typing import Dict

from mlcolvar.pairformer import cvs as pcvs
from mlcolvar.pairformer import data as pdata

from .utils import get_dataset_cv_gradients, get_dataset_cv_gradients_pair

"""
Sensitivity analysis.
"""

__all__ = ['node_sensitivity', 'pair_sensitivity']


def node_sensitivity(
    model: pcvs.PairBaseCV,
    dataset: pdata.PairDataSet,
    component: int = 0,
    device: str = 'cpu',
    batch_size: int = None,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform a sensitivity analysis by calculating CV gradient w.r.t. nodes'
    positions. This allows us to measure which atom is most important to the
    CV.

    Parameters
    ----------
    model: mlcolvar.pairformer.cvs.PairBaseCV
        Collective variable model.
    dataset: mlcovar.pairformer.data.PairDataSet
        Dataset on which to compute the sensitivity analysis.
    device: str
        Name of the device.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.

    Returns
    -------
    results: dictionary
        Results of the sensitivity analysis, containing 'sensitivities' and
        'sensitivities_components', ordered according to the node indices.

    See also
    --------
    mlcolvar.utils.explain.sensitivity_analysis
        Perform the sensitivity analysis of a feedforward model.
    """

    try:
        device_org = model.device
        model = model.to(device)
    except AttributeError:
        device_org = None

    gradients = get_dataset_cv_gradients(
        model,
        dataset,
        component,
        device,
        batch_size,
        show_progress,
        'Getting gradients'
    )
    sensitivities_components = [np.linalg.norm(g, axis=-1) for g in gradients]

    results = {}
    try:
        sensitivities_components = np.vstack(sensitivities_components)
        results['sensitivities'] = sensitivities_components.mean(axis=0)
    except ValueError:
        warnings.warn(
            'Cannot compute avg. sensitivity for a variable-length dataset. '
        )
        sensitivities_components = np.array(
            sensitivities_components, dtype=object
        )
        results['sensitivities'] = None
    results['sensitivities_components'] = sensitivities_components

    if device_org is not None:
        model = model.to(device_org)

    return results


def pair_sensitivity(
    model: pcvs.PairBaseCV,
    dataset: pdata.PairDataSet,
    component: int = 0,
    device: str = 'cpu',
    batch_size: int = None,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform a sensitivity analysis by calculating CV gradient w.r.t. pairs'
    distances. This allows us to measure which pair is most important to the
    CV.

    Parameters
    ----------
    model: mlcolvar.pairformer.cvs.PairBaseCV
        Collective variable model.
    dataset: mlcovar.pairformer.data.PairDataSet
        Dataset on which to compute the sensitivity analysis.
    device: str
        Name of the device.
    batch_size:
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.

    Returns
    -------
    results: dictionary
        Results of the sensitivity analysis, containing 'sensitivities' and
        'sensitivities_components'.

    See also
    --------
    mlcolvar.utils.explain.sensitivity_analysis
        Perform the sensitivity analysis of a feedforward model.
    """

    try:
        device_org = model.device
        model = model.to(device)
    except AttributeError:
        device_org = None

    gradients = get_dataset_cv_gradients_pair(
        model,
        dataset,
        component,
        device,
        batch_size,
        show_progress,
        'Getting gradients'
    )
    sensitivities_components = gradients
    print(sensitivities_components.shape)

    results = {}
    sensitivities = np.abs(sensitivities_components).mean(axis=0)
    sensitivities = sensitivities * (1 - np.eye(dataset.n_atoms_padded))
    results['sensitivities'] = (sensitivities + sensitivities.T) / 2
    results['sensitivities_components'] = sensitivities_components

    if device_org is not None:
        model = model.to(device_org)

    return results
