import warnings
import numpy as np
from typing import Dict

from mlcolvar.graph import cvs as gcvs
from mlcolvar.graph import data as gdata

from .utils import get_dataset_cv_gradients

"""
Sensitivity analysis.
"""

__all__ = ['graph_node_sensitivity']


def graph_node_sensitivity(
    model: gcvs.GraphBaseCV,
    dataset: gdata.GraphDataSet,
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
    model: mlcolvar.graph.cvs.GraphBaseCV
        Collective variable model.
    dataset: mlcovar.graph.data.GraphDataSet
        Dataset on which to compute the sensitivity analysis.
    device: str
        Name of the device.
    batch_size: int
        Batch size used for evaluating the CV.
    show_progress: bool
        If show the progress bar.

    Returns
    -------
    results: dictionary
        Results of the sensitivity analysis, containing 'node_indices',
        'sensitivities', and 'sensitivities_components', ordered according to
        the node indices.

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
    weights = [d['weight'].item() for d in dataset]
    sensitivities_components = [
        w * np.linalg.norm(g, axis=-1) for g, w in zip(gradients, weights)
    ]

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
