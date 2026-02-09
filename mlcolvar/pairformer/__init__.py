"""The Pairformer module."""

# first we disable torch_sparse, since it will cause compile problems
try:
    __ts = __import__('torch_sparse')
except Exception:
    __ts = None

__import__('sys').modules['torch_sparse'] = None

from . import (  # noqa: E402
    utils,
    core,
    data,
    cvs,
    explain,
)

utils.torch_tools.set_default_dtype('float32')
# torch_scatter will cause compile problems.
__import__('torch_geometric').typing.WITH_TORCH_SCATTER = False
# now we restore torch_sparse
__import__('sys').modules['torch_sparse'] = __ts


__all__ = ['utils', 'core', 'data', 'cvs', 'explain']
