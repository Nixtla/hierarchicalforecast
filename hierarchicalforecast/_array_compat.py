"""Array compatibility utilities for NumPy and Dask arrays."""

__all__ = [
    "is_dask_array",
    "get_array_module",
    "to_numpy",
    "eye",
    "zeros",
    "zeros_like",
    "full_like",
    "concatenate",
    "vstack",
    "hstack",
    "diag",
    "matmul",
    "compute_if_dask",
]

from typing import Optional, Union

import numpy as np


def _dask_available():
    """Check if dask is available."""
    try:
        import dask.array  # noqa: F401

        return True
    except ImportError:
        return False


def is_dask_array(arr) -> bool:
    """Check if array is a Dask array."""
    if not _dask_available():
        return False
    import dask.array as da

    return isinstance(arr, da.Array)


def get_array_module(arr):
    """Get the array module (numpy or dask.array) based on array type."""
    if is_dask_array(arr):
        import dask.array as da

        return da
    return np


def to_numpy(arr) -> np.ndarray:
    """Convert array to numpy, computing if necessary."""
    if is_dask_array(arr):
        return arr.compute()
    return np.asarray(arr)


def compute_if_dask(arr):
    """Compute array if it's a Dask array, otherwise return as-is."""
    if is_dask_array(arr):
        return arr.compute()
    return arr


def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype=np.float64,
    like=None,
    chunks: Union[int, str, tuple] = "auto",
):
    """Create identity matrix, using Dask if like is a Dask array."""
    if like is not None and is_dask_array(like):
        import dask.array as da

        return da.eye(N, M=M, k=k, dtype=dtype, chunks=chunks)
    return np.eye(N, M=M, k=k, dtype=dtype)


def zeros(shape, dtype=np.float64, like=None, chunks: Union[int, str, tuple] = "auto"):
    """Create zeros array, using Dask if like is a Dask array."""
    if like is not None and is_dask_array(like):
        import dask.array as da

        return da.zeros(shape, dtype=dtype, chunks=chunks)
    return np.zeros(shape, dtype=dtype)


def zeros_like(arr, dtype=None):
    """Create zeros array with same shape and type as input."""
    xp = get_array_module(arr)
    return xp.zeros_like(arr, dtype=dtype)


def full_like(arr, fill_value, dtype=None):
    """Create full array with same shape as input."""
    xp = get_array_module(arr)
    return xp.full_like(arr, fill_value, dtype=dtype)


def concatenate(arrays, axis=0):
    """Concatenate arrays along axis."""
    if any(is_dask_array(a) for a in arrays):
        import dask.array as da

        return da.concatenate(arrays, axis=axis)
    return np.concatenate(arrays, axis=axis)


def vstack(arrays):
    """Stack arrays vertically."""
    if any(is_dask_array(a) for a in arrays):
        import dask.array as da

        return da.vstack(arrays)
    return np.vstack(arrays)


def hstack(arrays):
    """Stack arrays horizontally."""
    if any(is_dask_array(a) for a in arrays):
        import dask.array as da

        return da.hstack(arrays)
    return np.hstack(arrays)


def diag(arr, k=0):
    """Extract diagonal or construct diagonal array."""
    xp = get_array_module(arr)
    return xp.diag(arr, k=k)


def matmul(a, b):
    """Matrix multiplication supporting Dask arrays."""
    if is_dask_array(a) or is_dask_array(b):
        import dask.array as da

        if not is_dask_array(a):
            a = da.from_array(a)
        if not is_dask_array(b):
            b = da.from_array(b)
        return da.matmul(a, b)
    return np.matmul(a, b)


def numpy_to_dask(arr: np.ndarray, chunks: Union[int, str, tuple] = "auto"):
    """Convert numpy array to dask array."""
    if not _dask_available():
        raise ImportError(
            "dask is required for distributed computation. "
            "Install it with: pip install 'dask[array]'"
        )
    import dask.array as da

    return da.from_array(arr, chunks=chunks)


def ensure_dask(arr, chunks: Union[int, str, tuple] = "auto"):
    """Ensure array is a Dask array, converting if necessary."""
    if is_dask_array(arr):
        return arr
    return numpy_to_dask(arr, chunks=chunks)
