from __future__ import annotations

"""Sparse summing matrix wrapper for hierarchical forecasting."""

__all__ = ["SMatrix"]


from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
from scipy import sparse


class SMatrix:
    """Lightweight wrapper around a scipy.sparse summing matrix with labels.

    Stores the hierarchical summing matrix S natively as a
    ``scipy.sparse.csc_matrix``.  Dense NumPy arrays and
    Pandas/Polars DataFrames are materialised only on demand via
    :meth:`to_dense` and :meth:`to_frame`.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.spmatrix
        Summing matrix of shape ``(n_series, n_bottom)``.
    row_labels : np.ndarray
        Unique-id labels for every row (all hierarchical series).
    col_labels : np.ndarray
        Unique-id labels for every column (bottom-level series).
    id_col : str
        Name of the identifier column (default ``"unique_id"``).
    backend : str or nw.Implementation
        Narwhals backend hint (``"pandas"`` or ``"polars"``).
    """

    def __init__(
        self,
        sparse_matrix: sparse.spmatrix,
        row_labels: np.ndarray,
        col_labels: np.ndarray,
        id_col: str = "unique_id",
        backend: Any = "pandas",
    ):
        self._sparse = sparse.csc_matrix(sparse_matrix)
        self.row_labels = np.asarray(row_labels)
        self.col_labels = np.asarray(col_labels)
        self.id_col = id_col
        self.backend = backend
        # Lazily cached dense / frame representations
        self._dense: np.ndarray | None = None
        self._frame = None

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def to_sparse(self) -> sparse.csc_matrix:
        """Return the underlying sparse matrix (zero-copy)."""
        return self._sparse

    def to_csr(self) -> sparse.csr_matrix:
        """Return the matrix as CSR (efficient for row slicing)."""
        return sparse.csr_matrix(self._sparse)

    def to_dense(self) -> np.ndarray:
        """Return a dense ``float64`` NumPy array, cached after first call."""
        if self._dense is None:
            self._dense = self._sparse.toarray().astype(np.float64, copy=False)
        return self._dense

    def to_frame(self, backend: str | None = None):
        """Materialise S as a Pandas or Polars DataFrame.

        The result is cached on the first call (per backend).

        Parameters
        ----------
        backend : str, optional
            ``"pandas"`` or ``"polars"``.  Defaults to the backend that
            was used to create the original input data.

        Returns
        -------
        DataFrame
            A native Pandas or Polars DataFrame with the id column
            followed by one column per bottom-level series.
        """
        backend = backend or self.backend
        if self._frame is not None:
            return self._frame

        dense = self.to_dense()
        data = {self.id_col: self.row_labels}
        for j, col_name in enumerate(self.col_labels):
            data[col_name] = dense[:, j]

        frame_nw = nw.from_dict(data, backend=backend)
        frame_nw = nw.maybe_reset_index(frame_nw)
        self._frame = frame_nw.to_native()
        return self._frame

    # ------------------------------------------------------------------
    # Backward-compatible properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_series, n_bottom)`` — shape of the underlying sparse matrix."""
        return self._sparse.shape

    @property
    def columns(self):
        """Column names matching the DataFrame representation."""
        cols = [self.id_col] + list(self.col_labels)
        # Return a list (works for both pandas Index and plain list comparisons)
        return cols

    def __len__(self) -> int:
        return self._sparse.shape[0]

    def __getitem__(self, key):
        """Support ``S_matrix['unique_id']`` and ``S_matrix[col_name]``."""
        if isinstance(key, str):
            if key == self.id_col:
                return self.row_labels
            idx = np.searchsorted(self.col_labels, key)
            if idx < len(self.col_labels) and self.col_labels[idx] == key:
                return self._sparse[:, idx].toarray().ravel()
            raise KeyError(key)
        # Fall back to frame for complex indexing
        return self.to_frame()[key]

    def __repr__(self) -> str:
        nnz = self._sparse.nnz
        density = nnz / max(1, self._sparse.shape[0] * self._sparse.shape[1])
        return (
            f"SMatrix(shape={self.shape}, nnz={nnz}, "
            f"density={density:.4f}, id_col='{self.id_col}')"
        )

    def _repr_html_(self) -> str:
        """Jupyter notebook display — show truncated DataFrame."""
        return self.to_frame()._repr_html_()

    def __array__(self, dtype=None):
        """Support ``np.asarray(S_matrix)`` — returns dense values without id col."""
        arr = self.to_dense()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr
