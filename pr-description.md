# PR: Sparse Summing Matrix (`SMatrix`) for `sparse_s=True`

## Summary

Introduces `SMatrix`, a lightweight sparse wrapper for the hierarchical summing matrix S, replacing the previous approach of converting sparse matrices to dense DataFrames. When `aggregate(..., sparse_s=True)` is used, the S matrix now stays sparse throughout the pipeline — from construction through reconciliation — avoiding expensive dense materialisation.

## New: `SMatrix` class (`hierarchicalforecast/smatrix.py`)

- Wraps a `scipy.sparse.csc_matrix` with row/column labels and an id column
- Lazy materialisation: `to_dense()`, `to_csr()`, and `to_frame()` are cached on first call
- Backward-compatible interface: supports `shape`, `columns`, `__len__`, `__getitem__`, `__array__`, and Jupyter `_repr_html_`
- Works with both Pandas and Polars backends

## Changes

### `HierarchicalReconciliation.reconcile()` (`hierarchicalforecast/core.py`)
- Accepts `SMatrix` as `S_df` in addition to DataFrames
- Fast path: when an `SMatrix` is provided, extracts sparse/dense arrays directly via `to_csr()` / `to_dense()` instead of round-tripping through DataFrame → COO → CSR
- Falls back to existing DataFrame-based conversion when a plain DataFrame is passed

### Tests (`tests/test_utils.py`)
- Updated `test_equality_sparse_non_sparse` to assert `SMatrix` type and compare via `to_frame()`
- New `test_sparse_s_with_polars`: verifies `sparse_s=True` works with Polars input end-to-end
