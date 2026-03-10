"""Covariance estimation methods for hierarchical forecast reconciliation.

This module provides a registry of covariance estimation methods used by
MinTrace and other reconciliation approaches. Each method estimates the
error covariance matrix W from in-sample residuals and/or the structural
matrix S.

Methods fall into three categories:
- **OLS/WLS** (diagonal): `ols`, `wls_struct`, `wls_var`
- **GLS** (full matrix): `mint_cov`, `mint_shrink`, `sam`, `shr`, `oasd`
- **Structural**: `bu` (bottom-up covariance)

Users can register custom methods via `register_covariance`.
"""

__all__ = [
    "estimate_covariance",
    "register_covariance",
    "list_covariance_methods",
    "REQUIRES_RESIDUALS",
]

from collections.abc import Callable

import numpy as np

from hierarchicalforecast.utils import (
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)

# Registry: method_name -> (callable, requires_residuals)
_REGISTRY: dict[str, tuple[Callable, bool]] = {}

# Set of method names that require residuals
REQUIRES_RESIDUALS: set[str] = set()


def register_covariance(
    name: str,
    fn: Callable,
    requires_residuals: bool = True,
):
    """Register a covariance estimation method.

    Args:
        name: Method name (used in MinTrace(method=...)).
        fn: Callable with signature
            ``(S, residuals=None, **kwargs) -> W``
            where S is (n_hiers, n_bottom) and residuals is (n_hiers, n_obs).
            Must return W of shape (n_hiers, n_hiers).
        requires_residuals: Whether this method needs in-sample residuals.
    """
    _REGISTRY[name] = (fn, requires_residuals)
    if requires_residuals:
        REQUIRES_RESIDUALS.add(name)
    elif name in REQUIRES_RESIDUALS:
        REQUIRES_RESIDUALS.discard(name)


def list_covariance_methods() -> list[str]:
    """Return names of all registered covariance methods."""
    return sorted(_REGISTRY.keys())


def estimate_covariance(
    method: str,
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Estimate the covariance matrix W for reconciliation.

    Args:
        method: Name of a registered covariance method.
        S: Summing matrix of shape (n_hiers, n_bottom).
        residuals: In-sample residuals of shape (n_hiers, n_obs).
            Required for residual-based methods.
        **kwargs: Additional keyword arguments passed to the method
            (e.g. ``mint_shr_ridge`` for ``mint_shrink``).

    Returns:
        W: Covariance matrix of shape (n_hiers, n_hiers).

    Raises:
        ValueError: If method is unknown or residuals are missing when required.
    """
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            f"Available: {list_covariance_methods()}"
        )
    fn, needs_res = _REGISTRY[method]
    if needs_res and residuals is None:
        raise ValueError(
            f"Covariance method '{method}' requires residuals. "
            f"Pass y_insample and y_hat_insample."
        )
    return fn(S=S, residuals=residuals, **kwargs)


# ---------------------------------------------------------------------------
# Built-in covariance methods
# ---------------------------------------------------------------------------

def _validate_residuals(residuals: np.ndarray) -> np.ndarray:
    """Validate and return residuals in (n_hiers, n_obs) layout.

    Also checks for degenerate residuals (nearly all zero).
    """
    if residuals.ndim != 2:
        raise ValueError(
            f"residuals must be 2-d (n_hiers, n_obs), got shape {residuals.shape}"
        )
    residuals_sum = np.sum(residuals, axis=1)
    zero_prc = np.mean(np.abs(residuals_sum) < 1e-4)
    if zero_prc > 0.98:
        raise ValueError(
            f"Insample residuals close to 0 (zero_prc={zero_prc:.2f}). "
            "Check Y_df — the model may be overfitting."
        )
    return residuals


# --- OLS ---
def _cov_ols(S: np.ndarray, residuals: np.ndarray | None = None, **kw) -> np.ndarray:
    n_hiers = S.shape[0]
    return np.eye(n_hiers, dtype=np.float64)


register_covariance("ols", _cov_ols, requires_residuals=False)


# --- WLS structural ---
def _cov_wls_struct(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    Wdiag = np.sum(S, axis=1, dtype=np.float64)
    return np.diag(Wdiag)


register_covariance("wls_struct", _cov_wls_struct, requires_residuals=False)


# --- WLS variance ---
def _cov_wls_var(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    residuals = _validate_residuals(residuals)
    n_hiers = S.shape[0]
    Wdiag = np.nanmean(residuals**2, axis=1, dtype=np.float64)
    Wdiag += np.full(n_hiers, 2e-8, dtype=np.float64)
    return np.diag(Wdiag)


register_covariance("wls_var", _cov_wls_var, requires_residuals=True)


# --- Sample covariance (full, unregularized) ---
def _cov_sam(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    """Full sample covariance (same as mint_cov, explicit alias)."""
    residuals = _validate_residuals(residuals)
    nan_mask = np.isnan(residuals)
    if np.any(nan_mask):
        return _ma_cov(residuals, ~nan_mask)
    return np.cov(residuals)


register_covariance("sam", _cov_sam, requires_residuals=True)
# mint_cov is an alias
register_covariance("mint_cov", _cov_sam, requires_residuals=True)


# --- Shrinkage covariance (Schäfer-Strimmer) ---
def _cov_shr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    mint_shr_ridge: float = 2e-8,
    **kw,
) -> np.ndarray:
    """Shrunk covariance (Schäfer-Strimmer, 2005)."""
    residuals = _validate_residuals(residuals)
    nan_mask = np.isnan(residuals)
    if np.any(nan_mask):
        return _shrunk_covariance_schaferstrimmer_with_nans(
            residuals, ~nan_mask, mint_shr_ridge
        )
    return _shrunk_covariance_schaferstrimmer_no_nans(residuals, mint_shr_ridge)


register_covariance("shr", _cov_shr, requires_residuals=True)
# mint_shrink is an alias
register_covariance("mint_shrink", _cov_shr, requires_residuals=True)


# --- Bottom-up covariance ---
def _cov_bu(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    """Bottom-up covariance: W = S @ cov_bottom @ S'.

    Estimates the covariance of the bottom-level residuals and propagates
    it through the summing matrix.
    """
    residuals = _validate_residuals(residuals)
    n_hiers, n_bottom = S.shape
    # Bottom-level residuals are the last n_bottom rows
    res_bottom = residuals[n_hiers - n_bottom:]
    nan_mask = np.isnan(res_bottom)
    if np.any(nan_mask):
        cov_bottom = _ma_cov(res_bottom, ~nan_mask)
    else:
        cov_bottom = np.cov(res_bottom)
    if cov_bottom.ndim == 0:
        cov_bottom = cov_bottom.reshape(1, 1)
    return S @ cov_bottom @ S.T


register_covariance("bu", _cov_bu, requires_residuals=True)


# --- Oracle Approximating Shrinkage Diagonal (OASD) ---
def _cov_oasd(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Oracle Approximating Shrinkage Diagonal (Ando & Xiao, 2023).

    Shrinks each off-diagonal element of the sample correlation matrix
    toward zero individually, then rescales back to covariance scale.
    The shrinkage intensity for each pair (i,j) is:

        rho_ij* = max(0, 1 - kappa * Var(r_ij) / r_ij^2)

    where r_ij is the sample correlation and kappa = n / (n-2).

    This is delegated to C++ for performance.
    """
    from hierarchicalforecast._lib import reconciliation as _lib_recon

    residuals = _validate_residuals(residuals)
    nan_mask = np.isnan(residuals)
    if np.any(nan_mask):
        return _lib_recon._oasd_with_nans(residuals, ~nan_mask)
    return _lib_recon._oasd_no_nans(residuals)


register_covariance("oasd", _cov_oasd, requires_residuals=True)


# --- Temporal covariance methods ---

def _cov_wlsv(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    agg_order: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """WLS with one variance per temporal aggregation level.

    For temporal hierarchies, estimates one variance per aggregation order
    (e.g., one for yearly, one for semi-annual, one for quarterly).
    Series within the same aggregation level share the same variance.

    Args:
        S: Summing matrix (n_temporal_periods, n_bottom_periods).
        residuals: Residuals of shape (n_temporal_periods, n_obs).
        agg_order: Array mapping each row of S to its aggregation order.
            E.g., [4, 2, 2, 1, 1, 1, 1] for a quarterly hierarchy.
    """
    residuals = _validate_residuals(residuals)
    n = S.shape[0]
    if agg_order is None:
        # Fall back to wls_var if no aggregation order provided
        Wdiag = np.nanmean(residuals**2, axis=1, dtype=np.float64)
        Wdiag += 2e-8
        return np.diag(Wdiag)

    agg_order = np.asarray(agg_order)
    Wdiag = np.zeros(n, dtype=np.float64)
    for k in np.unique(agg_order):
        mask = agg_order == k
        # Pool variance across all series with this aggregation order
        res_k = residuals[mask]
        var_k = np.nanmean(res_k**2)
        Wdiag[mask] = var_k
    Wdiag += 2e-8
    return np.diag(Wdiag)


register_covariance("wlsv", _cov_wlsv, requires_residuals=True)


def _cov_wlsh(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """WLS with one variance per series in the hierarchy.

    Each series gets its own variance estimate (identical to wls_var).
    Included for API consistency with FoReco's temporal methods.
    """
    return _cov_wls_var(S=S, residuals=residuals, **kw)


register_covariance("wlsh", _cov_wlsh, requires_residuals=True)


def _cov_acov(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    agg_order: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Block-diagonal auto-covariance per aggregation order.

    For temporal hierarchies, estimates a separate covariance block for
    each aggregation level (e.g., one block for yearly, one for semi-annual,
    one for quarterly). Cross-level covariances are set to zero.

    Args:
        S: Summing matrix (n_temporal_periods, n_bottom_periods).
        residuals: Residuals of shape (n_temporal_periods, n_obs).
        agg_order: Array mapping each row to its aggregation order.
    """
    residuals = _validate_residuals(residuals)
    n = S.shape[0]

    if agg_order is None:
        # Without aggregation info, use full sample covariance
        return _cov_sam(S=S, residuals=residuals)

    agg_order = np.asarray(agg_order)
    W = np.zeros((n, n), dtype=np.float64)

    for k in np.unique(agg_order):
        mask = agg_order == k
        idx = np.where(mask)[0]
        res_k = residuals[mask]
        nan_mask = np.isnan(res_k)
        if np.any(nan_mask):
            cov_k = _ma_cov(res_k, ~nan_mask)
        else:
            cov_k = np.cov(res_k)
        if cov_k.ndim == 0:
            cov_k = cov_k.reshape(1, 1)
        W[np.ix_(idx, idx)] = cov_k

    return W


register_covariance("acov", _cov_acov, requires_residuals=True)


def _build_ar1_correlation(n: int, phi: float) -> np.ndarray:
    """Build an n x n AR(1) correlation matrix with parameter phi."""
    idx = np.arange(n)
    return phi ** np.abs(idx[:, None] - idx[None, :])


def _estimate_ar1_phi(residuals: np.ndarray) -> float:
    """Estimate AR(1) parameter from residuals using lag-1 autocorrelation.

    Args:
        residuals: (n_series, n_obs) array. May contain NaNs.

    Returns:
        phi: Estimated AR(1) parameter, clipped to [0, 0.99].
    """
    phis = []
    for i in range(residuals.shape[0]):
        r = residuals[i]
        valid = ~np.isnan(r)
        r_clean = r[valid]
        if len(r_clean) < 3:
            continue
        r_cent = r_clean - np.mean(r_clean)
        var = np.dot(r_cent, r_cent)
        if var < 1e-12:
            continue
        cov1 = np.dot(r_cent[:-1], r_cent[1:])
        phis.append(cov1 / var)
    if not phis:
        return 0.0
    return float(np.clip(np.mean(phis), 0.0, 0.99))


def _cov_strar1(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    agg_order: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Structural variance + Markov AR(1) correlation.

    Uses structural weights (S row sums) for the diagonal and AR(1)
    correlation for the off-diagonal. Only correlates series within the
    same aggregation level.

    Args:
        S: Summing matrix.
        residuals: Residuals (n_hiers, n_obs). Used for AR(1) estimation.
        agg_order: Aggregation order per row of S.
    """
    residuals = _validate_residuals(residuals)
    n = S.shape[0]
    std_diag = np.sqrt(np.sum(S, axis=1, dtype=np.float64))

    if agg_order is None:
        phi = _estimate_ar1_phi(residuals)
        R = _build_ar1_correlation(n, phi)
        return np.outer(std_diag, std_diag) * R

    agg_order = np.asarray(agg_order)
    W = np.zeros((n, n), dtype=np.float64)

    for k in np.unique(agg_order):
        mask = agg_order == k
        idx = np.where(mask)[0]
        n_k = len(idx)
        phi = _estimate_ar1_phi(residuals[mask])
        R_k = _build_ar1_correlation(n_k, phi)
        std_k = std_diag[idx]
        W[np.ix_(idx, idx)] = np.outer(std_k, std_k) * R_k

    return W


register_covariance("strar1", _cov_strar1, requires_residuals=True)


def _cov_sar1(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    agg_order: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Series variance + Markov AR(1) correlation.

    Uses per-series variance from residuals for the diagonal and AR(1)
    correlation for the off-diagonal within each aggregation level.

    Args:
        S: Summing matrix.
        residuals: Residuals (n_hiers, n_obs).
        agg_order: Aggregation order per row.
    """
    residuals = _validate_residuals(residuals)
    n = S.shape[0]
    std_diag = np.sqrt(np.nanmean(residuals**2, axis=1, dtype=np.float64) + 2e-8)

    if agg_order is None:
        phi = _estimate_ar1_phi(residuals)
        R = _build_ar1_correlation(n, phi)
        return np.outer(std_diag, std_diag) * R

    agg_order = np.asarray(agg_order)
    W = np.zeros((n, n), dtype=np.float64)

    for k in np.unique(agg_order):
        mask = agg_order == k
        idx = np.where(mask)[0]
        n_k = len(idx)
        phi = _estimate_ar1_phi(residuals[mask])
        R_k = _build_ar1_correlation(n_k, phi)
        std_k = std_diag[idx]
        W[np.ix_(idx, idx)] = np.outer(std_k, std_k) * R_k

    return W


register_covariance("sar1", _cov_sar1, requires_residuals=True)


def _cov_har1(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    agg_order: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Hierarchy (pooled) variance + Markov AR(1) correlation.

    Like sar1, but pools variance within each aggregation level (one
    variance per level, not per series). Combined with AR(1) correlation
    within each level.

    Args:
        S: Summing matrix.
        residuals: Residuals (n_hiers, n_obs).
        agg_order: Aggregation order per row.
    """
    residuals = _validate_residuals(residuals)
    n = S.shape[0]

    if agg_order is None:
        std_val = np.sqrt(np.nanmean(residuals**2) + 2e-8)
        phi = _estimate_ar1_phi(residuals)
        R = _build_ar1_correlation(n, phi)
        return std_val**2 * R

    agg_order = np.asarray(agg_order)
    W = np.zeros((n, n), dtype=np.float64)

    for k in np.unique(agg_order):
        mask = agg_order == k
        idx = np.where(mask)[0]
        n_k = len(idx)
        res_k = residuals[mask]
        var_k = np.nanmean(res_k**2) + 2e-8
        std_k = np.sqrt(var_k)
        phi = _estimate_ar1_phi(res_k)
        R_k = _build_ar1_correlation(n_k, phi)
        W[np.ix_(idx, idx)] = std_k**2 * R_k

    return W


register_covariance("har1", _cov_har1, requires_residuals=True)


# ---------------------------------------------------------------------------
# Cross-temporal covariance methods
# ---------------------------------------------------------------------------
#
# These methods operate on the full cross-temporal stacked vector of dimension
# n_ct = n_cs * n_te, where n_cs is the number of cross-sectional series and
# n_te is the number of temporal periods in the hierarchy.
#
# The vector ordering follows the Kronecker convention (S_ct = S_cs ⊗ S_te):
#   (cs_1×te_1, cs_1×te_2, ..., cs_1×te_K, cs_2×te_1, ..., cs_N×te_K)
#
# Required kwargs for all CT methods:
#   n_cs (int): number of cross-sectional series
#   n_te (int): number of temporal periods in hierarchy
#   S_cs (ndarray): cross-sectional summing matrix (n_cs, n_bottom_cs)
#   S_te (ndarray): temporal summing matrix (n_te, n_bottom_te)
# ---------------------------------------------------------------------------


def _validate_ct_args(
    n_cs: int | None,
    n_te: int | None,
    S_cs: np.ndarray | None,
    S_te: np.ndarray | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Validate and return cross-temporal arguments."""
    if n_cs is None or n_te is None:
        raise ValueError(
            "Cross-temporal covariance methods require 'n_cs' and 'n_te' kwargs."
        )
    if S_cs is None or S_te is None:
        raise ValueError(
            "Cross-temporal covariance methods require 'S_cs' and 'S_te' kwargs."
        )
    return n_cs, n_te, np.asarray(S_cs), np.asarray(S_te)


def _reshape_ct_residuals(
    residuals: np.ndarray, n_cs: int, n_te: int
) -> np.ndarray:
    """Reshape (n_ct, n_obs) residuals to (n_cs, n_te, n_obs)."""
    n_ct, n_obs = residuals.shape
    if n_ct != n_cs * n_te:
        raise ValueError(
            f"residuals have {n_ct} rows but n_cs * n_te = {n_cs * n_te}"
        )
    return residuals.reshape(n_cs, n_te, n_obs)


# --- csstr: Cross-sectional structural ---
def _cov_csstr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Cross-sectional structural variance for cross-temporal reconciliation.

    Diagonal matrix where each entry equals the structural weight
    (row sum of S_cs) of the corresponding cross-sectional series,
    replicated across all temporal periods.

    W = diag(kron(S_cs.sum(1), ones(n_te)))
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    cs_weights = np.sum(S_cs, axis=1, dtype=np.float64)
    # Kronecker ordering: repeat each CS weight n_te times
    Wdiag = np.repeat(cs_weights, n_te)
    return np.diag(Wdiag)


register_covariance("csstr", _cov_csstr, requires_residuals=False)


# --- testr: Temporal structural ---
def _cov_testr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Temporal structural variance for cross-temporal reconciliation.

    Diagonal matrix where each entry equals the structural weight
    (row sum of S_te) of the corresponding temporal period,
    replicated across all cross-sectional series.

    W = diag(kron(ones(n_cs), S_te.sum(1)))
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    te_weights = np.sum(S_te, axis=1, dtype=np.float64)
    # Kronecker ordering: tile temporal weights for each CS series
    Wdiag = np.tile(te_weights, n_cs)
    return np.diag(Wdiag)


register_covariance("testr", _cov_testr, requires_residuals=False)


# --- Helpers ---


def _block_diagonal_by_series(
    residuals_3d: np.ndarray,
    n_cs: int,
    n_te: int,
    cov_fn,
) -> np.ndarray:
    """Build block-diagonal W with n_cs blocks of size n_te × n_te.

    Each block captures temporal covariance for one cross-sectional series.

    Args:
        residuals_3d: (n_cs, n_te, n_obs) residuals.
        n_cs: number of cross-sectional series.
        n_te: number of temporal periods.
        cov_fn: callable(residuals_2d) -> (n_te, n_te) covariance matrix.

    Returns:
        W: (n_ct, n_ct) block-diagonal matrix.
    """
    n_ct = n_cs * n_te
    W = np.zeros((n_ct, n_ct), dtype=np.float64)
    for i in range(n_cs):
        # Residuals for series i across all temporal periods
        res_i = residuals_3d[i, :, :]  # (n_te, n_obs)
        cov_i = cov_fn(res_i)
        if cov_i.ndim == 0:
            cov_i = cov_i.reshape(1, 1)
        # Place block: indices for series i are i*n_te, i*n_te+1, ..., i*n_te+(n_te-1)
        idx = np.arange(i * n_te, (i + 1) * n_te)
        W[np.ix_(idx, idx)] = cov_i
    return W


def _safe_cov_sam(res: np.ndarray) -> np.ndarray:
    """Sample covariance with NaN handling."""
    nan_mask = np.isnan(res)
    if np.any(nan_mask):
        return _ma_cov(res, ~nan_mask)
    return np.cov(res) if res.shape[0] > 1 else np.atleast_2d(np.nanvar(res))


def _safe_cov_shr(res: np.ndarray, ridge: float = 2e-8) -> np.ndarray:
    """Shrinkage covariance with NaN handling."""
    nan_mask = np.isnan(res)
    if np.any(nan_mask):
        return _shrunk_covariance_schaferstrimmer_with_nans(res, ~nan_mask, ridge)
    return _shrunk_covariance_schaferstrimmer_no_nans(res, ridge)


def _ridge_regularize(W: np.ndarray) -> np.ndarray:
    """Add minimal ridge to ensure positive definiteness.

    Uses FoReco's strategy: add the smallest positive eigenvalue (above 1e-6)
    as a ridge parameter.
    """
    eigvals = np.linalg.eigvalsh(W)
    pos_eigvals = eigvals[eigvals > 1e-6]
    if len(pos_eigvals) > 0:
        eps = pos_eigvals.min()
    else:
        eps = 1e-6
    return W + eps * np.eye(W.shape[0], dtype=np.float64)


def _extract_bottom_temporal_residuals(
    residuals_3d: np.ndarray,
    S_te: np.ndarray,
) -> np.ndarray:
    """Extract residuals at the bottom (highest frequency) temporal level.

    Args:
        residuals_3d: (n_cs, n_te, n_obs) residuals.
        S_te: temporal summing matrix (n_te, n_bottom_te).

    Returns:
        Bottom temporal residuals of shape (n_cs, n_bottom_te, n_obs).
    """
    n_bottom_te = S_te.shape[1]
    return residuals_3d[:, -n_bottom_te:, :]


def _extract_bottom_cs_residuals(
    residuals_3d: np.ndarray,
    S_cs: np.ndarray,
) -> np.ndarray:
    """Extract residuals at the bottom cross-sectional level.

    Args:
        residuals_3d: (n_cs, n_te, n_obs) residuals.
        S_cs: cross-sectional summing matrix (n_cs, n_bottom_cs).

    Returns:
        Bottom CS residuals of shape (n_bottom_cs, n_te, n_obs).
    """
    n_bottom_cs = S_cs.shape[1]
    return residuals_3d[-n_bottom_cs:, :, :]


# --- bdshr: Block-diagonal shrinkage (by temporal period) ---
def _cov_bdshr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    agg_order_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Block-diagonal shrinkage covariance (blocks by aggregation order).

    For each temporal aggregation order k, pools residuals across all
    temporal periods at that frequency to estimate one (n_cs × n_cs)
    cross-sectional shrinkage covariance. Periods at the same aggregation
    order share the same block. Cross-frequency covariances are zero.

    If agg_order_te is not provided, falls back to one block per
    temporal period (no pooling).

    Args:
        agg_order_te: array mapping each row of S_te to its aggregation
            order. E.g., [4, 2, 2, 1, 1, 1, 1] for a quarterly hierarchy.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)

    n_ct = n_cs * n_te
    W = np.zeros((n_ct, n_ct), dtype=np.float64)

    if agg_order_te is not None:
        agg_order_te = np.asarray(agg_order_te)
        for k in np.unique(agg_order_te):
            te_mask = agg_order_te == k
            te_indices = np.where(te_mask)[0]
            # Pool residuals across all temporal periods at this level
            # res_3d[:, te_indices, :] -> (n_cs, n_periods_at_k, n_obs)
            res_k = res_3d[:, te_indices, :]
            n_periods_k = len(te_indices)
            # Reshape to (n_cs, n_periods_k * n_obs) to pool
            res_pooled = res_k.reshape(n_cs, n_periods_k * res_3d.shape[2])
            cov_k = _safe_cov_shr(res_pooled)
            if cov_k.ndim == 0:
                cov_k = cov_k.reshape(1, 1)
            # Place this block at each temporal period with this order
            for j in te_indices:
                idx = np.arange(n_cs) * n_te + j
                W[np.ix_(idx, idx)] = cov_k
    else:
        # Fallback: one block per temporal period
        for j in range(n_te):
            res_j = res_3d[:, j, :]
            cov_j = _safe_cov_shr(res_j)
            if cov_j.ndim == 0:
                cov_j = cov_j.reshape(1, 1)
            idx = np.arange(n_cs) * n_te + j
            W[np.ix_(idx, idx)] = cov_j

    return W


register_covariance("bdshr", _cov_bdshr, requires_residuals=True)


# --- bdsam: Block-diagonal sample covariance (by aggregation order) ---
def _cov_bdsam(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    agg_order_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Block-diagonal sample covariance (blocks by aggregation order).

    Same structure as bdshr but uses unregularized sample covariance.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)

    n_ct = n_cs * n_te
    W = np.zeros((n_ct, n_ct), dtype=np.float64)

    if agg_order_te is not None:
        agg_order_te = np.asarray(agg_order_te)
        for k in np.unique(agg_order_te):
            te_mask = agg_order_te == k
            te_indices = np.where(te_mask)[0]
            res_k = res_3d[:, te_indices, :]
            n_periods_k = len(te_indices)
            res_pooled = res_k.reshape(n_cs, n_periods_k * res_3d.shape[2])
            cov_k = _safe_cov_sam(res_pooled)
            if cov_k.ndim == 0:
                cov_k = cov_k.reshape(1, 1)
            for j in te_indices:
                idx = np.arange(n_cs) * n_te + j
                W[np.ix_(idx, idx)] = cov_k
    else:
        for j in range(n_te):
            res_j = res_3d[:, j, :]
            cov_j = _safe_cov_sam(res_j)
            if cov_j.ndim == 0:
                cov_j = cov_j.reshape(1, 1)
            idx = np.arange(n_cs) * n_te + j
            W[np.ix_(idx, idx)] = cov_j

    return W


register_covariance("bdsam", _cov_bdsam, requires_residuals=True)


# --- Sshr: Series-level block-diagonal shrinkage ---
def _cov_Sshr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Series-level block-diagonal shrinkage covariance.

    Constructs n_cs blocks of size n_te × n_te. Each block is the
    Schäfer-Strimmer shrinkage covariance of residuals across all
    temporal periods for that cross-sectional series.

    Each series gets its own shrinkage intensity (following FoReco).
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)
    return _block_diagonal_by_series(res_3d, n_cs, n_te, _safe_cov_shr)


register_covariance("Sshr", _cov_Sshr, requires_residuals=True)


# --- Ssam: Series-level block-diagonal sample covariance ---
def _cov_Ssam(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Series-level block-diagonal sample covariance.

    Constructs n_cs blocks of size n_te × n_te. Each block is the
    sample covariance of residuals across all temporal periods for
    that cross-sectional series.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)
    return _block_diagonal_by_series(res_3d, n_cs, n_te, _safe_cov_sam)


register_covariance("Ssam", _cov_Ssam, requires_residuals=True)


# --- hshr/hsam: High-frequency covariance ---
# Estimates (n_cs * n_bottom_te) covariance from high-frequency residuals,
# then propagates through kron(I_n, S_te) to get full (n_ct × n_ct).

def _cov_hf_propagated(
    residuals: np.ndarray,
    n_cs: int,
    n_te: int,
    S_cs: np.ndarray,
    S_te: np.ndarray,
    cov_fn,
) -> np.ndarray:
    """High-frequency covariance propagated through temporal summing matrix.

    Estimates covariance from high-frequency (bottom temporal) residuals
    across ALL cross-sectional series, then propagates via kron(I_n, S_te).

    W = kron(I_n, S_te) @ cov_hf @ kron(I_n, S_te)' + ridge * I
    """
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)
    n_bottom_te = S_te.shape[1]
    n_obs = res_3d.shape[2]

    # Extract high-frequency residuals: (n_cs, n_bottom_te, n_obs)
    res_hf = _extract_bottom_temporal_residuals(res_3d, S_te)
    # Flatten to (n_cs * n_bottom_te, n_obs) in Kronecker order
    res_hf_flat = res_hf.reshape(n_cs * n_bottom_te, n_obs)

    cov_hf = cov_fn(res_hf_flat)
    if cov_hf.ndim == 0:
        cov_hf = cov_hf.reshape(1, 1)

    # Propagate: kron(I_n, S_te) @ cov_hf @ kron(I_n, S_te)'
    kI_Ste = np.kron(np.eye(n_cs, dtype=np.float64), S_te)
    W = kI_Ste @ cov_hf @ kI_Ste.T

    return _ridge_regularize(W)


def _cov_hshr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """High-frequency shrinkage covariance.

    Estimates (n_cs * n_bottom_te) shrinkage covariance from high-frequency
    residuals across all cross-sectional series, then propagates through
    kron(I_n, S_te) to the full CT space. Ridge-regularized.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    return _cov_hf_propagated(residuals, n_cs, n_te, S_cs, S_te, _safe_cov_shr)


register_covariance("hshr", _cov_hshr, requires_residuals=True)


def _cov_hsam(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """High-frequency sample covariance.

    Same as hshr but uses unregularized sample covariance.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    return _cov_hf_propagated(residuals, n_cs, n_te, S_cs, S_te, _safe_cov_sam)


register_covariance("hsam", _cov_hsam, requires_residuals=True)


# --- hbshr/hbsam: High-frequency bottom time series covariance ---
# Estimates (n_bottom_cs * n_bottom_te) covariance from bottom CS × bottom
# temporal residuals, then propagates through kron(S_cs, S_te).

def _cov_hf_bottom_propagated(
    residuals: np.ndarray,
    n_cs: int,
    n_te: int,
    S_cs: np.ndarray,
    S_te: np.ndarray,
    cov_fn,
) -> np.ndarray:
    """High-frequency bottom covariance propagated through full CT summing matrix.

    W = kron(S_cs, S_te) @ cov_bottom_hf @ kron(S_cs, S_te)' + ridge * I
    """
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)
    n_bottom_cs = S_cs.shape[1]
    n_bottom_te = S_te.shape[1]
    n_obs = res_3d.shape[2]

    # Extract bottom CS × bottom temporal: (n_bottom_cs, n_bottom_te, n_obs)
    res_bottom = res_3d[-n_bottom_cs:, -n_bottom_te:, :]
    # Flatten to (n_bottom_cs * n_bottom_te, n_obs)
    res_bottom_flat = res_bottom.reshape(n_bottom_cs * n_bottom_te, n_obs)

    cov_bottom_hf = cov_fn(res_bottom_flat)
    if cov_bottom_hf.ndim == 0:
        cov_bottom_hf = cov_bottom_hf.reshape(1, 1)

    # Propagate through full CT summing matrix: kron(S_cs, S_te)
    S_ct = np.kron(S_cs, S_te)
    W = S_ct @ cov_bottom_hf @ S_ct.T

    return _ridge_regularize(W)


def _cov_hbshr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """High-frequency bottom time series shrinkage covariance.

    Estimates covariance from bottom CS × bottom temporal residuals,
    propagates through kron(S_cs, S_te). Ridge-regularized.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    return _cov_hf_bottom_propagated(residuals, n_cs, n_te, S_cs, S_te, _safe_cov_shr)


register_covariance("hbshr", _cov_hbshr, requires_residuals=True)


def _cov_hbsam(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """High-frequency bottom time series sample covariance.

    Same as hbshr but uses unregularized sample covariance.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    return _cov_hf_bottom_propagated(residuals, n_cs, n_te, S_cs, S_te, _safe_cov_sam)


register_covariance("hbsam", _cov_hbsam, requires_residuals=True)


# --- bshr/bsam: Bottom time series covariance ---
# Estimates (n_bottom_cs * n_te) covariance from bottom CS residuals across
# all temporal levels, then propagates through kron(S_cs, I_{n_te}).

def _cov_bottom_cs_propagated(
    residuals: np.ndarray,
    n_cs: int,
    n_te: int,
    S_cs: np.ndarray,
    S_te: np.ndarray,
    cov_fn,
) -> np.ndarray:
    """Bottom CS covariance propagated through kron(S_cs, I_{n_te}).

    W = kron(S_cs, I) @ cov_bottom_all_te @ kron(S_cs, I)' + ridge * I
    """
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)
    n_bottom_cs = S_cs.shape[1]
    n_obs = res_3d.shape[2]

    # Bottom CS residuals across all temporal levels: (n_bottom_cs, n_te, n_obs)
    res_bottom_cs = _extract_bottom_cs_residuals(res_3d, S_cs)
    # Flatten to (n_bottom_cs * n_te, n_obs)
    res_bottom_flat = res_bottom_cs.reshape(n_bottom_cs * n_te, n_obs)

    cov_bottom = cov_fn(res_bottom_flat)
    if cov_bottom.ndim == 0:
        cov_bottom = cov_bottom.reshape(1, 1)

    # Propagate through kron(S_cs, I_{n_te})
    kScs_I = np.kron(S_cs, np.eye(n_te, dtype=np.float64))
    W = kScs_I @ cov_bottom @ kScs_I.T

    return _ridge_regularize(W)


def _cov_bshr(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Bottom time series shrinkage covariance.

    Estimates (n_bottom_cs * n_te) shrinkage covariance from bottom CS
    residuals across all temporal levels, then propagates through
    kron(S_cs, I_{n_te}). Ridge-regularized.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    return _cov_bottom_cs_propagated(residuals, n_cs, n_te, S_cs, S_te, _safe_cov_shr)


register_covariance("bshr", _cov_bshr, requires_residuals=True)


def _cov_bsam(
    S: np.ndarray,
    residuals: np.ndarray | None = None,
    n_cs: int | None = None,
    n_te: int | None = None,
    S_cs: np.ndarray | None = None,
    S_te: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """Bottom time series sample covariance.

    Same as bshr but uses unregularized sample covariance.
    """
    n_cs, n_te, S_cs, S_te = _validate_ct_args(n_cs, n_te, S_cs, S_te)
    residuals = _validate_residuals(residuals)
    return _cov_bottom_cs_propagated(residuals, n_cs, n_te, S_cs, S_te, _safe_cov_sam)


register_covariance("bsam", _cov_bsam, requires_residuals=True)
