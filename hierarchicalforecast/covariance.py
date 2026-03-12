"""Covariance estimation methods for hierarchical forecast reconciliation.

This module provides a registry of covariance estimation methods used by
MinTrace and other reconciliation approaches. Each method estimates the
error covariance matrix W from in-sample residuals and/or the structural
matrix S.

Available Methods
-----------------

**Cross-sectional methods** (standard hierarchies):

Methods ``ols``, ``wls_struct``, ``wls_var``, ``sam``/``mint_cov``, and
``shr``/``mint_shrink`` implement the covariance estimators from
Wickramasuriya et al. (2019) [1]_. The ``mint_cov`` and ``mint_shrink``
aliases match the naming in the original MinTrace paper.

=============  ===========  ================================================
Name           Alias for    Description
=============  ===========  ================================================
``ols``                     Identity (OLS reconciliation). Diagonal.
``wls_struct``              Structural scaling (proportional to number of
                            bottom nodes). Diagonal.
``wls_var``                 Variance scaling from in-sample residuals.
                            Diagonal.
``sam``                     Sample covariance of residuals.
``mint_cov``   ``sam``      Alias from the original MinTrace paper.
``shr``                     Schafer-Strimmer shrinkage covariance [2]_.
``mint_shrink`` ``shr``     Alias from the original MinTrace paper.
``bu``                      Bottom-up covariance (zero weight on upper
                            levels).
``oasd``                    Oracle Approximating Shrinkage Diagonal [3]_.
                            Per-element shrinkage of the correlation matrix.
=============  ===========  ================================================

**Temporal methods** (single series, multiple aggregation levels):

Temporal methods follow the framework of Athanasopoulos et al. (2017) [4]_
for temporal hierarchies. The AR(1) correlation models (``strar1``,
``sar1``, ``har1``) are from Nystrup et al. (2020) [5]_. Method names
and definitions follow the FoReco package [6]_.

=============  ================================================
Name           Description
=============  ================================================
``wlsv``       Variance scaling across temporal aggregations.
                Diagonal.
``wlsh``       Per-series variance scaling from residuals.
                Diagonal.
``acov``       Auto-covariance from overlapping residual blocks.
``strar1``     Structural AR(1) correlation model.
``sar1``       Shrinkage AR(1) correlation model.
``har1``       Heterogeneous AR(1) correlation model.
=============  ================================================

**Cross-temporal methods** (both cross-sectional and temporal):

Cross-temporal methods implement the reconciliation framework of
Di Fonzo & Girolimetto (2023) [7]_. Method names match the FoReco
package [6]_.

=============  ================================================
Name           Description
=============  ================================================
``csstr``      Cross-sectional structural scaling. Diagonal.
``testr``      Temporal structural scaling. Diagonal.
``bdshr``      Block-diagonal shrinkage by aggregation order.
``bdsam``      Block-diagonal sample covariance by aggregation
                order.
``sshr``       Series-level block-diagonal shrinkage.
``ssam``       Series-level block-diagonal sample covariance.
``hshr``       High-frequency shrinkage, propagated to full
                cross-temporal dimension.
``hsam``       High-frequency sample covariance, propagated.
``hbshr``      High-frequency bottom-level shrinkage, propagated.
``hbsam``      High-frequency bottom-level sample covariance,
                propagated.
``bshr``       Bottom cross-sectional shrinkage, propagated
                through temporal structure.
``bsam``       Bottom cross-sectional sample covariance,
                propagated.
=============  ================================================

Users can register custom methods via :func:`register_covariance`.

References
----------
.. [1] Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
   "Optimal forecast reconciliation for hierarchical and grouped time series
   through trace minimization". Journal of the American Statistical
   Association, 114(526), 804-819.
   https://doi.org/10.1080/01621459.2018.1448825

.. [2] Schafer, J. & Strimmer, K. (2005). "A shrinkage approach to
   large-scale covariance matrix estimation and implications for functional
   genomics". Statistical Applications in Genetics and Molecular Biology,
   4(1). https://doi.org/10.2202/1544-6115.1175

.. [3] Ando, S. & Xiao, M. (2023). "High-dimensional covariance matrix
   estimation: Shrinkage toward a diagonal target". IMF Working Paper
   WP/23/257. https://doi.org/10.5089/9798400261718.001

.. [4] Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., & Petropoulos, F.
   (2017). "Forecasting with temporal hierarchies". European Journal of
   Operational Research, 262(1), 60-74.
   https://doi.org/10.1016/j.ejor.2017.02.046

.. [5] Nystrup, P., Lindstrom, E., Pinson, P., & Madsen, H. (2020).
   "Temporal hierarchies with autocorrelation for conditional reconciliation".
   European Journal of Operational Research, 280(3), 876-888.
   https://doi.org/10.1016/j.ejor.2019.07.061

.. [6] Girolimetto, D. & Di Fonzo, T. (2024). "FoReco: Forecast
   reconciliation". R package version 1.0.
   https://CRAN.R-project.org/package=FoReco

.. [7] Di Fonzo, T. & Girolimetto, D. (2023). "Cross-temporal forecast
   reconciliation: Optimal combination method and heuristic alternatives".
   International Journal of Forecasting, 39(1), 39-57.
   https://doi.org/10.1016/j.ijforecast.2021.08.004
"""

__all__ = [
    "estimate_covariance",
    "register_covariance",
    "list_covariance_methods",
    "is_diagonal_method",
    "REQUIRES_RESIDUALS",
]

from collections.abc import Callable

import numpy as np

from hierarchicalforecast.utils import (
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)

# Default ridge/epsilon value for diagonal regularization
_DEFAULT_RIDGE: float = 2e-8

# Minimum number of observations required by C++ backends
_MIN_OBS: int = 3

# Registry: method_name -> (callable, requires_residuals, diagonal)
_REGISTRY: dict[str, tuple[Callable, bool, bool]] = {}

# Set of method names that require residuals
REQUIRES_RESIDUALS: set[str] = set()


def register_covariance(
    name: str,
    fn: Callable,
    requires_residuals: bool = True,
    diagonal: bool = False,
):
    """Register a covariance estimation method.

    Args:
        name: Method name (used in MinTrace(method=...)).
        fn: Callable with signature
            ``(S, residuals=None, **kwargs) -> W``
            where S is (n_hiers, n_bottom) and residuals is (n_hiers, n_obs).
            Must return W of shape (n_hiers, n_hiers).
        requires_residuals: Whether this method needs in-sample residuals.
        diagonal: Whether this method always produces a diagonal W matrix.
    """
    _REGISTRY[name] = (fn, requires_residuals, diagonal)
    if requires_residuals:
        REQUIRES_RESIDUALS.add(name)
    elif name in REQUIRES_RESIDUALS:
        REQUIRES_RESIDUALS.discard(name)


def list_covariance_methods() -> list[str]:
    """Return names of all registered covariance methods."""
    return sorted(_REGISTRY.keys())


def is_diagonal_method(method: str) -> bool:
    """Return True if the named method always produces a diagonal W matrix."""
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            f"Available: {list_covariance_methods()}"
        )
    return _REGISTRY[method][2]


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
    fn, needs_res, _diag = _REGISTRY[method]
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

    Also checks for degenerate residuals (nearly all zero) and
    enforces a minimum number of observations for numerical stability.
    """
    if residuals.ndim != 2:
        raise ValueError(
            f"residuals must be 2-d (n_hiers, n_obs), got shape {residuals.shape}"
        )
    if residuals.shape[1] < _MIN_OBS:
        raise ValueError(
            f"residuals must have at least {_MIN_OBS} observations (columns), "
            f"got {residuals.shape[1]}"
        )
    # Check for non-finite values (inf/-inf) that would corrupt C++ backends.
    finite_mask = np.isfinite(residuals) | np.isnan(residuals)
    if not np.all(finite_mask):
        raise ValueError(
            "residuals contain non-finite values (inf/-inf). "
            "Check Y_df for overflow or invalid computations."
        )
    # Check per-series variance to detect overfitting.
    # Use mean of squared residuals (not sum) to avoid scale dependence on n_obs.
    per_series_var = np.nanmean(residuals**2, axis=1)
    zero_prc = np.mean(per_series_var < 1e-8)
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


register_covariance("ols", _cov_ols, requires_residuals=False, diagonal=True)


# --- WLS structural ---
def _cov_wls_struct(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    Wdiag = np.sum(S, axis=1, dtype=np.float64)
    return np.diag(Wdiag)


register_covariance("wls_struct", _cov_wls_struct, requires_residuals=False, diagonal=True)


# --- WLS variance ---
def _cov_wls_var(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    residuals = _validate_residuals(residuals)
    Wdiag = np.nanmean(residuals**2, axis=1, dtype=np.float64)
    Wdiag += _DEFAULT_RIDGE
    return np.diag(Wdiag)


register_covariance("wls_var", _cov_wls_var, requires_residuals=True, diagonal=True)


# --- Sample covariance (full, unregularized) ---
def _cov_sam(
    S: np.ndarray, residuals: np.ndarray | None = None, **kw
) -> np.ndarray:
    """Full sample covariance (same as mint_cov, explicit alias).

    Note: produces a rank-deficient matrix when n_hiers > n_obs.
    In that case, consider using 'shr' or 'mint_shrink' instead.
    """
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
    mint_shr_ridge: float = _DEFAULT_RIDGE,
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
    """Oracle Approximating Shrinkage Diagonal (Ando & Xiao, 2023) [3]_.

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
        Wdiag += _DEFAULT_RIDGE
        return np.diag(Wdiag)

    agg_order = np.asarray(agg_order)
    Wdiag = np.zeros(n, dtype=np.float64)
    for k in np.unique(agg_order):
        mask = agg_order == k
        # Pool variance across all series with this aggregation order
        res_k = residuals[mask]
        var_k = np.nanmean(res_k**2)
        Wdiag[mask] = var_k
    Wdiag += _DEFAULT_RIDGE
    return np.diag(Wdiag)


register_covariance("wlsv", _cov_wlsv, requires_residuals=True, diagonal=True)


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


register_covariance("wlsh", _cov_wlsh, requires_residuals=True, diagonal=True)


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

    Vectorized implementation that handles NaNs by masking.

    Args:
        residuals: (n_series, n_obs) array. May contain NaNs.

    Returns:
        phi: Estimated AR(1) parameter, clipped to [0, 0.99].
    """
    if residuals.shape[1] < 3:
        return 0.0

    valid = ~np.isnan(residuals)
    counts = valid.sum(axis=1)

    # Compute means ignoring NaNs
    means = np.nanmean(residuals, axis=1, keepdims=True)
    r_cent = np.where(valid, residuals - means, 0.0)

    var = np.sum(r_cent**2, axis=1)

    # Lag-1 autocovariance: only where both t and t+1 are valid
    valid_lag = valid[:, :-1] & valid[:, 1:]
    cov1 = np.sum(
        np.where(valid_lag, r_cent[:, :-1] * r_cent[:, 1:], 0.0),
        axis=1,
    )

    good = (counts >= 3) & (var > 1e-12)
    if not np.any(good):
        return 0.0
    return float(np.clip(np.mean(cov1[good] / var[good]), 0.0, 0.99))


def _build_ar1_covariance(
    residuals: np.ndarray,
    std_diag: np.ndarray,
    agg_order: np.ndarray | None,
) -> np.ndarray:
    """Build AR(1) covariance from standard deviations and residuals.

    Shared helper for strar1, sar1, and har1 methods.

    Args:
        residuals: Validated residuals (n_hiers, n_obs).
        std_diag: Per-series standard deviation vector (n_hiers,).
        agg_order: Aggregation order per row (None for no blocking).

    Returns:
        W: Covariance matrix (n_hiers, n_hiers).
    """
    n = residuals.shape[0]

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
    std_diag = np.sqrt(np.sum(S, axis=1, dtype=np.float64))
    return _build_ar1_covariance(residuals, std_diag, agg_order)


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
    std_diag = np.sqrt(np.nanmean(residuals**2, axis=1, dtype=np.float64) + _DEFAULT_RIDGE)
    return _build_ar1_covariance(residuals, std_diag, agg_order)


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
        std_val = np.sqrt(np.nanmean(residuals**2) + _DEFAULT_RIDGE)
        std_diag = np.full(n, std_val, dtype=np.float64)
    else:
        agg_order = np.asarray(agg_order)
        std_diag = np.zeros(n, dtype=np.float64)
        for k in np.unique(agg_order):
            mask = agg_order == k
            var_k = np.nanmean(residuals[mask] ** 2) + _DEFAULT_RIDGE
            std_diag[mask] = np.sqrt(var_k)

    return _build_ar1_covariance(residuals, std_diag, agg_order)


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


register_covariance("csstr", _cov_csstr, requires_residuals=False, diagonal=True)


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


register_covariance("testr", _cov_testr, requires_residuals=False, diagonal=True)


# --- Helpers ---


def _block_diagonal_by_series(
    residuals_3d: np.ndarray,
    n_cs: int,
    n_te: int,
    cov_fn,
) -> np.ndarray:
    """Build block-diagonal W with n_cs blocks of size n_te x n_te.

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


def _block_diagonal_by_agg_order(
    res_3d: np.ndarray,
    n_cs: int,
    n_te: int,
    agg_order_te: np.ndarray | None,
    cov_fn,
) -> np.ndarray:
    """Build block-diagonal W with blocks by temporal aggregation order.

    For each temporal aggregation order k, pools residuals across all
    temporal periods at that frequency to estimate one (n_cs x n_cs)
    cross-sectional covariance. Periods at the same aggregation order
    share the same block. Cross-frequency covariances are zero.

    Args:
        res_3d: (n_cs, n_te, n_obs) residuals.
        n_cs: number of cross-sectional series.
        n_te: number of temporal periods.
        agg_order_te: array mapping each temporal period to its aggregation
            order. If None, falls back to one block per period.
        cov_fn: callable(residuals_2d) -> covariance matrix.

    Returns:
        W: (n_ct, n_ct) block-diagonal matrix.
    """
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
            cov_k = cov_fn(res_pooled)
            if cov_k.ndim == 0:
                cov_k = cov_k.reshape(1, 1)
            for j in te_indices:
                idx = np.arange(n_cs) * n_te + j
                W[np.ix_(idx, idx)] = cov_k
    else:
        for j in range(n_te):
            res_j = res_3d[:, j, :]
            cov_j = cov_fn(res_j)
            if cov_j.ndim == 0:
                cov_j = cov_j.reshape(1, 1)
            idx = np.arange(n_cs) * n_te + j
            W[np.ix_(idx, idx)] = cov_j

    return W


def _safe_cov_sam(res: np.ndarray) -> np.ndarray:
    """Sample covariance with NaN handling."""
    nan_mask = np.isnan(res)
    if np.any(nan_mask):
        return _ma_cov(res, ~nan_mask)
    return np.cov(res) if res.shape[0] > 1 else np.atleast_2d(np.nanvar(res, ddof=1))


def _safe_cov_shr(res: np.ndarray, ridge: float = _DEFAULT_RIDGE) -> np.ndarray:
    """Shrinkage covariance with NaN handling."""
    nan_mask = np.isnan(res)
    if np.any(nan_mask):
        return _shrunk_covariance_schaferstrimmer_with_nans(res, ~nan_mask, ridge)
    return _shrunk_covariance_schaferstrimmer_no_nans(res, ridge)


def _ridge_regularize(W: np.ndarray) -> np.ndarray:
    """Add minimal ridge to ensure positive definiteness.

    Uses a trace-based ridge: eps = max(1e-6, 1e-6 * trace(W) / n).
    This is cheaper than eigendecomposition and scale-adaptive.
    """
    n = W.shape[0]
    trace_val = np.trace(W)
    eps = max(1e-6, 1e-6 * trace_val / n) if n > 0 else 1e-6
    return W + eps * np.eye(n, dtype=np.float64)


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


# --- bdshr/bdsam: Block-diagonal covariance (by temporal period) ---
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
    temporal periods at that frequency to estimate one (n_cs x n_cs)
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
    return _block_diagonal_by_agg_order(res_3d, n_cs, n_te, agg_order_te, _safe_cov_shr)


register_covariance("bdshr", _cov_bdshr, requires_residuals=True)


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
    return _block_diagonal_by_agg_order(res_3d, n_cs, n_te, agg_order_te, _safe_cov_sam)


register_covariance("bdsam", _cov_bdsam, requires_residuals=True)


# --- Sshr/Ssam: Series-level block-diagonal covariance ---

def _make_ct_series_method(cov_fn):
    """Factory for series-level block-diagonal CT methods."""
    def _cov(
        S: np.ndarray,
        residuals: np.ndarray | None = None,
        n_cs: int | None = None,
        n_te: int | None = None,
        S_cs: np.ndarray | None = None,
        S_te: np.ndarray | None = None,
        **kw,
    ) -> np.ndarray:
        n_cs_, n_te_, S_cs_, S_te_ = _validate_ct_args(n_cs, n_te, S_cs, S_te)
        residuals = _validate_residuals(residuals)
        res_3d = _reshape_ct_residuals(residuals, n_cs_, n_te_)
        return _block_diagonal_by_series(res_3d, n_cs_, n_te_, cov_fn)
    return _cov


_cov_Sshr = _make_ct_series_method(_safe_cov_shr)
_cov_Sshr.__doc__ = """Series-level block-diagonal shrinkage covariance.

Constructs n_cs blocks of size n_te x n_te. Each block is the
Schafer-Strimmer shrinkage covariance of residuals across all
temporal periods for that cross-sectional series.
"""
register_covariance("sshr", _cov_Sshr, requires_residuals=True)

_cov_Ssam = _make_ct_series_method(_safe_cov_sam)
_cov_Ssam.__doc__ = """Series-level block-diagonal sample covariance.

Constructs n_cs blocks of size n_te x n_te. Each block is the
sample covariance of residuals across all temporal periods for
that cross-sectional series.
"""
register_covariance("ssam", _cov_Ssam, requires_residuals=True)


# --- hshr/hsam: High-frequency covariance ---
# Estimates (n_cs * n_bottom_te) covariance from high-frequency residuals,
# then propagates through kron(I_n, S_te) to get full (n_ct x n_ct).
# Uses block-wise computation to avoid materializing the full Kronecker product.

def _cov_hf_propagated(
    residuals: np.ndarray,
    n_cs: int,
    n_te: int,
    S_cs: np.ndarray,
    S_te: np.ndarray,
    cov_fn,
) -> np.ndarray:
    """High-frequency covariance propagated through temporal summing matrix.

    Computes W = kron(I_n, S_te) @ cov_hf @ kron(I_n, S_te)' + ridge * I
    using block-wise multiplication to avoid materializing the full
    Kronecker product.
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

    # Block-wise: kron(I_n, S_te) has n_cs diagonal blocks of S_te.
    # W[i,j] block = S_te @ cov_hf[i_block, j_block] @ S_te.T
    n_ct = n_cs * n_te
    n_bt = n_bottom_te
    W = np.zeros((n_ct, n_ct), dtype=np.float64)
    for i in range(n_cs):
        for j in range(i + 1):
            cov_block = cov_hf[i * n_bt:(i + 1) * n_bt, j * n_bt:(j + 1) * n_bt]
            block = S_te @ cov_block @ S_te.T
            W[i * n_te:(i + 1) * n_te, j * n_te:(j + 1) * n_te] = block
            if i != j:
                W[j * n_te:(j + 1) * n_te, i * n_te:(i + 1) * n_te] = block.T

    return _ridge_regularize(W)


def _make_ct_hf_method(cov_fn):
    """Factory for high-frequency CT methods."""
    def _cov(
        S: np.ndarray,
        residuals: np.ndarray | None = None,
        n_cs: int | None = None,
        n_te: int | None = None,
        S_cs: np.ndarray | None = None,
        S_te: np.ndarray | None = None,
        **kw,
    ) -> np.ndarray:
        n_cs_, n_te_, S_cs_, S_te_ = _validate_ct_args(n_cs, n_te, S_cs, S_te)
        residuals = _validate_residuals(residuals)
        return _cov_hf_propagated(residuals, n_cs_, n_te_, S_cs_, S_te_, cov_fn)
    return _cov


_cov_hshr = _make_ct_hf_method(_safe_cov_shr)
_cov_hshr.__doc__ = """High-frequency shrinkage covariance.

Estimates (n_cs * n_bottom_te) shrinkage covariance from high-frequency
residuals across all cross-sectional series, then propagates through
kron(I_n, S_te) to the full CT space. Ridge-regularized.
"""
register_covariance("hshr", _cov_hshr, requires_residuals=True)

_cov_hsam = _make_ct_hf_method(_safe_cov_sam)
_cov_hsam.__doc__ = """High-frequency sample covariance.

Same as hshr but uses unregularized sample covariance.
"""
register_covariance("hsam", _cov_hsam, requires_residuals=True)


# --- hbshr/hbsam: High-frequency bottom time series covariance ---
# Estimates (n_bottom_cs * n_bottom_te) covariance from bottom CS x bottom
# temporal residuals, then propagates through kron(S_cs, S_te).
# Uses the Kronecker mixed-product property: (A kron B)(C kron D) = (AC) kron (BD).

def _cov_hf_bottom_propagated(
    residuals: np.ndarray,
    n_cs: int,
    n_te: int,
    S_cs: np.ndarray,
    S_te: np.ndarray,
    cov_fn,
) -> np.ndarray:
    """High-frequency bottom covariance propagated through full CT summing matrix.

    Computes W = kron(S_cs, S_te) @ cov_bottom_hf @ kron(S_cs, S_te)' + ridge * I
    """
    res_3d = _reshape_ct_residuals(residuals, n_cs, n_te)
    n_bottom_cs = S_cs.shape[1]
    n_bottom_te = S_te.shape[1]
    n_obs = res_3d.shape[2]

    # Extract bottom CS x bottom temporal: (n_bottom_cs, n_bottom_te, n_obs)
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


def _make_ct_hf_bottom_method(cov_fn):
    """Factory for high-frequency bottom CT methods."""
    def _cov(
        S: np.ndarray,
        residuals: np.ndarray | None = None,
        n_cs: int | None = None,
        n_te: int | None = None,
        S_cs: np.ndarray | None = None,
        S_te: np.ndarray | None = None,
        **kw,
    ) -> np.ndarray:
        n_cs_, n_te_, S_cs_, S_te_ = _validate_ct_args(n_cs, n_te, S_cs, S_te)
        residuals = _validate_residuals(residuals)
        return _cov_hf_bottom_propagated(residuals, n_cs_, n_te_, S_cs_, S_te_, cov_fn)
    return _cov


_cov_hbshr = _make_ct_hf_bottom_method(_safe_cov_shr)
_cov_hbshr.__doc__ = """High-frequency bottom time series shrinkage covariance.

Estimates covariance from bottom CS x bottom temporal residuals,
propagates through kron(S_cs, S_te). Ridge-regularized.
"""
register_covariance("hbshr", _cov_hbshr, requires_residuals=True)

_cov_hbsam = _make_ct_hf_bottom_method(_safe_cov_sam)
_cov_hbsam.__doc__ = """High-frequency bottom time series sample covariance.

Same as hbshr but uses unregularized sample covariance.
"""
register_covariance("hbsam", _cov_hbsam, requires_residuals=True)


# --- bshr/bsam: Bottom time series covariance ---
# Estimates (n_bottom_cs * n_te) covariance from bottom CS residuals across
# all temporal levels, then propagates through kron(S_cs, I_{n_te}).
# Uses einsum to contract over bottom CS dimensions without materializing
# the full Kronecker product.

def _cov_bottom_cs_propagated(
    residuals: np.ndarray,
    n_cs: int,
    n_te: int,
    S_cs: np.ndarray,
    S_te: np.ndarray,
    cov_fn,
) -> np.ndarray:
    """Bottom CS covariance propagated through kron(S_cs, I_{n_te}).

    Computes W = kron(S_cs, I) @ cov_bottom_all_te @ kron(S_cs, I)' + ridge * I
    using einsum to contract over bottom CS dimensions without materializing
    the full Kronecker product.
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

    # W = kron(S_cs, I_{n_te}) @ cov_bottom @ kron(S_cs, I_{n_te}).T
    # Use einsum to contract over bottom CS dimensions without materializing
    # the full Kronecker product.
    n_ct = n_cs * n_te
    # Reshape cov_bottom to (n_bottom_cs, n_te, n_bottom_cs, n_te) block view
    C = cov_bottom.reshape(n_bottom_cs, n_te, n_bottom_cs, n_te)
    # Contract: W_block[i,t,j,u] = sum_a sum_b S_cs[i,a] * S_cs[j,b] * C[a,t,b,u]
    W = np.einsum('ia,jb,atbu->itju', S_cs, S_cs, C).reshape(n_ct, n_ct)

    return _ridge_regularize(W)


def _make_ct_bottom_cs_method(cov_fn):
    """Factory for bottom CS CT methods."""
    def _cov(
        S: np.ndarray,
        residuals: np.ndarray | None = None,
        n_cs: int | None = None,
        n_te: int | None = None,
        S_cs: np.ndarray | None = None,
        S_te: np.ndarray | None = None,
        **kw,
    ) -> np.ndarray:
        n_cs_, n_te_, S_cs_, S_te_ = _validate_ct_args(n_cs, n_te, S_cs, S_te)
        residuals = _validate_residuals(residuals)
        return _cov_bottom_cs_propagated(residuals, n_cs_, n_te_, S_cs_, S_te_, cov_fn)
    return _cov


_cov_bshr = _make_ct_bottom_cs_method(_safe_cov_shr)
_cov_bshr.__doc__ = """Bottom time series shrinkage covariance.

Estimates (n_bottom_cs * n_te) shrinkage covariance from bottom CS
residuals across all temporal levels, then propagates through
kron(S_cs, I_{n_te}). Ridge-regularized.
"""
register_covariance("bshr", _cov_bshr, requires_residuals=True)

_cov_bsam = _make_ct_bottom_cs_method(_safe_cov_sam)
_cov_bsam.__doc__ = """Bottom time series sample covariance.

Same as bshr but uses unregularized sample covariance.
"""
register_covariance("bsam", _cov_bsam, requires_residuals=True)
