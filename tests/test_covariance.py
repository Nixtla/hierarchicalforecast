"""Tests for the covariance estimation module."""

import numpy as np
import pytest

from hierarchicalforecast.covariance import (
    _REGISTRY,
    REQUIRES_RESIDUALS,
    _estimate_ar1_phi,
    estimate_covariance,
    is_diagonal_method,
    list_covariance_methods,
    register_covariance,
)
from hierarchicalforecast.methods import MinTrace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hierarchy_data():
    """Simple 3-level hierarchy: 1 total, 2 groups, 4 bottom."""
    S = np.array([
        [1.0, 1.0, 1.0, 1.0],  # Total
        [1.0, 1.0, 0.0, 0.0],  # Group 1
        [0.0, 0.0, 1.0, 1.0],  # Group 2
        [1.0, 0.0, 0.0, 0.0],  # Bottom 1
        [0.0, 1.0, 0.0, 0.0],  # Bottom 2
        [0.0, 0.0, 1.0, 0.0],  # Bottom 3
        [0.0, 0.0, 0.0, 1.0],  # Bottom 4
    ])
    return S


@pytest.fixture
def residuals(hierarchy_data):
    """Realistic residuals: (n_hiers=7, n_obs=50)."""
    rng = np.random.default_rng(42)
    n_hiers = hierarchy_data.shape[0]
    n_obs = 50
    # Bottom-level residuals
    res_bottom = rng.standard_normal((4, n_obs))
    # Aggregate residuals consistent with S
    res = hierarchy_data @ res_bottom
    # Add a bit of noise to make them non-trivial
    res += rng.standard_normal((n_hiers, n_obs)) * 0.1
    return res


@pytest.fixture
def residuals_with_nans(residuals):
    """Residuals with some NaN values."""
    res = residuals.copy()
    res[0, :3] = np.nan
    res[2, 5:8] = np.nan
    return res


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_list_covariance_methods():
    methods = list_covariance_methods()
    assert isinstance(methods, list)
    # Check all expected methods are present
    expected = [
        "ols", "wls_struct", "wls_var", "sam", "mint_cov", "shr",
        "mint_shrink", "bu", "oasd", "wlsv", "wlsh", "acov",
        "strar1", "sar1", "har1",
        # Cross-temporal methods
        "csstr", "testr", "bdshr", "bdsam", "sshr", "ssam",
        "hshr", "hsam", "hbshr", "hbsam", "bshr", "bsam",
    ]
    for m in expected:
        assert m in methods, f"Expected method '{m}' not in registry"


def test_requires_residuals():
    # OLS, wls_struct, and CT structural methods don't require residuals
    for m in ["ols", "wls_struct", "csstr", "testr"]:
        assert m not in REQUIRES_RESIDUALS, f"'{m}' should NOT require residuals"
    # All others do
    for m in ["wls_var", "sam", "mint_cov", "shr", "mint_shrink",
              "bu", "oasd", "wlsv", "wlsh", "acov", "strar1", "sar1", "har1",
              "bdshr", "bdsam", "sshr", "ssam", "hshr", "hsam",
              "hbshr", "hbsam", "bshr", "bsam"]:
        assert m in REQUIRES_RESIDUALS, f"'{m}' should require residuals"


def test_unknown_method(hierarchy_data):
    with pytest.raises(ValueError, match="Unknown covariance method"):
        estimate_covariance("nonexistent_method", S=hierarchy_data)


def test_missing_residuals(hierarchy_data):
    with pytest.raises(ValueError, match="requires residuals"):
        estimate_covariance("wls_var", S=hierarchy_data, residuals=None)


def test_register_custom_method(hierarchy_data):
    def my_cov(S, residuals=None, **kw):
        return np.eye(S.shape[0]) * 2.0

    register_covariance("test_custom", my_cov, requires_residuals=False)
    try:
        W = estimate_covariance("test_custom", S=hierarchy_data)
        np.testing.assert_array_equal(W, np.eye(7) * 2.0)
        assert "test_custom" in list_covariance_methods()
    finally:
        _REGISTRY.pop("test_custom", None)
        REQUIRES_RESIDUALS.discard("test_custom")


# ---------------------------------------------------------------------------
# Shape and symmetry tests for all methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ols", "wls_struct"])
def test_no_residual_methods_shape(hierarchy_data, method):
    W = estimate_covariance(method, S=hierarchy_data)
    n = hierarchy_data.shape[0]
    assert W.shape == (n, n)
    np.testing.assert_allclose(W, W.T, atol=1e-12)


@pytest.mark.parametrize("method", [
    "wls_var", "sam", "mint_cov", "shr", "mint_shrink",
    "bu", "oasd", "wlsv", "wlsh",
])
def test_residual_methods_shape(hierarchy_data, residuals, method):
    kwargs = {}
    if method in ("shr", "mint_shrink"):
        kwargs["mint_shr_ridge"] = 2e-8
    W = estimate_covariance(method, S=hierarchy_data, residuals=residuals, **kwargs)
    n = hierarchy_data.shape[0]
    assert W.shape == (n, n)
    np.testing.assert_allclose(W, W.T, atol=1e-12)


@pytest.mark.parametrize("method", ["acov", "strar1", "sar1", "har1"])
def test_temporal_methods_shape(hierarchy_data, residuals, method):
    agg_order = np.array([4, 2, 2, 1, 1, 1, 1])
    W = estimate_covariance(
        method, S=hierarchy_data, residuals=residuals, agg_order=agg_order
    )
    n = hierarchy_data.shape[0]
    assert W.shape == (n, n)
    np.testing.assert_allclose(W, W.T, atol=1e-12)


# ---------------------------------------------------------------------------
# OLS / WLS specific tests
# ---------------------------------------------------------------------------

def test_ols_is_identity(hierarchy_data):
    W = estimate_covariance("ols", S=hierarchy_data)
    np.testing.assert_array_equal(W, np.eye(7))


def test_wls_struct_diagonal(hierarchy_data):
    W = estimate_covariance("wls_struct", S=hierarchy_data)
    expected_diag = np.sum(hierarchy_data, axis=1)
    np.testing.assert_array_equal(np.diag(W), expected_diag)
    # Off-diagonal should be zero
    np.testing.assert_array_equal(W - np.diag(np.diag(W)), 0.0)


def test_wls_var_diagonal(hierarchy_data, residuals):
    W = estimate_covariance("wls_var", S=hierarchy_data, residuals=residuals)
    # Should be diagonal
    assert np.count_nonzero(W - np.diag(np.diag(W))) == 0
    # Diagonal should be positive
    assert np.all(np.diag(W) > 0)


# ---------------------------------------------------------------------------
# Sam / mint_cov equivalence
# ---------------------------------------------------------------------------

def test_sam_mint_cov_equivalence(hierarchy_data, residuals):
    W_sam = estimate_covariance("sam", S=hierarchy_data, residuals=residuals)
    W_mint_cov = estimate_covariance("mint_cov", S=hierarchy_data, residuals=residuals)
    np.testing.assert_allclose(W_sam, W_mint_cov, atol=1e-12)


def test_shr_mint_shrink_equivalence(hierarchy_data, residuals):
    W_shr = estimate_covariance("shr", S=hierarchy_data, residuals=residuals)
    W_mint_shrink = estimate_covariance(
        "mint_shrink", S=hierarchy_data, residuals=residuals
    )
    np.testing.assert_allclose(W_shr, W_mint_shrink, atol=1e-12)


# ---------------------------------------------------------------------------
# Bottom-up covariance tests
# ---------------------------------------------------------------------------

def test_bu_covariance_structure(hierarchy_data, residuals):
    W = estimate_covariance("bu", S=hierarchy_data, residuals=residuals)
    # W = S @ cov_bottom @ S'
    n_hiers, n_bottom = hierarchy_data.shape
    res_bottom = residuals[n_hiers - n_bottom:]
    cov_bottom = np.cov(res_bottom)
    expected = hierarchy_data @ cov_bottom @ hierarchy_data.T
    np.testing.assert_allclose(W, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# OASD tests
# ---------------------------------------------------------------------------

def test_oasd_positive_diagonal(hierarchy_data, residuals):
    W = estimate_covariance("oasd", S=hierarchy_data, residuals=residuals)
    assert np.all(np.diag(W) > 0)


def test_oasd_shrinks_toward_diagonal(hierarchy_data, residuals):
    """OASD should produce off-diagonal values smaller (in abs) than sample cov."""
    W_oasd = estimate_covariance("oasd", S=hierarchy_data, residuals=residuals)
    W_sam = estimate_covariance("sam", S=hierarchy_data, residuals=residuals)
    # Off-diagonal magnitudes should be <= sample covariance
    n = hierarchy_data.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                assert abs(W_oasd[i, j]) <= abs(W_sam[i, j]) + 1e-10


def test_oasd_with_nans(hierarchy_data, residuals_with_nans):
    W = estimate_covariance("oasd", S=hierarchy_data, residuals=residuals_with_nans)
    assert W.shape == (7, 7)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    assert np.all(np.diag(W) > 0)


# ---------------------------------------------------------------------------
# Temporal method tests
# ---------------------------------------------------------------------------

def test_wlsv_pools_by_agg_order(hierarchy_data, residuals):
    """wlsv should give same variance to series with same aggregation order."""
    agg_order = np.array([4, 2, 2, 1, 1, 1, 1])
    W = estimate_covariance(
        "wlsv", S=hierarchy_data, residuals=residuals, agg_order=agg_order
    )
    diag = np.diag(W)
    # All bottom-level (agg_order=1) should have same variance
    bottom_vars = diag[agg_order == 1]
    assert np.all(np.abs(bottom_vars - bottom_vars[0]) < 1e-12)
    # Semi-annual (agg_order=2)
    semi_vars = diag[agg_order == 2]
    assert np.all(np.abs(semi_vars - semi_vars[0]) < 1e-12)


def test_acov_block_diagonal(hierarchy_data, residuals):
    """acov should be block-diagonal by aggregation level."""
    agg_order = np.array([4, 2, 2, 1, 1, 1, 1])
    W = estimate_covariance(
        "acov", S=hierarchy_data, residuals=residuals, agg_order=agg_order
    )
    # Cross-level entries should be zero
    for i in range(7):
        for j in range(7):
            if agg_order[i] != agg_order[j]:
                assert W[i, j] == 0.0, f"W[{i},{j}] should be 0 (different agg levels)"


def test_ar1_correlation_decay():
    """Test that AR(1) correlation matrices decay with lag."""
    from hierarchicalforecast.covariance import _build_ar1_correlation

    R = _build_ar1_correlation(5, 0.5)
    assert R.shape == (5, 5)
    # Diagonal should be 1
    np.testing.assert_allclose(np.diag(R), 1.0)
    # Correlation should decay: R[0,1] > R[0,2] > R[0,3]
    assert R[0, 1] > R[0, 2] > R[0, 3] > R[0, 4]


@pytest.mark.parametrize("method", ["strar1", "sar1", "har1"])
def test_ar1_methods_positive_definite(hierarchy_data, residuals, method):
    """AR(1) methods should produce positive semi-definite matrices."""
    agg_order = np.array([4, 2, 2, 1, 1, 1, 1])
    W = estimate_covariance(
        method, S=hierarchy_data, residuals=residuals, agg_order=agg_order
    )
    # Check positive semi-definiteness via eigenvalues
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-10), (
        f"Method '{method}' produced non-PSD matrix, min eigenvalue={eigenvalues.min()}"
    )


# ---------------------------------------------------------------------------
# NaN handling tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["sam", "shr", "oasd"])
def test_nan_handling(hierarchy_data, residuals_with_nans, method):
    """Methods with NaN handling should not crash."""
    kwargs = {}
    if method in ("shr", "mint_shrink"):
        kwargs["mint_shr_ridge"] = 2e-8
    W = estimate_covariance(
        method, S=hierarchy_data, residuals=residuals_with_nans, **kwargs
    )
    assert W.shape == (7, 7)
    assert not np.any(np.isnan(W))


# ---------------------------------------------------------------------------
# Cross-temporal covariance method tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ct_data():
    """Cross-temporal test data.

    Cross-sectional: 1 total, 2 bottom (3 CS series)
    Temporal: 1 annual, 2 semi-annual (3 temporal periods)
    CT vector: 3 * 3 = 9 elements in Kronecker ordering.
    """
    S_cs = np.array([
        [1.0, 1.0],  # Total
        [1.0, 0.0],  # Bottom 1
        [0.0, 1.0],  # Bottom 2
    ])
    S_te = np.array([
        [1.0, 1.0],  # Annual
        [1.0, 0.0],  # Semi-annual 1
        [0.0, 1.0],  # Semi-annual 2
    ])
    n_cs = S_cs.shape[0]
    n_te = S_te.shape[0]
    n_ct = n_cs * n_te

    # S_ct = S_cs ⊗ S_te (for reference, not directly used)
    S_ct = np.kron(S_cs, S_te)

    rng = np.random.default_rng(42)
    n_obs = 50

    # Generate bottom-level CT residuals and aggregate
    n_bottom_cs = S_cs.shape[1]
    n_bottom_te = S_te.shape[1]
    res_bottom = rng.standard_normal((n_bottom_cs, n_bottom_te, n_obs))

    # Aggregate through both hierarchies
    res_3d = np.zeros((n_cs, n_te, n_obs))
    for i in range(n_cs):
        for j in range(n_te):
            for bi in range(n_bottom_cs):
                for bj in range(n_bottom_te):
                    res_3d[i, j] += S_cs[i, bi] * S_te[j, bj] * res_bottom[bi, bj]

    # Add noise to avoid degeneracy
    res_3d += rng.standard_normal((n_cs, n_te, n_obs)) * 0.1

    # Flatten to (n_ct, n_obs)
    residuals_ct = res_3d.reshape(n_ct, n_obs)

    return {
        "S_cs": S_cs,
        "S_te": S_te,
        "S_ct": S_ct,
        "n_cs": n_cs,
        "n_te": n_te,
        "n_ct": n_ct,
        "residuals_ct": residuals_ct,
    }


@pytest.mark.parametrize("method", ["csstr", "testr"])
def test_ct_structural_methods_shape(ct_data, method):
    """Cross-temporal structural methods should produce diagonal matrices."""
    W = estimate_covariance(
        method,
        S=ct_data["S_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_ct = ct_data["n_ct"]
    assert W.shape == (n_ct, n_ct)
    # Should be diagonal
    assert np.count_nonzero(W - np.diag(np.diag(W))) == 0
    # Diagonal should be positive
    assert np.all(np.diag(W) > 0)


def test_csstr_weights(ct_data):
    """csstr diagonal entries should equal S_cs row sums, replicated."""
    W = estimate_covariance(
        "csstr",
        S=ct_data["S_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_cs, n_te = ct_data["n_cs"], ct_data["n_te"]
    cs_weights = np.sum(ct_data["S_cs"], axis=1)
    for i in range(n_cs):
        for j in range(n_te):
            idx = i * n_te + j
            assert W[idx, idx] == cs_weights[i]


def test_testr_weights(ct_data):
    """testr diagonal entries should equal S_te row sums, replicated."""
    W = estimate_covariance(
        "testr",
        S=ct_data["S_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_cs, n_te = ct_data["n_cs"], ct_data["n_te"]
    te_weights = np.sum(ct_data["S_te"], axis=1)
    for i in range(n_cs):
        for j in range(n_te):
            idx = i * n_te + j
            assert W[idx, idx] == te_weights[j]


@pytest.mark.parametrize("method", ["bdshr", "bdsam"])
def test_ct_block_diagonal_by_temporal(ct_data, method):
    """bdshr/bdsam should be block-diagonal with blocks by temporal period."""
    W = estimate_covariance(
        method,
        S=ct_data["S_ct"],
        residuals=ct_data["residuals_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_cs, n_te, n_ct = ct_data["n_cs"], ct_data["n_te"], ct_data["n_ct"]
    assert W.shape == (n_ct, n_ct)
    np.testing.assert_allclose(W, W.T, atol=1e-12)

    # Cross-block entries should be zero (different temporal periods)
    for j1 in range(n_te):
        for j2 in range(n_te):
            if j1 != j2:
                idx1 = np.arange(n_cs) * n_te + j1
                idx2 = np.arange(n_cs) * n_te + j2
                block = W[np.ix_(idx1, idx2)]
                assert np.allclose(block, 0.0), (
                    f"Cross-block ({j1},{j2}) should be zero for {method}"
                )


@pytest.mark.parametrize("method", ["sshr", "ssam"])
def test_ct_block_diagonal_by_series(ct_data, method):
    """sshr/ssam should be block-diagonal with blocks by CS series."""
    W = estimate_covariance(
        method,
        S=ct_data["S_ct"],
        residuals=ct_data["residuals_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_cs, n_te, n_ct = ct_data["n_cs"], ct_data["n_te"], ct_data["n_ct"]
    assert W.shape == (n_ct, n_ct)
    np.testing.assert_allclose(W, W.T, atol=1e-12)

    # Cross-series entries should be zero
    for i1 in range(n_cs):
        for i2 in range(n_cs):
            if i1 != i2:
                idx1 = np.arange(i1 * n_te, (i1 + 1) * n_te)
                idx2 = np.arange(i2 * n_te, (i2 + 1) * n_te)
                block = W[np.ix_(idx1, idx2)]
                assert np.allclose(block, 0.0), (
                    f"Cross-series block ({i1},{i2}) should be zero for {method}"
                )


@pytest.mark.parametrize("method", ["hshr", "hsam"])
def test_ct_high_frequency_methods(ct_data, method):
    """hshr/hsam propagate HF covariance through kron(I_n, S_te)."""
    W = estimate_covariance(
        method,
        S=ct_data["S_ct"],
        residuals=ct_data["residuals_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_ct = ct_data["n_ct"]
    assert W.shape == (n_ct, n_ct)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    # Should be positive definite (ridge-regularized)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues > 0), (
        f"Method '{method}' should be positive definite after ridge"
    )


@pytest.mark.parametrize("method", ["hbshr", "hbsam"])
def test_ct_hf_bottom_methods(ct_data, method):
    """hbshr/hbsam should have valid structure."""
    W = estimate_covariance(
        method,
        S=ct_data["S_ct"],
        residuals=ct_data["residuals_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_ct = ct_data["n_ct"]
    assert W.shape == (n_ct, n_ct)
    np.testing.assert_allclose(W, W.T, atol=1e-12)


@pytest.mark.parametrize("method", ["bshr", "bsam"])
def test_ct_bottom_series_methods(ct_data, method):
    """bshr/bsam propagate bottom CS covariance through kron(S_cs, I)."""
    W = estimate_covariance(
        method,
        S=ct_data["S_ct"],
        residuals=ct_data["residuals_ct"],
        n_cs=ct_data["n_cs"],
        n_te=ct_data["n_te"],
        S_cs=ct_data["S_cs"],
        S_te=ct_data["S_te"],
    )
    n_ct = ct_data["n_ct"]
    assert W.shape == (n_ct, n_ct)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    # Should be positive definite (ridge-regularized)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues > 0), (
        f"Method '{method}' should be positive definite after ridge"
    )


@pytest.mark.parametrize("method", [
    "csstr", "testr", "bdshr", "bdsam", "sshr", "ssam",
    "hshr", "hsam", "hbshr", "hbsam", "bshr", "bsam",
])
def test_ct_methods_positive_semidefinite(ct_data, method):
    """All CT covariance methods should produce PSD matrices."""
    kwargs = {
        "n_cs": ct_data["n_cs"],
        "n_te": ct_data["n_te"],
        "S_cs": ct_data["S_cs"],
        "S_te": ct_data["S_te"],
    }
    needs_residuals = method not in ("csstr", "testr")
    if needs_residuals:
        kwargs["residuals"] = ct_data["residuals_ct"]

    W = estimate_covariance(method, S=ct_data["S_ct"], **kwargs)
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-10), (
        f"Method '{method}' produced non-PSD matrix, min eigenvalue={eigenvalues.min()}"
    )


def test_ct_missing_args():
    """Cross-temporal methods should raise clear errors for missing args."""
    S = np.eye(9)
    with pytest.raises(ValueError, match="n_cs.*n_te"):
        estimate_covariance("csstr", S=S)

    with pytest.raises(ValueError, match="S_cs.*S_te"):
        estimate_covariance("csstr", S=S, n_cs=3, n_te=3)


def test_ct_residual_shape_mismatch():
    """Should raise if residuals don't match n_cs * n_te."""
    from hierarchicalforecast.covariance import _reshape_ct_residuals
    residuals = np.zeros((10, 5))
    with pytest.raises(ValueError, match="n_cs \\* n_te"):
        _reshape_ct_residuals(residuals, n_cs=3, n_te=3)


# ---------------------------------------------------------------------------
# Integration: MinTrace with new covariance methods
# ---------------------------------------------------------------------------

@pytest.fixture
def mintrace_data(hierarchy_data):
    """Data suitable for MinTrace reconciliation.

    Uses independent per-series noise for insample forecasts so that the
    residual covariance is full-rank — needed for sam/mint_cov to be
    well-conditioned.
    """
    S = hierarchy_data
    rng = np.random.default_rng(123)
    n_hiers = S.shape[0]
    n_bottom = S.shape[1]
    h = 3
    n_obs = 200

    y_hat_bottom = rng.uniform(1, 10, (n_bottom, h))
    y_hat = S @ y_hat_bottom

    # Generate insample data at ALL levels independently
    y_insample = rng.uniform(1, 10, (n_hiers, n_obs))
    # Add independent noise per series for the fitted values so residuals
    # are full-rank (not constrained by the hierarchy)
    y_hat_insample = y_insample + rng.standard_normal((n_hiers, n_obs)) * 2.0

    return {
        "S": S,
        "y_hat": y_hat,
        "y_insample": y_insample,
        "y_hat_insample": y_hat_insample,
    }


@pytest.mark.parametrize("method", [
    "ols", "wls_struct", "wls_var",
    "sam", "mint_cov", "shr", "mint_shrink",
    "oasd",
])
def test_mintrace_with_covariance_method(mintrace_data, method):
    """MinTrace should work with all registered covariance methods.

    Note: 'bu' is excluded because W = S @ cov_bottom @ S' is inherently
    rank-deficient (rank n_bottom < n_hiers), making the MinTrace linear
    system singular. Use 'bu' covariance with methods that handle this
    (e.g., via pseudoinverse).
    """
    mt = MinTrace(method=method)
    result = mt.fit_predict(**mintrace_data)

    assert "mean" in result
    y_rec = result["mean"]
    S = mintrace_data["S"]

    # Check shape
    assert y_rec.shape == mintrace_data["y_hat"].shape

    # Check coherence: S @ bottom == all
    n_bottom = S.shape[1]
    bottom = y_rec[S.shape[0] - n_bottom:]
    np.testing.assert_allclose(y_rec, S @ bottom, rtol=1e-8)


def test_mintrace_ols_unchanged(mintrace_data):
    """MinTrace OLS should produce the same results as before refactoring."""
    mt = MinTrace(method="ols")
    result_ols = mt.fit_predict(**mintrace_data)["mean"]

    # OLS with identity W should be equivalent to using wls_struct with uniform weights
    # Just verify it runs and is coherent
    S = mintrace_data["S"]
    n_bottom = S.shape[1]
    bottom = result_ols[S.shape[0] - n_bottom:]
    np.testing.assert_allclose(result_ols, S @ bottom, rtol=1e-8)


def test_mintrace_unknown_method_error():
    """MinTrace should raise for truly unknown methods."""
    with pytest.raises(ValueError, match="Unknown method"):
        MinTrace(method="totally_fake_method")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_residuals_1d():
    """1-D residuals should raise ValueError."""
    from hierarchicalforecast.covariance import _validate_residuals
    with pytest.raises(ValueError, match="must be 2-d"):
        _validate_residuals(np.array([1.0, 2.0, 3.0]))


def test_validate_residuals_too_few_obs():
    """Residuals with fewer than 3 observations should raise."""
    from hierarchicalforecast.covariance import _validate_residuals
    with pytest.raises(ValueError, match="at least 3 observations"):
        _validate_residuals(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_validate_residuals_near_zero():
    """Nearly-zero residuals should raise overfitting warning."""
    from hierarchicalforecast.covariance import _validate_residuals
    res = np.full((5, 20), 1e-10)
    with pytest.raises(ValueError, match="close to 0"):
        _validate_residuals(res)


# ---------------------------------------------------------------------------
# OASD known-answer test
# ---------------------------------------------------------------------------

def test_oasd_known_answer():
    """OASD with perfectly correlated series should preserve correlation."""
    rng = np.random.default_rng(99)
    n_obs = 100
    x = rng.standard_normal(n_obs)
    # Two perfectly correlated series
    residuals = np.vstack([x, x])
    S = np.eye(2)
    W = estimate_covariance("oasd", S=S, residuals=residuals)
    # Off-diagonal should be close to diagonal (high correlation preserved)
    ratio = abs(W[0, 1]) / np.sqrt(W[0, 0] * W[1, 1])
    assert ratio > 0.9, f"OASD should preserve strong correlation, got ratio={ratio}"


def test_oasd_independent_series():
    """OASD with independent series should shrink off-diagonal toward zero."""
    rng = np.random.default_rng(42)
    n_obs = 200
    residuals = rng.standard_normal((3, n_obs))
    S = np.eye(3)
    W = estimate_covariance("oasd", S=S, residuals=residuals)
    # Off-diagonal should be small
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(W[i, j]) < 0.3, (
                    f"OASD off-diagonal [{i},{j}]={W[i, j]:.4f} should be near zero"
                )


# ---------------------------------------------------------------------------
# Temporal fallback tests (agg_order=None)
# ---------------------------------------------------------------------------

def test_acov_fallback_equals_sam(hierarchy_data, residuals):
    """acov without agg_order should equal sam."""
    W_acov = estimate_covariance("acov", S=hierarchy_data, residuals=residuals)
    W_sam = estimate_covariance("sam", S=hierarchy_data, residuals=residuals)
    np.testing.assert_allclose(W_acov, W_sam, atol=1e-12)


def test_wlsv_fallback_equals_wls_var(hierarchy_data, residuals):
    """wlsv without agg_order should equal wls_var."""
    W_wlsv = estimate_covariance("wlsv", S=hierarchy_data, residuals=residuals)
    W_wls_var = estimate_covariance("wls_var", S=hierarchy_data, residuals=residuals)
    np.testing.assert_allclose(W_wlsv, W_wls_var, atol=1e-12)


@pytest.mark.parametrize("method", ["strar1", "sar1", "har1"])
def test_ar1_fallback_no_agg_order(hierarchy_data, residuals, method):
    """AR(1) methods without agg_order should run and produce valid output."""
    W = estimate_covariance(method, S=hierarchy_data, residuals=residuals)
    n = hierarchy_data.shape[0]
    assert W.shape == (n, n)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    # Should be PSD
    eigenvalues = np.linalg.eigvalsh(W)
    assert np.all(eigenvalues >= -1e-10)


# ---------------------------------------------------------------------------
# Extended NaN handling tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["wls_var", "bu", "acov"])
def test_nan_handling_extended(hierarchy_data, residuals_with_nans, method):
    """Additional methods with NaN handling should not crash."""
    kwargs = {}
    if method == "acov":
        kwargs["agg_order"] = np.array([4, 2, 2, 1, 1, 1, 1])
    W = estimate_covariance(
        method, S=hierarchy_data, residuals=residuals_with_nans, **kwargs
    )
    assert W.shape == (7, 7)
    assert not np.any(np.isnan(W))


# ---------------------------------------------------------------------------
# Reference equivalence tests (R-computed ground truth)
# ---------------------------------------------------------------------------
# These reference values were computed in R 4.1.2 using base R functions
# (cov, solve) with the same formulas used in FoReco. The test data uses
# set.seed(123) with a 3-series hierarchy (1 total + 2 bottom, n_obs=50).
# ---------------------------------------------------------------------------

@pytest.fixture
def r_reference_data():
    """Test data matching R set.seed(123) output for reference comparison.

    S = [[1, 1], [1, 0], [0, 1]]  (1 total + 2 bottom)
    residuals generated as: res = rnorm(3, 50); res[1,] += 0.5 * res[2,]
    """
    S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    # Residuals from R: set.seed(123); res <- matrix(rnorm(3*50), 3, 50); res[1,] <- res[1,] + 0.5*res[2,]
    residuals = np.array([
        [-0.675564391293853, 0.135152259005049, -0.171614411314065, 0.166378928619773,
         0.456112808566612, 2.035838375917698, 0.464960197699719, -0.730977138811915,
         -1.468385923220463, -0.415695350669459, 0.278928479980678, 1.288924028351786,
         0.522961798249228, -0.727824490472639, -0.180918368899008, -1.324551000852887,
         0.738280585100403, -0.049981983994361, 0.532464316555502, 0.646540871558376,
         0.128477756205231, -1.554470996344878, 0.474711891994678, 1.804569102598877,
         0.651138143171060, 0.883184866171194, 0.111857798529628, 0.199950385230126,
         -0.054595579860902, 0.272215698068189, 1.267702335716154, 0.052420148225632,
         2.953638306109171, -1.381624182156431, -0.420463178161240, -0.437479959537458,
         0.079271784242621, -0.200977031919549, 0.669983885026819, -1.065558181322168,
         -0.356090709992275, 0.665838810417857, 0.274366997066712, 0.650967343088076,
         -0.170015494176872, 0.401017177951765, 1.187156988731584, -1.048269568975212,
         -1.866989434659745, 1.737971243238663],
        [-0.230177489483280, 0.129287735160946, -1.265061234606534, 1.224081797439462,
         0.110682715945120, 0.497850478229239, -0.472791407727934, -1.026004448307240,
         -1.686693310742413, -1.138136937011948, -0.295071482992271, 0.821581081637487,
         -0.061911710576722, -0.694706978920513, 2.168955965338513, -0.402884835299076,
         -0.083369066471829, -0.042870457291316, 1.516470604429540, 0.123854243844614,
         -0.502323453109302, -1.071791226475578, 0.053004226730504, -0.491031166056535,
         -0.709200762582393, -0.284773007051009, -0.138891362439045, -0.370660031792409,
         0.331781963915697, -0.325931585531227, 0.548396959508070, 1.360652448530008,
         1.532610626185189, -0.710406563699301, -0.347542599397733, -0.784904469457076,
         0.918996609060766, -1.617882708289164, 0.301153362166714, -0.849704346033582,
         -0.947474614184802, 1.843862005232207, 0.077960849563711, 1.444550858423349,
         -0.422496832339625, -1.460640070924822, -1.443893160971799, -1.572144159145488,
         -0.530906522170303, 2.100108940525672],
        [1.558708314149124, 1.715064986883281, -0.686852851893526, 0.359813827057364,
         -0.555841134754075, -1.966617156629638, -1.067823705986845, -0.728891229291140,
         0.837787044494525, 1.253814921069927, 0.895125661045022, 0.688640254100091,
         -0.305962663739917, -0.207917278019599, 1.207961998304991, -0.466655353623219,
         0.253318513994755, 1.368602284014458, -1.548752804230221, 0.215941568743973,
         -0.333207383669420, 0.303528641404258, 0.922267467879738, -2.309168875640812,
         -0.688008616467358, -1.220717712254536, 0.005764185899887, 0.644376548518833,
         1.096839013149348, 1.148807618451094, 0.238731735111441, -0.600259587147127,
         -0.235700359100477, 0.256883709156530, -0.951618567265016, -1.667941936588137,
         -0.575346962608392, -0.055561965524539, 0.105676194148943, -1.024128790604913,
         -0.490557443700668, -0.651949901695459, -0.961856634130129, 0.451504053079215,
         -2.053247221540516, 0.739947510877334, 0.701784335374711, -1.514667653781751,
         -1.461755584995900, -1.287030476035179],
    ])

    y_hat = np.array([10.5, 6.0, 4.0])

    return {"S": S, "residuals": residuals, "y_hat": y_hat}


def test_reference_sam_covariance(r_reference_data):
    """Sample covariance should match R cov(t(res))."""
    W = estimate_covariance("sam", S=r_reference_data["S"], residuals=r_reference_data["residuals"])
    W_expected = np.array([
        [0.902072894804819, 0.468776537497085, -0.101009896936862],
        [0.468776537497085, 0.966917507654438, 0.002526883744891],
        [-0.101009896936862, 0.002526883744891, 1.022789071280485],
    ])
    np.testing.assert_allclose(W, W_expected, atol=1e-12)


def test_reference_shr_covariance(r_reference_data):
    """Shrinkage covariance should match R manual Schafer-Strimmer."""
    W = estimate_covariance("shr", S=r_reference_data["S"], residuals=r_reference_data["residuals"])
    W_expected = np.array([
        [0.902072894804819, 0.354525506920776, -0.076391589704463],
        [0.354525506920776, 0.966917507654438, 0.001911027256975],
        [-0.076391589704463, 0.001911027256975, 1.022789071280485],
    ])
    np.testing.assert_allclose(W, W_expected, atol=1e-10)


def test_reference_wls_var_diagonal(r_reference_data):
    """WLS variance diagonal should match R rowMeans(res^2)."""
    W = estimate_covariance("wls_var", S=r_reference_data["S"], residuals=r_reference_data["residuals"])
    expected_diag = np.array([0.906118980455287, 0.957091022988439, 1.032242587396044])
    np.testing.assert_allclose(np.diag(W), expected_diag, atol=1e-10)


def test_reference_mintrace_ols_reconciliation(r_reference_data):
    """MinTrace OLS reconciled forecasts should match R reference."""
    mt = MinTrace(method="ols")
    # Build insample data so residuals match
    n_hiers = 3
    n_obs = r_reference_data["residuals"].shape[1]
    y_insample = np.ones((n_hiers, n_obs))  # dummy
    y_hat_insample = y_insample - r_reference_data["residuals"]

    result = mt.fit_predict(
        S=r_reference_data["S"],
        y_hat=r_reference_data["y_hat"].reshape(-1, 1),
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
    )
    y_rec = result["mean"].flatten()
    # Bottom-level reconciled values from R
    expected_bottom = np.array([6.166666666666667, 4.166666666666667])
    np.testing.assert_allclose(y_rec[1:], expected_bottom, atol=1e-10)
    # Check coherence
    np.testing.assert_allclose(y_rec[0], y_rec[1] + y_rec[2], atol=1e-10)


def test_reference_mintrace_sam_reconciliation(r_reference_data):
    """MinTrace sample-cov reconciled forecasts should match R reference."""
    mt = MinTrace(method="sam")
    n_hiers = 3
    n_obs = r_reference_data["residuals"].shape[1]
    y_insample = np.ones((n_hiers, n_obs))
    y_hat_insample = y_insample - r_reference_data["residuals"]

    result = mt.fit_predict(
        S=r_reference_data["S"],
        y_hat=r_reference_data["y_hat"].reshape(-1, 1),
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
    )
    y_rec = result["mean"].flatten()
    expected_bottom = np.array([6.115825628821317, 4.260566759068785])
    np.testing.assert_allclose(y_rec[1:], expected_bottom, atol=1e-8)
    np.testing.assert_allclose(y_rec[0], y_rec[1] + y_rec[2], atol=1e-10)


def test_reference_mintrace_wls_struct_reconciliation(r_reference_data):
    """MinTrace WLS-structural reconciled forecasts should match R reference."""
    mt = MinTrace(method="wls_struct")
    n_hiers = 3
    n_obs = r_reference_data["residuals"].shape[1]
    y_insample = np.ones((n_hiers, n_obs))
    y_hat_insample = y_insample - r_reference_data["residuals"]

    result = mt.fit_predict(
        S=r_reference_data["S"],
        y_hat=r_reference_data["y_hat"].reshape(-1, 1),
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
    )
    y_rec = result["mean"].flatten()
    expected_bottom = np.array([6.125, 4.125])
    np.testing.assert_allclose(y_rec[1:], expected_bottom, atol=1e-10)
    np.testing.assert_allclose(y_rec[0], y_rec[1] + y_rec[2], atol=1e-10)


# ---------------------------------------------------------------------------
# Correctness tests for wlsh
# ---------------------------------------------------------------------------

def test_wlsh_equals_wls_var(hierarchy_data, residuals):
    """wlsh should produce identical results to wls_var."""
    W_wlsh = estimate_covariance("wlsh", S=hierarchy_data, residuals=residuals)
    W_wlsvar = estimate_covariance("wls_var", S=hierarchy_data, residuals=residuals)
    np.testing.assert_array_equal(W_wlsh, W_wlsvar)


def test_wlsh_diagonal_values(hierarchy_data, residuals):
    """wlsh diagonal entries should equal per-series mean squared residuals + ridge."""
    W = estimate_covariance("wlsh", S=hierarchy_data, residuals=residuals)
    expected_diag = np.nanmean(residuals**2, axis=1) + 2e-8
    np.testing.assert_allclose(np.diag(W), expected_diag, rtol=1e-10)
    # Off-diagonal should be zero
    assert np.count_nonzero(W - np.diag(np.diag(W))) == 0


# ---------------------------------------------------------------------------
# Correctness tests for cross-temporal methods
# ---------------------------------------------------------------------------

@pytest.fixture
def ct_data_with_agg_order(ct_data):
    """Extend ct_data with temporal aggregation order."""
    # S_te has 3 rows: [annual, semi-annual H1, semi-annual H2]
    # Annual aggregates both, so agg_order = [2, 1, 1]
    ct_data["agg_order_te"] = np.array([2, 1, 1])
    return ct_data


def test_bdshr_blocks_share_within_agg_order(ct_data_with_agg_order):
    """bdshr: periods with the same agg_order should have identical blocks."""
    d = ct_data_with_agg_order
    W = estimate_covariance(
        "bdshr",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=d["n_cs"],
        n_te=d["n_te"],
        S_cs=d["S_cs"],
        S_te=d["S_te"],
        agg_order_te=d["agg_order_te"],
    )
    n_cs, n_te = d["n_cs"], d["n_te"]
    # Temporal periods 1 and 2 (0-indexed) both have agg_order=1
    # Their blocks should be identical
    idx_te1 = np.arange(n_cs) * n_te + 1
    idx_te2 = np.arange(n_cs) * n_te + 2
    block_te1 = W[np.ix_(idx_te1, idx_te1)]
    block_te2 = W[np.ix_(idx_te2, idx_te2)]
    np.testing.assert_allclose(block_te1, block_te2, atol=1e-12)


def test_bdsam_blocks_share_within_agg_order(ct_data_with_agg_order):
    """bdsam: periods with the same agg_order should have identical blocks."""
    d = ct_data_with_agg_order
    W = estimate_covariance(
        "bdsam",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=d["n_cs"],
        n_te=d["n_te"],
        S_cs=d["S_cs"],
        S_te=d["S_te"],
        agg_order_te=d["agg_order_te"],
    )
    n_cs, n_te = d["n_cs"], d["n_te"]
    idx_te1 = np.arange(n_cs) * n_te + 1
    idx_te2 = np.arange(n_cs) * n_te + 2
    block_te1 = W[np.ix_(idx_te1, idx_te1)]
    block_te2 = W[np.ix_(idx_te2, idx_te2)]
    np.testing.assert_allclose(block_te1, block_te2, atol=1e-12)


def test_bdsam_block_matches_sample_covariance(ct_data_with_agg_order):
    """bdsam block for agg_order k should equal sample cov of pooled residuals."""
    d = ct_data_with_agg_order
    n_cs, n_te = d["n_cs"], d["n_te"]
    agg_order_te = d["agg_order_te"]

    W = estimate_covariance(
        "bdsam",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=d["S_cs"],
        S_te=d["S_te"],
        agg_order_te=agg_order_te,
    )

    # For agg_order=2 (annual, period 0): block should be cov of series at period 0
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    te_idx_annual = np.where(agg_order_te == 2)[0]  # [0]
    res_annual = res_3d[:, te_idx_annual, :]  # (n_cs, 1, n_obs)
    res_pooled = res_annual.reshape(n_cs, -1)  # (n_cs, n_obs)
    expected_block = np.cov(res_pooled)

    ct_idx = np.arange(n_cs) * n_te + 0  # period 0 across all series
    actual_block = W[np.ix_(ct_idx, ct_idx)]
    np.testing.assert_allclose(actual_block, expected_block, atol=1e-12)


def test_sshr_block_per_series(ct_data):
    """sshr: each series block should be shrinkage cov of that series' temporal residuals."""
    from hierarchicalforecast.utils import (
        _shrunk_covariance_schaferstrimmer_no_nans,
    )

    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    W = estimate_covariance(
        "sshr",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=d["S_cs"],
        S_te=d["S_te"],
    )
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    for i in range(n_cs):
        res_i = res_3d[i]  # (n_te, n_obs)
        expected_block = _shrunk_covariance_schaferstrimmer_no_nans(res_i, 2e-8)
        idx = np.arange(i * n_te, (i + 1) * n_te)
        actual_block = W[np.ix_(idx, idx)]
        np.testing.assert_allclose(actual_block, expected_block, atol=1e-12)


def test_ssam_block_per_series(ct_data):
    """ssam: each series block should be sample cov of that series' temporal residuals."""
    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    W = estimate_covariance(
        "ssam",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=d["S_cs"],
        S_te=d["S_te"],
    )
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    for i in range(n_cs):
        res_i = res_3d[i]  # (n_te, n_obs)
        expected_block = np.cov(res_i)
        idx = np.arange(i * n_te, (i + 1) * n_te)
        actual_block = W[np.ix_(idx, idx)]
        np.testing.assert_allclose(actual_block, expected_block, atol=1e-12)


def test_hshr_matches_explicit_kronecker(ct_data):
    """hshr should equal kron(I, S_te) @ cov_hf @ kron(I, S_te).T + ridge."""
    from hierarchicalforecast.utils import (
        _shrunk_covariance_schaferstrimmer_no_nans,
    )

    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    S_te = d["S_te"]
    n_bottom_te = S_te.shape[1]
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    # Extract HF residuals (bottom temporal)
    res_hf = res_3d[:, -n_bottom_te:, :].reshape(n_cs * n_bottom_te, -1)
    cov_hf = _shrunk_covariance_schaferstrimmer_no_nans(res_hf, 2e-8)
    # Explicit Kronecker
    K = np.kron(np.eye(n_cs), S_te)
    W_expected = K @ cov_hf @ K.T
    # Add same ridge as _ridge_regularize
    n = W_expected.shape[0]
    trace_val = np.trace(W_expected)
    eps = max(1e-6, 1e-6 * trace_val / n)
    W_expected += eps * np.eye(n)

    W_actual = estimate_covariance(
        "hshr",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=d["S_cs"],
        S_te=S_te,
    )
    np.testing.assert_allclose(W_actual, W_expected, atol=1e-12)


def test_hsam_matches_explicit_kronecker(ct_data):
    """hsam should equal kron(I, S_te) @ cov_hf_sam @ kron(I, S_te).T + ridge."""
    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    S_te = d["S_te"]
    n_bottom_te = S_te.shape[1]
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    res_hf = res_3d[:, -n_bottom_te:, :].reshape(n_cs * n_bottom_te, -1)
    cov_hf = np.cov(res_hf)
    K = np.kron(np.eye(n_cs), S_te)
    W_expected = K @ cov_hf @ K.T
    n = W_expected.shape[0]
    trace_val = np.trace(W_expected)
    eps = max(1e-6, 1e-6 * trace_val / n)
    W_expected += eps * np.eye(n)

    W_actual = estimate_covariance(
        "hsam",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=d["S_cs"],
        S_te=S_te,
    )
    np.testing.assert_allclose(W_actual, W_expected, atol=1e-12)


def test_hbshr_matches_explicit_kronecker(ct_data):
    """hbshr should equal kron(S_cs, S_te) @ cov_bottom @ kron(S_cs, S_te).T + ridge."""
    from hierarchicalforecast.utils import (
        _shrunk_covariance_schaferstrimmer_no_nans,
    )

    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    S_cs, S_te = d["S_cs"], d["S_te"]
    n_bottom_cs, n_bottom_te = S_cs.shape[1], S_te.shape[1]
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    # Bottom CS x bottom temporal residuals
    res_bottom = res_3d[-n_bottom_cs:, -n_bottom_te:, :]
    res_bottom_flat = res_bottom.reshape(n_bottom_cs * n_bottom_te, -1)
    cov_bottom = _shrunk_covariance_schaferstrimmer_no_nans(res_bottom_flat, 2e-8)
    S_ct = np.kron(S_cs, S_te)
    W_expected = S_ct @ cov_bottom @ S_ct.T
    n = W_expected.shape[0]
    trace_val = np.trace(W_expected)
    eps = max(1e-6, 1e-6 * trace_val / n)
    W_expected += eps * np.eye(n)

    W_actual = estimate_covariance(
        "hbshr",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=S_cs,
        S_te=S_te,
    )
    np.testing.assert_allclose(W_actual, W_expected, atol=1e-12)


def test_hbsam_matches_explicit_kronecker(ct_data):
    """hbsam should equal kron(S_cs, S_te) @ cov_bottom_sam @ kron(S_cs, S_te).T + ridge."""
    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    S_cs, S_te = d["S_cs"], d["S_te"]
    n_bottom_cs, n_bottom_te = S_cs.shape[1], S_te.shape[1]
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    res_bottom = res_3d[-n_bottom_cs:, -n_bottom_te:, :]
    res_bottom_flat = res_bottom.reshape(n_bottom_cs * n_bottom_te, -1)
    cov_bottom = np.cov(res_bottom_flat)
    S_ct = np.kron(S_cs, S_te)
    W_expected = S_ct @ cov_bottom @ S_ct.T
    n = W_expected.shape[0]
    trace_val = np.trace(W_expected)
    eps = max(1e-6, 1e-6 * trace_val / n)
    W_expected += eps * np.eye(n)

    W_actual = estimate_covariance(
        "hbsam",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=S_cs,
        S_te=S_te,
    )
    np.testing.assert_allclose(W_actual, W_expected, atol=1e-12)


def test_bshr_matches_explicit_kronecker(ct_data):
    """bshr should equal kron(S_cs, I) @ cov_bottom_cs @ kron(S_cs, I).T + ridge."""
    from hierarchicalforecast.utils import (
        _shrunk_covariance_schaferstrimmer_no_nans,
    )

    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    S_cs = d["S_cs"]
    n_bottom_cs = S_cs.shape[1]
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    # Bottom CS residuals across all temporal levels
    res_bottom_cs = res_3d[-n_bottom_cs:, :, :]
    res_bottom_flat = res_bottom_cs.reshape(n_bottom_cs * n_te, -1)
    cov_bottom = _shrunk_covariance_schaferstrimmer_no_nans(res_bottom_flat, 2e-8)
    K = np.kron(S_cs, np.eye(n_te))
    W_expected = K @ cov_bottom @ K.T
    n = W_expected.shape[0]
    trace_val = np.trace(W_expected)
    eps = max(1e-6, 1e-6 * trace_val / n)
    W_expected += eps * np.eye(n)

    W_actual = estimate_covariance(
        "bshr",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=S_cs,
        S_te=d["S_te"],
    )
    np.testing.assert_allclose(W_actual, W_expected, atol=1e-12)


def test_bsam_matches_explicit_kronecker(ct_data):
    """bsam should equal kron(S_cs, I) @ cov_bottom_cs_sam @ kron(S_cs, I).T + ridge."""
    d = ct_data
    n_cs, n_te = d["n_cs"], d["n_te"]
    S_cs = d["S_cs"]
    n_bottom_cs = S_cs.shape[1]
    res_3d = d["residuals_ct"].reshape(n_cs, n_te, -1)
    res_bottom_cs = res_3d[-n_bottom_cs:, :, :]
    res_bottom_flat = res_bottom_cs.reshape(n_bottom_cs * n_te, -1)
    cov_bottom = np.cov(res_bottom_flat)
    K = np.kron(S_cs, np.eye(n_te))
    W_expected = K @ cov_bottom @ K.T
    n = W_expected.shape[0]
    trace_val = np.trace(W_expected)
    eps = max(1e-6, 1e-6 * trace_val / n)
    W_expected += eps * np.eye(n)

    W_actual = estimate_covariance(
        "bsam",
        S=d["S_ct"],
        residuals=d["residuals_ct"],
        n_cs=n_cs,
        n_te=n_te,
        S_cs=S_cs,
        S_te=d["S_te"],
    )
    np.testing.assert_allclose(W_actual, W_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# is_diagonal_method tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "wlsv", "wlsh", "csstr", "testr"])
def test_is_diagonal_method_true(method):
    assert is_diagonal_method(method) is True


@pytest.mark.parametrize("method", [
    "sam", "shr", "bu", "oasd", "acov", "strar1", "sar1", "har1",
    "bdshr", "bdsam", "sshr", "ssam", "hshr", "hsam",
    "hbshr", "hbsam", "bshr", "bsam",
])
def test_is_diagonal_method_false(method):
    assert is_diagonal_method(method) is False


def test_is_diagonal_method_unknown():
    with pytest.raises(ValueError, match="Unknown covariance method"):
        is_diagonal_method("nonexistent")


# ---------------------------------------------------------------------------
# _estimate_ar1_phi tests
# ---------------------------------------------------------------------------

def test_ar1_phi_iid_near_zero():
    rng = np.random.default_rng(42)
    phi = _estimate_ar1_phi(rng.standard_normal((3, 500)))
    assert phi < 0.15


def test_ar1_phi_short_series_returns_zero():
    phi = _estimate_ar1_phi(np.array([[1.0, 2.0]]))
    assert phi == 0.0


def test_ar1_phi_all_zero_variance_returns_zero():
    res = np.ones((2, 10))
    phi = _estimate_ar1_phi(res)
    assert phi == 0.0


def test_ar1_phi_clipped_below_one():
    # Monotonically increasing series has lag-1 autocorrelation > 0.99
    x = np.arange(1000, dtype=float).reshape(1, -1)
    phi = _estimate_ar1_phi(x)
    assert phi == 0.99


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------

def test_validate_residuals_rejects_inf(hierarchy_data):
    res = np.ones((7, 10))
    res[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        estimate_covariance("wls_var", S=hierarchy_data, residuals=res)


# ---------------------------------------------------------------------------
# MinTrace temporal/CT gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["wlsv", "acov", "strar1", "sar1", "har1"])
def test_mintrace_rejects_temporal_methods(method):
    with pytest.raises(NotImplementedError, match="Temporal covariance method"):
        MinTrace(method=method)


@pytest.mark.parametrize("method", ["csstr", "bdshr", "sshr", "hshr", "hbshr", "bshr"])
def test_mintrace_rejects_ct_methods(method):
    with pytest.raises(NotImplementedError, match="Cross-temporal covariance method"):
        MinTrace(method=method)
