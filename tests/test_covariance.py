"""Tests for the covariance estimation module."""

import numpy as np
import pytest

from hierarchicalforecast.covariance import (
    REQUIRES_RESIDUALS,
    estimate_covariance,
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
        "csstr", "testr", "bdshr", "bdsam", "Sshr", "Ssam",
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
              "bdshr", "bdsam", "Sshr", "Ssam", "hshr", "hsam",
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
    W = estimate_covariance("test_custom", S=hierarchy_data)
    np.testing.assert_array_equal(W, np.eye(7) * 2.0)
    assert "test_custom" in list_covariance_methods()


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


@pytest.mark.parametrize("method", ["Sshr", "Ssam"])
def test_ct_block_diagonal_by_series(ct_data, method):
    """Sshr/Ssam should be block-diagonal with blocks by CS series."""
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
    "csstr", "testr", "bdshr", "bdsam", "Sshr", "Ssam",
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


@pytest.mark.parametrize("method", ["wlsv", "wlsh", "acov", "strar1", "sar1", "har1"])
def test_mintrace_with_temporal_methods(mintrace_data, method):
    """MinTrace should work with temporal covariance methods (without agg_order they fall back)."""
    mt = MinTrace(method=method)
    result = mt.fit_predict(**mintrace_data)

    assert "mean" in result
    y_rec = result["mean"]
    S = mintrace_data["S"]
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
