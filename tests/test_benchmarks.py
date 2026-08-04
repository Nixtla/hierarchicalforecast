import numpy as np
import pytest

from hierarchicalforecast.utils import (
    _lasso,
    _lasso_kron,
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)


@pytest.fixture(params=[50, 200, 500, 1000])
def cov_data(request):
    n_ts = request.param
    n_samples = 100
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal((n_ts, n_samples))
    mask = np.ones_like(residuals, dtype=bool)
    return residuals, mask


def test_bench_ma_cov(benchmark, cov_data):
    residuals, mask = cov_data
    benchmark(_ma_cov, residuals, mask)


def test_bench_shrunk_cov_no_nans(benchmark, cov_data):
    residuals, _ = cov_data
    benchmark(_shrunk_covariance_schaferstrimmer_no_nans, residuals, 2e-8)


def test_bench_shrunk_cov_with_nans(benchmark, cov_data):
    residuals, mask = cov_data
    benchmark(_shrunk_covariance_schaferstrimmer_with_nans, residuals, mask, 2e-8)


@pytest.fixture
def lasso_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 50))
    y = rng.standard_normal(200)
    return X, y


def test_bench_lasso(benchmark, lasso_data):
    X, y = lasso_data
    benchmark(_lasso, X, y, 0.1, 1000, 1e-4)


@pytest.fixture
def lasso_kron_data():
    """Kronecker factors shaped like the ERM reg/reg_bu design matrix.

    S is a summing-matrix-like factor (total + 5 groups + identity) and Y is
    the transposed insample forecast matrix; the implicit design matrix
    X = np.kron(S, Y) has shape (n_hiers * h, n_bottom * n_hiers).
    """
    rng = np.random.default_rng(42)
    n_bottom, h = 50, 8
    S = np.vstack(
        [
            np.ones((1, n_bottom)),
            np.repeat(np.eye(5), n_bottom // 5, axis=1),
            np.eye(n_bottom),
        ]
    )
    Y = rng.standard_normal((h, S.shape[0]))
    y = rng.standard_normal(S.shape[0] * h)
    return S, Y, y


def test_bench_lasso_kron(benchmark, lasso_kron_data):
    S, Y, y = lasso_kron_data
    benchmark(_lasso_kron, S, Y, y, 0.1, 1000, 1e-4)


def test_bench_lasso_kron_materialized_baseline(benchmark, lasso_kron_data):
    """Old ERM path: materialize np.kron(S, Y), then run the dense Lasso."""
    S, Y, y = lasso_kron_data

    def _materialized():
        return _lasso(np.kron(S, Y), y, 0.1, 1000, 1e-4)

    benchmark(_materialized)
