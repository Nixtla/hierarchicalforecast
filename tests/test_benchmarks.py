import numpy as np
import pytest
from scipy import sparse

from hierarchicalforecast.methods import _get_child_nodes
from hierarchicalforecast.probabilistic_methods import Bootstrap
from hierarchicalforecast.utils import (
    _lasso,
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)

from .conftest import _make_random_strict_hierarchy

pytestmark = pytest.mark.benchmark


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
def bootstrap_bench_data():
    """1000 bottom series, 3 levels (top / mid / bottom), matching the
    synthetic hierarchy used to benchmark the `Bootstrap.get_samples`
    vectorization."""
    rng = np.random.default_rng(123)
    S, _ = _make_random_strict_hierarchy(
        rng, level_sizes=[1, 20, 1000], shuffle_tags=False
    )
    n_hiers, n_bottom = S.shape
    h = 12
    insample_size = 60

    y_insample = 100.0 + 10.0 * rng.standard_normal((n_hiers, insample_size))
    y_hat_insample = y_insample + rng.standard_normal((n_hiers, insample_size))
    y_hat = 100.0 + 10.0 * rng.standard_normal((n_hiers, h))
    P = np.eye(n_bottom, n_hiers, n_hiers - n_bottom, np.float64)
    return S, P, y_hat, y_insample, y_hat_insample


@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
def test_bench_bootstrap_get_samples(benchmark, bootstrap_bench_data, as_sparse):
    S, P, y_hat, y_insample, y_hat_insample = bootstrap_bench_data
    if as_sparse:
        S = sparse.csr_matrix(S)
        P = sparse.csr_matrix(P)
    sampler = Bootstrap(
        S=S,
        P=P,
        y_hat=y_hat,
        y_insample=y_insample,
        y_hat_insample=y_hat_insample,
        num_samples=1000,
        seed=0,
    )
    benchmark(sampler.get_samples, num_samples=1000)


@pytest.fixture
def child_nodes_bench_data():
    """Same synthetic 1000-bottom-series/3-level hierarchy, for benchmarking
    `_get_child_nodes` (feeds TopDown/MiddleOut `forecast_proportions`)."""
    rng = np.random.default_rng(456)
    S, tags = _make_random_strict_hierarchy(
        rng, level_sizes=[1, 20, 1000], shuffle_tags=False
    )
    return S, tags


@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
def test_bench_get_child_nodes(benchmark, child_nodes_bench_data, as_sparse):
    S, tags = child_nodes_bench_data
    if as_sparse:
        S = sparse.csr_matrix(S)
    benchmark(_get_child_nodes, S=S, tags=tags)
