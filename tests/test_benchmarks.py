import numpy as np
import pytest
from scipy import sparse

from hierarchicalforecast.methods import (
    _flatten_child_nodes,
    _get_child_nodes,
    _reconcile_fcst_proportions_bootstrap,
    _reconcile_fcst_proportions_native,
)
from hierarchicalforecast.probabilistic_methods import Bootstrap
from hierarchicalforecast.utils import (
    _lasso,
    _ma_cov,
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)

from .conftest import (
    _make_random_strict_hierarchy,
    _reconcile_fcst_proportions_bootstrap_python,
    _reconcile_fcst_proportions_columns_python,
    _topdown_idxs_top,
)

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


@pytest.fixture
def fcst_proportions_bench_data():
    """1000-bottom-series/3-level hierarchy, h=12, for benchmarking the
    forecast-proportions traversal (old per-column Python loop vs the native
    batched kernel)."""
    rng = np.random.default_rng(789)
    S, tags = _make_random_strict_hierarchy(
        rng, level_sizes=[1, 20, 1000], shuffle_tags=False
    )
    n_hiers = S.shape[0]
    h = 12
    insample_size = 60

    y_hat = 100.0 + 10.0 * rng.standard_normal((n_hiers, h))
    y_insample = 100.0 + 10.0 * rng.standard_normal((n_hiers, insample_size))
    y_hat_insample = y_insample + rng.standard_normal((n_hiers, insample_size))

    levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
    nodes = _get_child_nodes(S=S, tags=levels_)
    flat = _flatten_child_nodes(nodes)
    idxs_top = _topdown_idxs_top(S)
    return {
        "S": S,
        "tags": levels_,
        "nodes": nodes,
        "flat": flat,
        "idxs_top": idxs_top,
        "y_hat": y_hat,
        "y_insample": y_insample,
        "y_hat_insample": y_hat_insample,
    }


@pytest.mark.parametrize("impl", ["python", "native"])
def test_bench_fcst_proportions_mean(benchmark, fcst_proportions_bench_data, impl):
    """TopDown `forecast_proportions` mean path: traversal over all 12
    horizon columns."""
    d = fcst_proportions_bench_data
    if impl == "python":
        benchmark(
            _reconcile_fcst_proportions_columns_python,
            S=d["S"],
            y_hat=d["y_hat"],
            tags=d["tags"],
            nodes=d["nodes"],
            idxs_top=d["idxs_top"],
        )
    else:
        benchmark(
            _reconcile_fcst_proportions_native,
            d["y_hat"][None, :, :],
            idxs_top=d["idxs_top"],
            flat=d["flat"],
        )


@pytest.mark.parametrize("impl", ["python", "native"])
def test_bench_fcst_proportions_bootstrap_100(
    benchmark, fcst_proportions_bench_data, impl
):
    """Bootstrap prediction intervals with 100 samples (the per-sample
    multiplier is what made the Python loop explode)."""
    d = fcst_proportions_bench_data
    if impl == "python":
        benchmark.pedantic(
            _reconcile_fcst_proportions_bootstrap_python,
            kwargs=dict(
                S=d["S"],
                y_hat=d["y_hat"],
                tags=d["tags"],
                y_insample=d["y_insample"],
                y_hat_insample=d["y_hat_insample"],
                num_samples=100,
                seed=0,
                level=[80, 95],
                nodes=d["nodes"],
                idxs_top=d["idxs_top"],
            ),
            rounds=3,
            iterations=1,
        )
    else:
        benchmark(
            _reconcile_fcst_proportions_bootstrap,
            S=d["S"],
            y_hat=d["y_hat"],
            tags=d["tags"],
            y_insample=d["y_insample"],
            y_hat_insample=d["y_hat_insample"],
            num_samples=100,
            seed=0,
            level=[80, 95],
            nodes=d["nodes"],
            idxs_top=d["idxs_top"],
            flat=d["flat"],
        )


def test_bench_fcst_proportions_bootstrap_1000_native(
    benchmark, fcst_proportions_bench_data
):
    """Native-only: bootstrap with 1000 samples (12,000 traversals per call),
    infeasible to benchmark with the old Python loop at this size."""
    d = fcst_proportions_bench_data
    benchmark(
        _reconcile_fcst_proportions_bootstrap,
        S=d["S"],
        y_hat=d["y_hat"],
        tags=d["tags"],
        y_insample=d["y_insample"],
        y_hat_insample=d["y_hat_insample"],
        num_samples=1000,
        seed=0,
        level=[80, 95],
        nodes=d["nodes"],
        idxs_top=d["idxs_top"],
        flat=d["flat"],
    )
