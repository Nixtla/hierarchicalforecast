import numpy as np
import pandas as pd
import pytest

from hierarchicalforecast.utils import aggregate


def assert_raises_with_message(func, expected_msg, *args, **kwargs):
    with pytest.raises((AssertionError, ValueError, Exception)) as exc_info:
        func(*args, **kwargs)
    assert expected_msg in str(exc_info.value)


def _random_partition(rng, n_items, n_parts):
    """Split `n_items` (shuffled) indices into `n_parts` non-empty groups."""
    order = rng.permutation(n_items)
    if n_parts > 1:
        cuts = sorted(rng.choice(np.arange(1, n_items), size=n_parts - 1, replace=False))
    else:
        cuts = []
    groups = []
    prev = 0
    for cut in [*cuts, n_items]:
        groups.append(order[prev:cut])
        prev = cut
    return groups


def _make_random_strict_hierarchy(rng, level_sizes, shuffle_tags=True):
    """Build a random strictly hierarchical `S` matrix and `tags` for testing.

    `level_sizes` goes from top to bottom, e.g. `[1, 5, 9, 40]`. Each node's
    children are a random non-empty partition of the level below, so the
    result is guaranteed strictly hierarchical. Row order within each level's
    `tags` entry is shuffled by default (row indices are still valid, but not
    in ascending order) to make sure ordering isn't accidentally assumed to
    follow `np.where`'s ascending output.

    Shared by `tests/test_methods.py` (correctness fuzzing) and
    `tests/test_benchmarks.py` (synthetic benchmark hierarchies) so the two
    suites don't drift apart on their own copies of the same generator.
    """
    n_bottom = level_sizes[-1]
    memberships = [np.eye(n_bottom, dtype=np.float64)]
    current_size = n_bottom
    for size in reversed(level_sizes[:-1]):
        groups = _random_partition(rng, current_size, size)
        prev_membership = memberships[-1]
        new_membership = np.zeros((size, n_bottom))
        for i, grp in enumerate(groups):
            new_membership[i] = prev_membership[grp].sum(axis=0)
        memberships.append(new_membership)
        current_size = size
    memberships = list(reversed(memberships))
    S = np.vstack(memberships)

    tags = {}
    row_offset = 0
    for i, size in enumerate(level_sizes):
        idx = np.arange(row_offset, row_offset + size)
        if shuffle_tags:
            idx = rng.permutation(idx)
        tags[f"level{i}"] = idx
        row_offset += size
    return S, tags

def _topdown_idxs_top(S):
    """Replicate `TopDown.fit_predict`'s top-node selection for direct calls
    into the forecast-proportions helpers from tests and benchmarks."""
    S_sum = np.sum(S, axis=1)
    if S.shape[1] > 1:
        S_max_idxs = np.argsort(S_sum)[::-1]
        return S_max_idxs[np.cumsum(S_sum[S_max_idxs]) <= S.shape[1]]
    return np.array([np.argmax(S_sum)])


def _reconcile_fcst_proportions_columns_python(S, y_hat, tags, nodes, idxs_top):
    """Pre-native runtime path: run the pure-Python
    `_reconcile_fcst_proportions` traversal once per horizon column.

    Kept as the behavioral reference the native batched kernel is asserted
    bit-identical against, and as the "before" side of the benchmarks. Shared
    by `tests/test_methods.py` and `tests/test_benchmarks.py`.
    """
    from hierarchicalforecast.methods import _reconcile_fcst_proportions

    return np.hstack(
        [
            _reconcile_fcst_proportions(
                S=S,
                y_hat=y_hat_[:, None],
                tags=tags,
                nodes=nodes,
                idxs_top=idxs_top,
            )
            for y_hat_ in y_hat.T
        ]
    )


def _reconcile_fcst_proportions_bootstrap_python(
    S, y_hat, tags, y_insample, y_hat_insample, num_samples, seed, level, nodes, idxs_top
):
    """Pre-native dense bootstrap path: resample residual blocks and run the
    pure-Python traversal per sample and per horizon column.

    Mirrors `_reconcile_fcst_proportions_bootstrap`'s dense branch before the
    native batched kernel replaced the inner loops; the rng usage is identical
    so, for the same seed, the native path must produce bit-identical
    quantiles. Shared by `tests/test_methods.py` and `tests/test_benchmarks.py`.
    """
    residuals = y_insample - y_hat_insample
    h = y_hat.shape[1]
    residuals = residuals[:, np.isnan(residuals).sum(axis=0) == 0]
    sample_idx = np.arange(residuals.shape[1] - h)
    rng = np.random.default_rng(seed)
    samples_idx = rng.choice(sample_idx, size=num_samples)
    bootstrap_samples = []
    for idx in samples_idx:
        y_hat_sample = y_hat + residuals[:, idx : (idx + h)]
        bootstrap_samples.append(
            _reconcile_fcst_proportions_columns_python(
                S=S, y_hat=y_hat_sample, tags=tags, nodes=nodes, idxs_top=idxs_top
            )
        )
    samples = np.stack(bootstrap_samples)
    samples = samples.transpose((1, 2, 0))
    quantiles = np.concatenate(
        [[(100 - lv) / 200, ((100 - lv) / 200) + lv / 100] for lv in level]
    )
    quantiles = np.sort(quantiles)
    sample_quantiles = np.quantile(samples, quantiles, axis=2)
    return sample_quantiles.transpose((1, 2, 0))


@pytest.fixture(scope="module")
def tourism_df():
    df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    df.insert(0, 'Country', 'Australia')
    return df

@pytest.fixture(scope="module")
def hiers_grouped():
    # grouped structure
    hiers_grouped = [['Country'],
                    ['Country', 'State'],
                    ['Country', 'Purpose'],
                    ['Country', 'State', 'Region'],
                    ['Country', 'State', 'Purpose'],
                    ['Country', 'State', 'Region', 'Purpose'],
                    ]
    return hiers_grouped


@pytest.fixture(scope="module")
def hiers_strictly():
    # strictly hierarchical structure
    hiers_strictly = [['Country'],
                    ['Country', 'State'],
                    ['Country', 'State', 'Region']]
    return hiers_strictly


@pytest.fixture
def strict_hierarchy_data(hiers_strictly):
    """Fixture providing strictly hierarchical data for coherency tests.

    Creates a simple strict hierarchy (Country -> State -> Region) with synthetic data.
    """
    # Create a simple dataset with a strict hierarchy structure
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    data = []

    # Bottom-level structure: Country -> State -> Region
    structure = {
        ("AU", "NSW", "Sydney"): 100,
        ("AU", "NSW", "Newcastle"): 50,
        ("AU", "VIC", "Melbourne"): 80,
        ("AU", "VIC", "Geelong"): 40,
    }

    for (country, state, region), base_val in structure.items():
        for ds in dates:
            data.append({
                "Country": country,
                "State": state,
                "Region": region,
                "ds": ds,
                "y": base_val + np.random.randn() * 10
            })

    df = pd.DataFrame(data)

    # Create aggregated hierarchy using the hiers_strictly fixture
    Y_df, S_df, tags = aggregate(df, hiers_strictly)

    # Prepare train/test split
    Y_df["y_model"] = Y_df["y"]
    Y_hat_df = Y_df.groupby("unique_id").tail(12).copy()
    ds_h = Y_hat_df["ds"].unique()  # noqa: F841
    Y_train_df = Y_df.query("~(ds in @ds_h)").copy()
    Y_train_df["y_model"] += np.random.uniform(-1, 1, len(Y_train_df))

    return {
        "Y_hat_df": Y_hat_df,
        "Y_train_df": Y_train_df,
        "S_df": S_df,
        "tags": tags,
    }
