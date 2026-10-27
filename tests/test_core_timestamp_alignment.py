import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import norm

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, BottomUpSparse


@pytest.fixture
def tiny_forecasts():
    # S order deliberately differs from lexical series order.
    summing = pd.DataFrame(
        {"series": ["total", "b", "a"], "b": [1.0, 1.0, 0.0], "a": [1.0, 0.0, 1.0]}
    )
    forecasts = pd.DataFrame(
        {
            "series": np.repeat(["total", "b", "a"], 2),
            "date": np.tile([1, 2], 3),
            "model": [999.0, 999.0, 10.0, 20.0, 1.0, 2.0],
        }
    )
    z = norm.ppf(0.95)
    forecasts["model-lo-90"] = forecasts["model"] - z
    forecasts["model-hi-90"] = forecasts["model"] + z
    tags = {"top": np.array(["total"]), "bottom": np.array(["b", "a"])}
    return forecasts, summing, tags


@pytest.mark.parametrize("backend", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("reconciler", [BottomUp, BottomUpSparse])
@pytest.mark.parametrize("level", [None, [90]])
@pytest.mark.parametrize("bad_times", [[2, 3], [1, 1]])
def test_reconcile_rejects_misaligned_forecast_timestamps(
    tiny_forecasts, backend, reconciler, level, bad_times
):
    forecasts, summing, tags = tiny_forecasts
    forecasts.loc[forecasts["series"] == "b", "date"] = bad_times
    with pytest.raises(ValueError, match="same timestamps|duplicate"):
        HierarchicalReconciliation([reconciler()]).reconcile(
            Y_hat_df=backend(forecasts.to_dict(orient="list")),
            S_df=backend(summing.to_dict(orient="list")),
            tags=tags,
            level=level,
            id_col="series",
            time_col="date",
        )


@pytest.mark.parametrize("backend", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("reconciler", [BottomUp, BottomUpSparse])
@pytest.mark.parametrize("horizon", [1, 2])
@pytest.mark.parametrize("datetime", [False, True])
def test_reconcile_preserves_shuffled_forecast_alignment(
    tiny_forecasts, backend, reconciler, horizon, datetime
):
    forecasts, summing, tags = tiny_forecasts
    forecasts = forecasts.loc[forecasts["date"] <= horizon].copy()
    if datetime:
        forecasts["date"] = pd.to_datetime(forecasts["date"], unit="D")
    forecasts = forecasts.sample(frac=1, random_state=0)
    result = HierarchicalReconciliation([reconciler()]).reconcile(
        Y_hat_df=backend(forecasts.to_dict(orient="list")),
        S_df=backend(summing.to_dict(orient="list")),
        tags=tags,
        level=[90],
        id_col="series",
        time_col="date",
    )
    model = f"model/{reconciler.__name__}"
    expected = np.array([[11.0, 22.0], [10.0, 20.0], [1.0, 2.0]])[:, :horizon]
    mean = np.asarray(result[model]).reshape(3, horizon)
    lo = np.asarray(result[f"{model}-lo-90"]).reshape(3, horizon)
    hi = np.asarray(result[f"{model}-hi-90"]).reshape(3, horizon)
    np.testing.assert_allclose(mean, expected)
    width = norm.ppf(0.95) * np.sqrt([[2.0], [1.0], [1.0]])
    np.testing.assert_allclose(lo, expected - width)
    np.testing.assert_allclose(hi, expected + width)
