# test_hierarchical_reconciliation.py

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hierarchicalforecast.core import HierarchicalReconciliation, _build_fn_name
from hierarchicalforecast.methods import (
    BottomUp,
    MiddleOut,
    MinTrace,
    TopDown,
)
from hierarchicalforecast.utils import aggregate

#region: Fixtures

@pytest.fixture(scope="module")
def common_data():
    """Prepares the base tourism dataframe and hierarchy specifications."""
    df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
    df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
    df.insert(0, 'Country', 'Australia')
    df['ds'] = pd.to_datetime(df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True))
    
    # Hierarchy specifications
    hier_grouped_spec = [
        ['Country'], ['Country', 'State'], ['Country', 'Purpose'], 
        ['Country', 'State', 'Region'], ['Country', 'State', 'Purpose'], 
        ['Country', 'State', 'Region', 'Purpose']
    ]
    hier_strict_spec = [
        ['Country'], ['Country', 'State'], ['Country', 'State', 'Region']
    ]
    
    return {
        "df": df,
        "grouped_spec": hier_grouped_spec,
        "strict_spec": hier_strict_spec
    }

@pytest.fixture(scope="module")
def grouped_data(common_data):
    """Prepares data for a grouped (non-strictly hierarchical) structure."""
    df = common_data['df']
    spec = common_data['grouped_spec']
    
    # Pandas
    Y_df, S_df, tags = aggregate(df, spec)
    Y_df['y_model'] = Y_df['y']
    Y_hat_df = Y_df.groupby('unique_id').tail(12).copy()
    ds_h = Y_hat_df['ds'].unique()
    Y_train_df = Y_df.query('~(ds in @ds_h)').copy()
    Y_train_df['y_model'] += np.random.uniform(-1, 1, len(Y_train_df))
    
    # Polars
    df_pl = pl.from_pandas(df)
    Y_df_pl, S_df_pl, tags_pl = aggregate(df_pl, spec)
    Y_hat_df_pl = pl.from_pandas(Y_hat_df)
    Y_train_df_pl = pl.from_pandas(Y_train_df)

    return {
        "Y_hat_df": Y_hat_df, "Y_train_df": Y_train_df, "S_df": S_df, "tags": tags,
        "Y_hat_df_pl": Y_hat_df_pl, "Y_train_df_pl": Y_train_df_pl, "S_df_pl": S_df_pl, "tags_pl": tags_pl
    }
    
@pytest.fixture(scope="module")
def strict_data(common_data):
    """Prepares data for a strictly hierarchical structure."""
    df = common_data['df']
    spec = common_data['strict_spec']
    middle_out_level = 'Country/State'

    # Pandas
    Y_df, S_df, tags = aggregate(df, spec)
    Y_df['y_model'] = Y_df['y']
    Y_hat_df = Y_df.groupby('unique_id').tail(12).copy()
    ds_h = Y_hat_df['ds'].unique()
    Y_train_df = Y_df.query('~(ds in @ds_h)').copy()
    Y_train_df['y_model'] += np.random.uniform(-1, 1, len(Y_train_df))

    # Polars
    df_pl = pl.from_pandas(df)
    Y_df_pl, S_df_pl, tags_pl = aggregate(df_pl, spec)
    Y_hat_df_pl = pl.from_pandas(Y_hat_df)
    Y_train_df_pl = pl.from_pandas(Y_train_df)

    return {
        "Y_hat_df": Y_hat_df, "Y_train_df": Y_train_df, "S_df": S_df, "tags": tags,
        "Y_hat_df_pl": Y_hat_df_pl, "Y_train_df_pl": Y_train_df_pl, "S_df_pl": S_df_pl, "tags_pl": tags_pl,
        "middle_out_level": middle_out_level
    }

#endregion

#region: Tests

def test_fn_name():
    """Tests the _build_fn_name utility function."""
    assert _build_fn_name(BottomUp()) == 'BottomUp'
    assert _build_fn_name(MinTrace(method='ols')) == 'MinTrace_method-ols'
    assert _build_fn_name(MinTrace(method='ols', nonnegative=True)) == 'MinTrace_method-ols_nonnegative-True'
    assert _build_fn_name(MinTrace(method='mint_shrink')) == 'MinTrace_method-mint_shrink'

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_reconciliation_recovers_original_y(grouped_data, lib):
    """Tests if various reconciliation methods can reconstruct the original `y` values."""
    Y_hat = grouped_data[f'Y_hat_df{"" if lib == "pandas" else "_pl"}']
    Y_train = grouped_data[f'Y_train_df{"" if lib == "pandas" else "_pl"}']
    S = grouped_data[f'S_df{"" if lib == "pandas" else "_pl"}']
    tags = grouped_data['tags']
    
    reconcilers = [
        BottomUp(), MinTrace(method='ols'), MinTrace(method='wls_struct'),
        MinTrace(method='wls_var'), MinTrace(method='mint_shrink'),
        MinTrace(method='ols', nonnegative=True), MinTrace(method='wls_struct', nonnegative=True),
        MinTrace(method='wls_var', nonnegative=True), MinTrace(method='mint_shrink', nonnegative=True),
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    reconciled_df = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S=S, tags=tags)
    
    drop_cols = ["unique_id", "ds", "y"]
    models = reconciled_df.drop(columns=drop_cols).columns if lib == "pandas" else reconciled_df.drop(drop_cols).columns

    for model in models:
        eps = 1e-1
        y_true = reconciled_df['y'].to_numpy() if lib == "polars" else reconciled_df['y']
        y_pred = reconciled_df[model].to_numpy() if lib == "polars" else reconciled_df[model]
        np.testing.assert_allclose(y_true, y_pred, atol=eps)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
@pytest.mark.parametrize("modifier, error_msg", [
    ("nan", "null values"), ("none", "null values"), ("str", "numeric")
])
def test_reconcile_raises_on_invalid_y_hat_dtype(grouped_data, lib, modifier, error_msg):
    """Tests for failures when Y_hat_df contains non-numeric or null data."""
    # Skip nan test for polars since it doesn't treat NaN as null
    if lib == "polars" and modifier == "nan":
        pytest.skip("Polars doesn't treat NaN as null values")
        
    Y_hat_df = grouped_data[f'Y_hat_df{"" if lib == "pandas" else "_pl"}']
    S_df = grouped_data[f'S_df{"" if lib == "pandas" else "_pl"}']
    Y_train_df = grouped_data[f'Y_train_df{"" if lib == "pandas" else "_pl"}']
    tags = grouped_data['tags']
    hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])
    
    Y_hat_invalid = Y_hat_df.clone() if lib == "polars" else Y_hat_df.copy()

    if lib == "pandas":
        idx = Y_hat_invalid.query("unique_id == 'Australia'").index
        val = np.nan if modifier == "nan" else None if modifier == "none" else "a_string"
        Y_hat_invalid.loc[idx, 'y_model'] = val
        if modifier == "str":
            Y_hat_invalid['y_model'] = Y_hat_invalid['y_model'].astype(str) # Ensure entire column is string
    else: # polars
        if modifier == "str":
            Y_hat_invalid = Y_hat_invalid.with_columns(pl.col('y_model').cast(pl.String))
        else:
            val = None if modifier == "none" else float('nan')
            Y_hat_invalid = Y_hat_invalid.with_columns(
                pl.when(pl.col('unique_id') == 'Australia').then(pl.lit(val)).otherwise(pl.col('y_model')).alias('y_model')
            )

    with pytest.raises(ValueError, match=error_msg):
        hrec.reconcile(Y_hat_df=Y_hat_invalid, S=S_df, tags=tags, Y_df=Y_train_df)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_topdown_raises_on_non_strict_hierarchy(grouped_data, lib):
    """Tests that TopDown fails for non-strictly hierarchical (grouped) data."""
    Y_hat = grouped_data[f'Y_hat_df{"" if lib == "pandas" else "_pl"}']
    Y_train = grouped_data[f'Y_train_df{"" if lib == "pandas" else "_pl"}']
    S = grouped_data[f'S_df{"" if lib == "pandas" else "_pl"}']
    tags = grouped_data['tags']
    hrec = HierarchicalReconciliation([TopDown(method='average_proportions')])

    with pytest.raises(ValueError, match='Top-down reconciliation requires strictly hierarchical structures'):
        hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S=S, tags=tags)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_reconciliation_with_strict_hierarchy(strict_data, lib):
    """Tests reconciliation methods with a strictly hierarchical structure."""
    key_suffix = "" if lib == "pandas" else "_pl"
    Y_hat, Y_train, S, tags, middle_out_level = (
        strict_data[f'Y_hat_df{key_suffix}'], strict_data[f'Y_train_df{key_suffix}'],
        strict_data[f'S_df{key_suffix}'], strict_data['tags'], strict_data['middle_out_level']
    )
    
    reconcilers = [
        BottomUp(), MinTrace(method='ols'), TopDown(method='forecast_proportions'),
        TopDown(method='average_proportions'), MiddleOut(middle_level=middle_out_level, top_down_method='forecast_proportions')
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    reconciled = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S=S, tags=tags)

    y_true = reconciled['y'].to_numpy() if lib == "polars" else reconciled['y']
    for model_name in ['y_model/BottomUp', 'y_model/MinTrace_method-ols', 'y_model/TopDown_method-forecast_proportions']:
        y_pred = reconciled[model_name].to_numpy() if lib == "polars" else reconciled[model_name]
        np.testing.assert_allclose(y_true, y_pred, atol=1e-1)
    
    y_pred_fail = reconciled['y_model/TopDown_method-average_proportions'].to_numpy() if lib == "polars" else reconciled['y_model/TopDown_method-average_proportions']
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_true, y_pred_fail, atol=1e-1)
    
    total_tag = tags['Country'][0]
    total_y_df = reconciled.filter(pl.col("unique_id") == total_tag) if lib == 'polars' else reconciled.query(f"unique_id == '{total_tag}'")
    
    y_true_total = total_y_df['y'].to_numpy() if lib == "polars" else total_y_df['y']
    y_pred_total = total_y_df['y_model/TopDown_method-average_proportions'].to_numpy() if lib == "polars" else total_y_df['y_model/TopDown_method-average_proportions']
    np.testing.assert_allclose(y_true_total, y_pred_total, atol=1e-2)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_mintrace_raises_on_zero_residuals(grouped_data, lib):
    """Tests that MinTrace fails gracefully when insample residuals are all zero."""
    key_suffix = "" if lib == "pandas" else "_pl"
    Y_hat, S, tags = grouped_data[f'Y_hat_df{key_suffix}'], grouped_data[f'S_df{key_suffix}'], grouped_data['tags']
    
    zero_df = grouped_data[f'Y_train_df{key_suffix}'].clone() if lib == 'polars' else grouped_data[f'Y_train_df{key_suffix}'].copy()
    if lib == 'pandas':
        zero_df['y'] = 0
        zero_df['y_model'] = 0
    else:
        zero_df = zero_df.with_columns([pl.lit(0, dtype=pl.Float64).alias('y'), pl.lit(0, dtype=pl.Float64).alias('y_model')])

    hrec = HierarchicalReconciliation([MinTrace(method='mint_shrink')])
    with pytest.raises(Exception, match='Insample residuals close to 0'):
        hrec.reconcile(Y_hat_df=Y_hat, S=S, tags=tags, Y_df=zero_df)

@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_permbu_raises_on_non_strict_hierarchy(grouped_data, lib):
    """Ensures permbu intervals fail with non-strictly hierarchical data."""
    key_suffix = "" if lib == "pandas" else "_pl"
    Y_hat, Y_train, S, tags = (
        grouped_data[f'Y_hat_df{key_suffix}'], grouped_data[f'Y_train_df{key_suffix}'],
        grouped_data[f'S_df{key_suffix}'], grouped_data['tags']
    )
    
    # Add prediction intervals to test the hierarchy validation
    Y_hat_with_intervals = Y_hat.copy() if lib == "pandas" else Y_hat.clone()
    if lib == "pandas":
        Y_hat_with_intervals['y_model-lo-80'] = Y_hat_with_intervals['y_model'] * 0.9
        Y_hat_with_intervals['y_model-hi-80'] = Y_hat_with_intervals['y_model'] * 1.1
    else:  # polars
        Y_hat_with_intervals = Y_hat_with_intervals.with_columns([
            (pl.col('y_model') * 0.9).alias('y_model-lo-80'),
            (pl.col('y_model') * 1.1).alias('y_model-hi-80')
        ])
    
    hrec = HierarchicalReconciliation([BottomUp()])
    with pytest.raises(ValueError, match='PERMBU probabilistic reconciliation requires strictly hierarchical structures'):
        hrec.reconcile(
            Y_hat_df=Y_hat_with_intervals, Y_df=Y_train, S=S, tags=tags,
            level=[80], intervals_method='permbu'
        )

@pytest.mark.parametrize("lib", ["pandas", "polars"])
@pytest.mark.parametrize("level, method", [([-1, 80], 'permbu'), ([80, 101], 'normality')])
def test_reconcile_raises_on_invalid_level(grouped_data, lib, level, method):
    """Tests failure when prediction interval levels are outside the [0, 100] range."""
    key_suffix = "" if lib == "pandas" else "_pl"
    Y_hat, Y_train, S, tags = (
        grouped_data[f'Y_hat_df{key_suffix}'], grouped_data[f'Y_train_df{key_suffix}'],
        grouped_data[f'S_df{key_suffix}'], grouped_data['tags']
    )
    hrec = HierarchicalReconciliation([BottomUp()])
    with pytest.raises(ValueError, match=r"Level must be a list containing floating values in the interval \[0, 100\)"):
         hrec.reconcile(
            Y_hat_df=Y_hat, Y_df=Y_train, S=S, tags=tags,
            level=level, intervals_method=method
        )
