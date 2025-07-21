# from nbdev.showdoc import show_doc
import numpy as np
import pandas as pd
from fastcore.test import test_close, test_eq, test_fail

from hierarchicalforecast.core import HierarchicalReconciliation, _build_fn_name
from hierarchicalforecast.methods import BottomUp, MinTrace

test_eq(_build_fn_name(BottomUp()), 'BottomUp')
test_eq(
    _build_fn_name(MinTrace(method='ols')), 
    'MinTrace_method-ols'
)
test_eq(
    _build_fn_name(MinTrace(method='ols', nonnegative=True)), 
    'MinTrace_method-ols_nonnegative-True'
)
test_eq(
    _build_fn_name(MinTrace(method='mint_shrink')), 
    'MinTrace_method-mint_shrink'
)

from hierarchicalforecast.methods import (
    ERM,
    BottomUp,
    MiddleOut,
    MinTrace,
    TopDown,
)
from hierarchicalforecast.utils import aggregate

df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
df.insert(0, 'Country', 'Australia')
df['ds'] = df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
df['ds'] = pd.to_datetime(df['ds'])

# non strictly hierarchical structure
hierS_grouped_df = [
    ['Country'],
    ['Country', 'State'], 
    ['Country', 'Purpose'], 
    ['Country', 'State', 'Region'], 
    ['Country', 'State', 'Purpose'], 
    ['Country', 'State', 'Region', 'Purpose']
]
# strictly hierarchical structure
hiers_strictly = [
    ['Country'],
    ['Country', 'State'], 
    ['Country', 'State', 'Region'], 
]

# getting df
hier_grouped_df, S_grouped_df, tags_grouped = aggregate(df, hierS_grouped_df)
hier_strict_df, S_strict, tags_strict = aggregate(df, hiers_strictly)

# check categorical input produces same output
df2 = df.copy()
for col in ['Country', 'State', 'Purpose', 'Region']:
    df2[col] = df2[col].astype('category')

for spec in [hierS_grouped_df, hiers_strictly]:
    Y_orig, S_orig, tags_orig = aggregate(df, spec)
    Y_cat, S_cat, tags_cat = aggregate(df2, spec)
    pd.testing.assert_frame_equal(Y_cat, Y_orig)
    pd.testing.assert_frame_equal(S_cat, S_orig)
    assert all(np.array_equal(tags_orig[k], tags_cat[k]) for k in tags_orig.keys())
# polars
import polars as pl
import polars.testing as pltest

# polars
df_pl = pl.DataFrame(df)

# getting df
hier_grouped_df_pl, S_grouped_df_pl, tags_grouped_pl = aggregate(df_pl, hierS_grouped_df)
hier_strict_df_pl, S_strict_pl, tags_strict_pl = aggregate(df_pl, hiers_strictly)

# check categorical input produces same output
df2_pl = df_pl.clone()
for col in ['Country', 'State', 'Purpose', 'Region']:
    df2_pl = df2_pl.with_columns(pl.col(col).cast(pl.Categorical))

for spec in [hierS_grouped_df, hiers_strictly]:
    Y_orig_pl, S_orig_pl, tags_orig_pl = aggregate(df_pl, spec)
    Y_cat_pl, S_cat_pl, tags_cat_pl = aggregate(df2_pl, spec)
    pltest.assert_frame_equal(Y_cat_pl, Y_orig_pl)
    pltest.assert_frame_equal(S_cat_pl, S_orig_pl)
    assert all(np.array_equal(tags_orig_pl[k], tags_cat_pl[k]) for k in tags_orig_pl.keys())

hier_grouped_df['y_model'] = hier_grouped_df['y']
# we should be able to recover y using the methods
hier_grouped_hat_df = hier_grouped_df.groupby('unique_id').tail(12)
ds_h = hier_grouped_hat_df['ds'].unique()
hier_grouped_df_filtered = hier_grouped_df.query('~(ds in @ds_h)').copy()
# adding noise to `y_model` to avoid perfect fited values
hier_grouped_df_filtered['y_model'] += np.random.uniform(-1, 1, len(hier_grouped_df_filtered))

#hierachical reconciliation
hrec = HierarchicalReconciliation(reconcilers=[
    #these methods should reconstruct the original y
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='wls_var'),
    MinTrace(method='mint_shrink'),
    MinTrace(method='ols', nonnegative=True),
    MinTrace(method='wls_struct', nonnegative=True),
    MinTrace(method='wls_var', nonnegative=True),
    MinTrace(method='mint_shrink', nonnegative=True),
])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df, 
                            Y_df=hier_grouped_df_filtered, 
                            S=S_grouped_df, tags=tags_grouped)
for model in reconciled.drop(columns=["unique_id", "ds", "y"]).columns:
    if 'ERM' in model:
        eps = 3
    elif 'nonnegative' in model:
        eps = 1e-1
    else:
        eps = 1e-1
    test_close(reconciled['y'], reconciled[model], eps=eps)
# polars
hier_grouped_hat_df_pl = pl.from_pandas(hier_grouped_hat_df)
hier_grouped_df_filtered_pl = pl.from_pandas(hier_grouped_df_filtered)
S_grouped_df_pl = pl.from_pandas(S_grouped_df)

reconciled_pl = hrec.reconcile(Y_hat_df=hier_grouped_hat_df_pl, 
                            Y_df=hier_grouped_df_filtered_pl, 
                            S=S_grouped_df_pl, 
                            tags=tags_grouped)

for model in reconciled_pl.drop(["unique_id", "ds", "y"]).columns:
    if 'ERM' in model:
        eps = 3
    elif 'nonnegative' in model:
        eps = 1e-1
    else:
        eps = 1e-1
    test_close(reconciled_pl['y'], reconciled_pl[model], eps=eps)
# test incorrect Y_hat_df datatypes
hier_grouped_hat_df_nan = hier_grouped_hat_df.copy()
hier_grouped_hat_df_idx_changed = hier_grouped_hat_df_nan.query("unique_id == 'Australia'").index
hier_grouped_hat_df_nan.loc[hier_grouped_hat_df_idx_changed, 'y_model'] = float('nan')
test_fail(
    hrec.reconcile,
    contains='null values',
    args=(hier_grouped_hat_df_nan, S_grouped_df, tags_grouped, hier_grouped_df),
)

hier_grouped_hat_df_none = hier_grouped_hat_df.copy()
hier_grouped_hat_df_idx_changed = hier_grouped_hat_df_none.query("unique_id == 'Australia'").index
hier_grouped_hat_df_none.loc[hier_grouped_hat_df_idx_changed, 'y_model'] = None
test_fail(
    hrec.reconcile,
    contains='null values',
    args=(hier_grouped_hat_df_none, S_grouped_df, tags_grouped, hier_grouped_df),
)

hier_grouped_hat_df_str = hier_grouped_hat_df.copy()
hier_grouped_hat_df_str['y_model'] = hier_grouped_hat_df_str['y_model'].astype(str)
test_fail(
    hrec.reconcile,
    contains='numeric values',
    args=(hier_grouped_hat_df_str, S_grouped_df, tags_grouped, hier_grouped_df),
)
# polars
# test incorrect Y_hat_df datatypes
hier_grouped_hat_df_nan_pl = pl.from_pandas(hier_grouped_hat_df_nan)
test_fail(
    hrec.reconcile,
    contains='null values',
    args=(hier_grouped_hat_df_nan_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl),
)

hier_grouped_hat_df_none_pl = pl.from_pandas(hier_grouped_hat_df_none)
test_fail(
    hrec.reconcile,
    contains='null values',
    args=(hier_grouped_hat_df_none_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl),
)

hier_grouped_hat_df_str_pl = pl.from_pandas(hier_grouped_hat_df_str)
test_fail(
    hrec.reconcile,
    contains='numeric values',
    args=(hier_grouped_hat_df_str_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl),
)
# test expected error
# different series S and Y_hat_df
drop_idx = hier_grouped_hat_df.query("unique_id == 'Australia'").index
test_fail(
    hrec.reconcile,
    contains='There are unique_ids in S_df that are not in Y_hat_df',
    args=(hier_grouped_hat_df.drop(index=drop_idx), S_grouped_df, tags_grouped, hier_grouped_df),
    
)

drop_idx = S_grouped_df.query("unique_id == 'Australia'").index
test_fail(
    hrec.reconcile,
    contains='There are unique_ids in Y_hat_df that are not in S_df',
    args=(hier_grouped_hat_df, S_grouped_df.drop(index=drop_idx), tags_grouped, hier_grouped_df),
)

drop_idx = hier_grouped_df.query("unique_id == 'Australia'").index
test_fail(
    hrec.reconcile,
    contains='There are unique_ids in Y_hat_df that are not in Y_df',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped, hier_grouped_df.drop(index=drop_idx)),   
)
# polars
# test expected error
# different series S and Y_hat_df
test_fail(
    hrec.reconcile,
    contains='There are unique_ids in S_df that are not in Y_hat_df',
    args=(hier_grouped_hat_df_pl.filter(pl.col("unique_id") != "Australia"), S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl),
)

test_fail(
    hrec.reconcile,
    contains='There are unique_ids in Y_hat_df that are not in S_df',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl.filter(pl.col("unique_id") != "Australia"), tags_grouped_pl, hier_grouped_df_pl),
)

test_fail(
    hrec.reconcile,
    contains='There are unique_ids in Y_hat_df that are not in Y_df',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl.filter(pl.col("unique_id") != "Australia")),   
)
# test expected error
# different columns Y_df and Y_hat_df
hrec = HierarchicalReconciliation(
            reconcilers=[ERM(method='reg_bu', lambda_reg=100)])
test_fail(
    hrec.reconcile,
    contains='Please include ',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped, 
          hier_grouped_df, [80], 'permbu'), # permbu needs y_hat_insample
)
# polars
# test expected error
# different columns Y_df and Y_hat_df
hier_grouped_hat_df_pl = pl.from_pandas(hier_grouped_hat_df)
hier_grouped_df_pl = pl.from_pandas(hier_grouped_df)
S_grouped_df_pl = pl.from_pandas(S_grouped_df)

test_fail(
    hrec.reconcile,
    contains='Please include ',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl, 
          hier_grouped_df_pl, [80], 'permbu'), # permbu needs y_hat_insample
)
# test reconcile method without insample
hrec = HierarchicalReconciliation(reconcilers=[
    #these methods should reconstruct the original y
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='ols', nonnegative=True),
    MinTrace(method='wls_struct', nonnegative=True),
])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df,
                            S=S_grouped_df, tags=tags_grouped)
for model in reconciled.drop(columns=['ds', 'y', 'unique_id']).columns:
    if 'ERM' in model:
        eps = 3
    elif 'nonnegative' in model:
        eps = 1e-1
    else:
        eps = 1e-1
    test_close(reconciled['y'], reconciled[model], eps=eps)
# polars
# test reconcile method without insample
hrec = HierarchicalReconciliation(reconcilers=[
    #these methods should reconstruct the original y
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='ols', nonnegative=True),
    MinTrace(method='wls_struct', nonnegative=True),
])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df_pl,
                            S=S_grouped_df_pl, 
                            tags=tags_grouped_pl)
for model in reconciled.drop(['ds', 'y', 'unique_id']).columns:
    if 'ERM' in model:
        eps = 3
    elif 'nonnegative' in model:
        eps = 1e-1
    else:
        eps = 1e-1
    test_close(reconciled['y'], reconciled[model], eps=eps)
# top down should break
# with non strictly hierarchical structures
hrec = HierarchicalReconciliation([TopDown(method='average_proportions')])
test_fail(
    hrec.reconcile,
    contains='requires strictly hierarchical structures',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped,  hier_grouped_df,)
)
# polars
# top down should break
# with non strictly hierarchical structures
hrec = HierarchicalReconciliation([TopDown(method='average_proportions')])
test_fail(
    hrec.reconcile,
    contains='requires strictly hierarchical structures',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl,  hier_grouped_df_pl,)
)
# methods should work with strictly hierarchical structures
hier_strict_df['y_model'] = hier_strict_df['y']
# we should be able to recover y using the methods
hier_strict_df_h = hier_strict_df.groupby('unique_id').tail(12)
ds_h = hier_strict_df_h['ds'].unique()
hier_strict_df = hier_strict_df.query('~(ds in @ds_h)')
#adding noise to `y_model` to avoid perfect fited values
hier_strict_df['y_model'] += np.random.uniform(-1, 1, len(hier_strict_df))

middle_out_level = 'Country/State'
# hierarchical reconciliation
hrec = HierarchicalReconciliation(reconcilers=[
    #these methods should reconstruct the original y
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='wls_var'),
    MinTrace(method='mint_shrink'),
    MinTrace(method='ols', nonnegative=True),
    MinTrace(method='wls_struct', nonnegative=True),
    MinTrace(method='wls_var', nonnegative=True),
    MinTrace(method='mint_shrink', nonnegative=True),
    # top down doesnt recover the original y
    # but it should recover the total level
    TopDown(method='forecast_proportions'),
    TopDown(method='average_proportions'),
    TopDown(method='proportion_averages'),
    # middle out doesnt recover the original y
    # but it should recover the total level
    MiddleOut(middle_level=middle_out_level, top_down_method='forecast_proportions'),
    MiddleOut(middle_level=middle_out_level, top_down_method='average_proportions'),
    MiddleOut(middle_level=middle_out_level, top_down_method='proportion_averages'),
    # ERM recovers but needs bigger eps
    #ERM(method='reg_bu', lambda_reg=None),
])
reconciled = hrec.reconcile(
    Y_hat_df=hier_strict_df_h, 
    Y_df=hier_strict_df, 
    S=S_strict, 
    tags=tags_strict
)
for model in reconciled.drop(columns=['ds', 'y', 'unique_id']).columns:
    if 'ERM' in model:
        eps = 3
    elif 'nonnegative' in model:
        eps = 1e-1
    else:
        eps = 1e-1
    if 'TopDown' in model:
        if 'forecast_proportions' in model:
            test_close(reconciled['y'], reconciled[model], eps)
        else:
            # top down doesnt recover the original y
            test_fail(
                test_close,
                args=(reconciled['y'], reconciled[model], eps),
            )
        # but it should recover the total level
        total_tag = tags_strict['Country']
        test_close(reconciled[["unique_id", "y"]].query("unique_id == @total_tag[0]")["y"], 
                   reconciled[["unique_id", model]].query("unique_id == @total_tag[0]")[model], 1e-2)
    elif 'MiddleOut' in model:
        if 'forecast_proportions' in model:
            test_close(reconciled['y'], reconciled[model], eps)
        else:
            # top down doesnt recover the original y
            test_fail(
                test_close,
                args=(reconciled['y'], reconciled[model], eps),
            )
        # but it should recover the total level
        total_tag = tags_strict[middle_out_level]
        test_close(reconciled[["unique_id", "y"]].query("unique_id == @total_tag[0]")["y"], 
                   reconciled[["unique_id", model]].query("unique_id == @total_tag[0]")[model], 1e-2)
    else:
        test_close(reconciled['y'], reconciled[model], eps)
# polars
# methods should work with strictly hierarchical structures
hier_strict_df_pl = hier_strict_df_pl.with_columns(hier_strict_df_pl['y'].alias('y_model'))
# we should be able to recover y using the methods
hier_strict_df_h_pl = hier_strict_df_pl.group_by('unique_id').tail(12)
ds_h = set(hier_strict_df_h_pl['ds'])
hier_strict_df_pl = hier_strict_df_pl.filter(~pl.col("ds").is_in(ds_h))
#adding noise to `y_model` to avoid perfect fited values
hier_strict_df_pl = hier_strict_df_pl.with_columns(pl.col('y_model') + np.random.uniform(-1, 1, len(hier_strict_df_pl)))

middle_out_level = 'Country/State'
# hierarchical reconciliation
hrec = HierarchicalReconciliation(reconcilers=[
    #these methods should reconstruct the original y
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='wls_var'),
    MinTrace(method='mint_shrink'),
    MinTrace(method='ols', nonnegative=True),
    MinTrace(method='wls_struct', nonnegative=True),
    MinTrace(method='wls_var', nonnegative=True),
    MinTrace(method='mint_shrink', nonnegative=True),
    # top down doesnt recover the original y
    # but it should recover the total level
    TopDown(method='forecast_proportions'),
    TopDown(method='average_proportions'),
    TopDown(method='proportion_averages'),
    # middle out doesnt recover the original y
    # but it should recover the total level
    MiddleOut(middle_level=middle_out_level, top_down_method='forecast_proportions'),
    MiddleOut(middle_level=middle_out_level, top_down_method='average_proportions'),
    MiddleOut(middle_level=middle_out_level, top_down_method='proportion_averages'),
    # ERM recovers but needs bigger eps
    #ERM(method='reg_bu', lambda_reg=None),
])
reconciled_pl = hrec.reconcile(
    Y_hat_df=hier_strict_df_h_pl, 
    Y_df=hier_strict_df_pl, 
    S=S_strict_pl, 
    tags=tags_strict_pl
)
for model in reconciled_pl.drop(['ds', 'y', 'unique_id']).columns:
    if 'ERM' in model:
        eps = 3
    elif 'nonnegative' in model:
        eps = 1e-1
    else:
        eps = 1e-1
    if 'TopDown' in model:
        if 'forecast_proportions' in model:
            test_close(reconciled_pl['y'], reconciled_pl[model], eps)
        else:
            # top down doesnt recover the original y
            test_fail(
                test_close,
                args=(reconciled_pl['y'], reconciled_pl[model], eps),
            )
        # but it should recover the total level
        total_tag = tags_strict['Country']
        test_close(reconciled_pl[["unique_id", "y"]].filter(pl.col("unique_id") == total_tag[0])["y"], 
                   reconciled_pl[["unique_id", model]].filter(pl.col("unique_id") == total_tag[0])[model], 1e-2)
    elif 'MiddleOut' in model:
        if 'forecast_proportions' in model:
            test_close(reconciled_pl['y'], reconciled_pl[model], eps)
        else:
            # top down doesnt recover the original y
            test_fail(
                test_close,
                args=(reconciled_pl['y'], reconciled_pl[model], eps),
            )
        # but it should recover the total level
        total_tag = tags_strict[middle_out_level]
        test_close(reconciled_pl[["unique_id", "y"]].filter(pl.col("unique_id") == total_tag[0])["y"], 
                   reconciled_pl[["unique_id", model]].filter(pl.col("unique_id") == total_tag[0])[model], 1e-2)
    else:
        test_close(reconciled_pl['y'], reconciled_pl[model], eps)
# test is_balanced behaviour
reconciled_balanced = hrec.reconcile(
    Y_hat_df=hier_strict_df_h, 
    Y_df=hier_strict_df, 
    S=S_strict, 
    tags=tags_strict,
    is_balanced=True,
)
test_close(reconciled.drop(columns=["unique_id", "ds"]).values, reconciled_balanced.drop(columns=["unique_id", "ds"]).values, eps=1e-10)
# polars
# test is_balanced behaviour
reconciled_balanced = hrec.reconcile(
    Y_hat_df=hier_strict_df_h_pl, 
    Y_df=hier_strict_df_pl, 
    S=S_strict_pl, 
    tags=tags_strict_pl,
    is_balanced=True,
)
test_close(reconciled_pl.drop(["unique_id", "ds"]).to_numpy(), reconciled_balanced.drop(["unique_id", "ds"]).to_numpy(), eps=1e-10)
from statsforecast import StatsForecast
from statsforecast.models import RandomWalkWithDrift
from statsforecast.utils import generate_series

# test unbalanced dataset
max_tenure = 24
dates = pd.date_range(start='2019-01-31', freq='ME', periods=max_tenure)
cohort_tenure = [24, 23, 22, 21]

ts_list = []

# Create ts for each cohort
for i in range(len(cohort_tenure)):
    ts_list.append(
        generate_series(n_series=1, freq='ME', min_length=cohort_tenure[i], max_length=cohort_tenure[i]).reset_index() \
            .assign(ult=i) \
            .assign(ds=dates[-cohort_tenure[i]:]) \
            .drop(columns=['unique_id'])
    )
df = pd.concat(ts_list, ignore_index=True)

# Create categories
df.loc[df['ult'] < 2, 'pen'] = 'a'
df.loc[df['ult'] >= 2, 'pen'] = 'b'
# Note that unique id requires strings
df['ult'] = df['ult'].astype(str)

hier_levels = [
    ['pen'],
    ['pen', 'ult'],
]
hier_df, S_df, tags = aggregate(df=df, spec=hier_levels)

train_df = hier_df.query("ds <= @pd.to_datetime('2019-12-31')")
test_df = hier_df.query("ds > @pd.to_datetime('2019-12-31')")

fcst = StatsForecast(
    models=[
        RandomWalkWithDrift(),
    ],
    freq='ME',
    n_jobs=1,
)

hrec = HierarchicalReconciliation(
    reconcilers=[
        BottomUp(),
        MinTrace(method='mint_shrink'),
    ]
)

fcst_df = fcst.forecast(df=train_df, h=12, fitted=True)
fitted_df = fcst.forecast_fitted_values()

fcst_df = hrec.reconcile(
    Y_hat_df=fcst_df,
    Y_df=fitted_df,
    S=S_df,
    tags=tags,
)
# polars
# test unbalanced dataset
df_pl = pl.from_pandas(df)
hier_df_pl, S_df_pl, tags_pl = aggregate(df=df_pl, spec=hier_levels)

train_df = hier_df_pl.filter(pl.col("ds") <= pl.lit('2019-12-31').str.to_date())
test_df = hier_df_pl.filter(pl.col("ds") > pl.lit('2019-12-31').str.to_date())

fcst = StatsForecast(
    models=[
        RandomWalkWithDrift(),
    ],
    freq='1mo',
    n_jobs=1,
)

hrec = HierarchicalReconciliation(
    reconcilers=[
        BottomUp(),
        MinTrace(method='mint_shrink'),
    ]
)

fcst_df_pl = fcst.forecast(df=train_df, h=12, fitted=True)
fitted_df = fcst.forecast_fitted_values()

fcst_df_pl = hrec.reconcile(
    Y_hat_df=fcst_df_pl,
    Y_df=fitted_df,
    S=S_df_pl,
    tags=tags,
)

# Test equivalence
pd.testing.assert_frame_equal(fcst_df, fcst_df_pl.to_pandas())
# MinTrace should break
# with extremely overfitted model, y_model==y
zero_df = hier_grouped_df.copy()
zero_df['y'] = 0
zero_df['y_model'] = 0
hrec = HierarchicalReconciliation([MinTrace(method='mint_shrink')])
test_fail(
    hrec.reconcile,
    contains='Insample residuals',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped,  zero_df)
)
# polars
# MinTrace should break
# with extremely overfitted model, y_model==y
zero_df_pl = pl.from_pandas(zero_df)    
hrec = HierarchicalReconciliation([MinTrace(method='mint_shrink')])
test_fail(
    hrec.reconcile,
    contains='Insample residuals',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl,  zero_df_pl)
)
#test methods that dont use residuals
#even if their signature includes
#that argument
hrec = HierarchicalReconciliation([MinTrace(method='ols')])
reconciled = hrec.reconcile(
    Y_hat_df=hier_grouped_hat_df, 
    Y_df=hier_grouped_df.drop(columns=['y_model']), 
    S=S_grouped_df, 
    tags=tags_grouped
)
for model in reconciled.drop(columns=['ds', 'y', 'unique_id']).columns:
    test_close(reconciled['y'], reconciled[model], eps=1e-1)
# polars
#test methods that dont use residuals
#even if their signature includes
#that argument
hrec = HierarchicalReconciliation([MinTrace(method='ols')])
reconciled = hrec.reconcile(
    Y_hat_df=hier_grouped_hat_df_pl, 
    Y_df=hier_grouped_df_pl.drop(['y_model']), 
    S=S_grouped_df_pl, 
    tags=tags_grouped_pl
)
for model in reconciled.drop(['ds', 'y', 'unique_id']).columns:
    test_close(reconciled['y'], reconciled[model], eps=1e-1)
# test methods with bootstrap prediction intervals
hrec = HierarchicalReconciliation([BottomUp()])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df, 
                            Y_df=hier_grouped_df, S=S_grouped_df, tags=tags_grouped,
                            level=[80, 90], 
                            intervals_method='bootstrap')
total = reconciled.query("unique_id in @tags_grouped['Country/State/Region/Purpose']").groupby('ds').sum().reset_index()
pd.testing.assert_frame_equal(
    total[['ds', 'y_model/BottomUp']],
    reconciled.query("unique_id == 'Australia'")[['ds', 'y_model/BottomUp']].reset_index(drop=True)
)
assert 'y_model/BottomUp-lo-80' in reconciled.columns
# polars
# test methods with bootstrap prediction intervals
hrec = HierarchicalReconciliation([BottomUp()])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df_pl, 
                            Y_df=hier_grouped_df_pl, 
                            S=S_grouped_df_pl, 
                            tags=tags_grouped_pl,
                            level=[80, 90], 
                            intervals_method='bootstrap')
total = reconciled.filter(pl.col("unique_id").is_in(tags_grouped['Country/State/Region/Purpose'])).group_by('ds', maintain_order=True).sum()
pltest.assert_frame_equal(
    total[['ds', 'y_model/BottomUp']],
    reconciled.filter(pl.col("unique_id") == 'Australia')[['ds', 'y_model/BottomUp']]
)
assert 'y_model/BottomUp-lo-80' in reconciled.columns
# test methods with  normality prediction intervals
hier_grouped_hat_df['y_model-lo-80'] = hier_grouped_hat_df['y_model'] - 1.96
hier_grouped_hat_df['y_model-hi-80'] = hier_grouped_hat_df['y_model'] + 1.96
hrec = HierarchicalReconciliation([BottomUp()])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df,
                            Y_df=hier_grouped_df, S=S_grouped_df, tags=tags_grouped,
                            level=[80, 90], 
                            intervals_method='normality')
total = reconciled.query("unique_id in @tags_grouped['Country/State/Region/Purpose']").groupby('ds').sum().reset_index()
pd.testing.assert_frame_equal(
    total[['ds', 'y_model/BottomUp']],
    reconciled.query("unique_id == 'Australia'")[['ds', 'y_model/BottomUp']].reset_index(drop=True)
)
assert 'y_model/BottomUp-lo-80' in reconciled.columns
# polars
# test methods with  normality prediction intervals
hier_grouped_hat_df_pl = pl.from_pandas(hier_grouped_hat_df)
hrec = HierarchicalReconciliation([BottomUp()])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_hat_df_pl,
                            Y_df=hier_grouped_df_pl, 
                            S=S_grouped_df_pl, 
                            tags=tags_grouped_pl,
                            level=[80, 90], 
                            intervals_method='normality')
total = reconciled.filter(pl.col("unique_id").is_in(tags_grouped['Country/State/Region/Purpose'])).group_by('ds', maintain_order=True).sum()
pltest.assert_frame_equal(
    total[['ds', 'y_model/BottomUp']],
    reconciled.filter(pl.col("unique_id") == 'Australia')[['ds', 'y_model/BottomUp']]
)
assert 'y_model/BottomUp-lo-80' in reconciled.columns
# test methods with PERMBU prediction intervals

# test expect error with grouped structure
# (non strictly hierarchical)
hier_grouped_hat_df['y_model-lo-80'] = hier_grouped_hat_df['y_model'] - 1.96
hier_grouped_hat_df['y_model-hi-80'] = hier_grouped_hat_df['y_model'] + 1.96
hrec = HierarchicalReconciliation([BottomUp()])
test_fail(
    hrec.reconcile,
    contains='requires strictly hierarchical structures',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped, hier_grouped_df, [80, 90], 'permbu',)
)

# test PERMBU
hier_strict_df_h['y_model-lo-80'] = hier_strict_df_h['y_model'] - 1.96
hier_strict_df_h['y_model-hi-80'] = hier_strict_df_h['y_model'] + 1.96
hrec = HierarchicalReconciliation([BottomUp()])
reconciled = hrec.reconcile(Y_hat_df=hier_strict_df_h,
                            Y_df=hier_strict_df, 
                            S=S_strict, 
                            tags=tags_strict,
                            level=[80, 90], 
                            intervals_method='permbu')
total = reconciled.query("unique_id in @tags_grouped['Country/State/Region']").groupby('ds').sum().reset_index()
pd.testing.assert_frame_equal(
    total[['ds', 'y_model/BottomUp']],
    reconciled.query("unique_id == 'Australia'")[['ds', 'y_model/BottomUp']].reset_index(drop=True)
)
assert 'y_model/BottomUp-lo-80' in reconciled.columns
# polars
# test methods with PERMBU prediction intervals

# test expect error with grouped structure
# (non strictly hierarchical)
hier_grouped_hat_df_pl = pl.from_pandas(hier_grouped_hat_df)

hrec = HierarchicalReconciliation([BottomUp()])
test_fail(
    hrec.reconcile,
    contains='requires strictly hierarchical structures',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl, [80, 90], 'permbu',)
)

# test PERMBU
hier_strict_df_h_pl = pl.from_pandas(hier_strict_df_h)
hrec = HierarchicalReconciliation([BottomUp()])
reconciled = hrec.reconcile(Y_hat_df=hier_strict_df_h_pl,
                            Y_df=hier_strict_df_pl, 
                            S=S_strict_pl, 
                            tags=tags_strict_pl,
                            level=[80, 90], 
                            intervals_method='permbu')
total = reconciled.filter(pl.col("unique_id").is_in(tags_grouped['Country/State/Region'])).group_by('ds', maintain_order=True).sum()
pltest.assert_frame_equal(
    total[['ds', 'y_model/BottomUp']],
    reconciled.filter(pl.col("unique_id") == 'Australia')[['ds', 'y_model/BottomUp']]
)
assert 'y_model/BottomUp-lo-80' in reconciled.columns
# test methods with Bootraped Bootstap prediction intervals
hrec = HierarchicalReconciliation([BottomUp()])
bootstrap_df = hrec.bootstrap_reconcile(Y_hat_df=hier_grouped_hat_df,
                                        Y_df=hier_grouped_df, S_df=S_grouped_df, tags=tags_grouped,
                                        level=[80, 90],
                                        intervals_method='bootstrap',
                                        num_seeds=2)
assert 'y_model/BottomUp-lo-80' in bootstrap_df.columns
assert 'seed' in bootstrap_df.columns
assert len(set(bootstrap_df["seed"]))==2
# polars
# test methods with Bootraped Bootstap prediction intervals
hrec = HierarchicalReconciliation([BottomUp()])
bootstrap_df = hrec.bootstrap_reconcile(Y_hat_df=hier_grouped_hat_df_pl,
                                        Y_df=hier_grouped_df_pl, 
                                        S_df=S_grouped_df_pl, 
                                        tags=tags_grouped_pl,
                                        level=[80, 90],
                                        intervals_method='bootstrap',
                                        num_seeds=2)
assert 'y_model/BottomUp-lo-80' in bootstrap_df.columns
assert 'seed' in bootstrap_df.columns
assert len(set(bootstrap_df["seed"]))==2
# test level protection for PERMBU and Normality probabilistic methods
hrec = HierarchicalReconciliation([BottomUp()])
test_fail(
    hrec.reconcile,
    contains='Level must be a list containing floating values in the interval [0, 100',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped, hier_grouped_df, [-1, 80, 90], 'permbu',)
)
test_fail(
    hrec.reconcile,
    contains='Level must be a list containing floating values in the interval [0, 100',
    args=(hier_grouped_hat_df, S_grouped_df, tags_grouped, hier_grouped_df, [80, 90, 101], 'normality',)
)
# polars
# test level protection for PERMBU and Normality probabilistic methods
hrec = HierarchicalReconciliation([BottomUp()])
test_fail(
    hrec.reconcile,
    contains='Level must be a list containing floating values in the interval [0, 100',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl, [-1, 80, 90], 'permbu',)
)
test_fail(
    hrec.reconcile,
    contains='Level must be a list containing floating values in the interval [0, 100',
    args=(hier_grouped_hat_df_pl, S_grouped_df_pl, tags_grouped_pl, hier_grouped_df_pl, [80, 90, 101], 'normality',)
)
