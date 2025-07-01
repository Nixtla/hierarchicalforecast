from functools import partial

import numpy as np
import pandas as pd

# polars
import polars as pl
import utilsforecast.losses as ufl

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import HierarchicalEvaluation, evaluate, mse
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate


def rmse(y, y_hat):
    return np.mean(np.sqrt(np.mean((y-y_hat)**2, axis=1)))

def mase(y, y_hat, y_insample, seasonality=4):
    errors = np.mean(np.abs(y - y_hat), axis=1)
    scale = np.mean(np.abs(y_insample[:, seasonality:] - y_insample[:, :-seasonality]), axis=1)
    return np.mean(errors / scale)
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
df.insert(0, 'Country', 'Australia')

# non strictly hierarchical structure
hiers_grouped = [
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
hier_grouped_df, S_grouped, tags_grouped = aggregate(df, hiers_grouped)

#split train/test
hier_grouped_df['y_model'] = hier_grouped_df['y']
# we should be able to recover y using the methods
hier_grouped_df_h = hier_grouped_df.groupby('unique_id').tail(12).reset_index(drop=True)
ds_h = hier_grouped_df_h['ds'].unique()
hier_grouped_df = hier_grouped_df.query('~(ds in @ds_h)')
#adding noise to `y_model` to avoid perfect fited values
rng = np.random.default_rng(0)
hier_grouped_df['y_model'] += rng.uniform(-1, 1, len(hier_grouped_df))

#hierachical reconciliation
hrec = HierarchicalReconciliation(reconcilers=[
    #these methods should reconstruct the original y
    BottomUp(),
    MinTrace(method='ols'),
    MinTrace(method='wls_struct'),
    MinTrace(method='wls_var'),
    MinTrace(method='mint_shrink'),
])
reconciled = hrec.reconcile(Y_hat_df=hier_grouped_df_h, Y_df=hier_grouped_df,
                            S=S_grouped, tags=tags_grouped)
# Test mse, rmse
evaluator = HierarchicalEvaluation([mse, rmse])
evaluation_old = evaluator.evaluate(Y_hat_df=reconciled.drop(columns='y'),
                   Y_test_df=reconciled[['unique_id', 'ds', 'y']],
                   tags=tags_grouped,
                #    benchmark='y_model',
                   )
# Test mse, rmse
evaluation = evaluate(reconciled,
         metrics=[ufl.mse, ufl.rmse],
         tags=tags_grouped)

pd.testing.assert_frame_equal(evaluation_old, evaluation)
# polars
# Test mse, rmse
reconciled_pl = pl.from_pandas(reconciled)
evaluation_pl = evaluator.evaluate(Y_hat_df=reconciled_pl.drop('y'),
                   Y_test_df=reconciled_pl[['unique_id', 'ds', 'y']],
                   tags=tags_grouped)

pd.testing.assert_frame_equal(evaluation, evaluation_pl.to_pandas())
# Test mase
evaluator = HierarchicalEvaluation([mase])
evaluation_old = evaluator.evaluate(Y_hat_df=reconciled.drop(columns='y'),
                   Y_test_df=reconciled[['unique_id', 'ds', 'y']],
                   tags=tags_grouped,
                   Y_df=hier_grouped_df)
evaluation = evaluate(reconciled,
         metrics=[partial(ufl.mase, seasonality=4)],
         train_df=hier_grouped_df,
         tags=tags_grouped)

pd.testing.assert_frame_equal(evaluation_old, evaluation)
# polars
reconciled_pl = pl.from_pandas(reconciled)
hier_grouped_df_pl = pl.from_pandas(hier_grouped_df)

evaluation_pl = evaluator.evaluate(Y_hat_df=reconciled_pl.drop('y'),
                   Y_test_df=reconciled_pl[['unique_id', 'ds', 'y']],
                   tags=tags_grouped,
                   Y_df=hier_grouped_df_pl)

pd.testing.assert_frame_equal(evaluation, evaluation_pl.to_pandas())
# test work for h=1
evaluator = HierarchicalEvaluation([mase])
evaluation_old = evaluator.evaluate(Y_hat_df=reconciled.groupby('unique_id').tail(1).drop(columns='y'),
                   Y_test_df=reconciled.groupby('unique_id').tail(1)[['unique_id', 'ds', 'y']],
                   tags=tags_grouped,
                   Y_df=hier_grouped_df)
# test work for h=1
evaluation = evaluate(reconciled.groupby('unique_id').tail(1),
         metrics=[partial(ufl.mase, seasonality=4)],
         train_df=hier_grouped_df,
         tags=tags_grouped)

pd.testing.assert_frame_equal(evaluation_old, evaluation)
# polars
# test work for h=1
evaluation_pl = evaluate(
         reconciled_pl.group_by('unique_id').tail(1),
         metrics=[partial(ufl.mase, seasonality=4)],
         train_df=pl.from_pandas(hier_grouped_df),
         tags=tags_grouped)

pd.testing.assert_frame_equal(evaluation_old, evaluation)
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS

# Load TourismSmall dataset
df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/tourism.csv')
df = df.rename({'Trips': 'y', 'Quarter': 'ds'}, axis=1)
df.insert(0, 'Country', 'Australia')
qs = df['ds'].str.replace(r'(\d+) (Q\d)', r'\1-\2', regex=True)
df['ds'] = pd.PeriodIndex(qs, freq='Q').to_timestamp()

# Create hierarchical seires based on geographic levels and purpose
# And Convert quarterly ds string to pd.datetime format
hierarchy_levels = [['Country'],
                    ['Country', 'State'],
                    ['Country', 'Purpose'],
                    ['Country', 'State', 'Region'],
                    ['Country', 'State', 'Purpose'],
                    ['Country', 'State', 'Region', 'Purpose']]

Y_df, S_df, tags = aggregate(df=df, spec=hierarchy_levels)

# Split train/test sets
Y_test_df  = Y_df.groupby('unique_id').tail(8)
Y_train_df = Y_df.drop(Y_test_df.index)

# Compute base auto-ETS predictions
# Careful identifying correct data freq, this data quarterly 'Q'
fcst = StatsForecast(models=[AutoETS(season_length=4, model='ZZA')], freq='QS', n_jobs=-1)
Y_hat_df = fcst.forecast(df=Y_train_df, h=8, fitted=True)
Y_fitted_df = fcst.forecast_fitted_values()

reconcilers = [
                BottomUp(),
                MinTrace(method='ols'),
                MinTrace(method='mint_shrink'),
               ]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df,
                          Y_df=Y_fitted_df,
                          S=S_df, tags=tags)

# Evaluate
eval_tags = {}
eval_tags['Total'] = tags['Country']
eval_tags['Purpose'] = tags['Country/Purpose']
eval_tags['State'] = tags['Country/State']
eval_tags['Regions'] = tags['Country/State/Region']
eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
evaluator = HierarchicalEvaluation(evaluators=[mase, rmse])
evaluation_old = evaluator.evaluate(
        Y_hat_df=Y_rec_df, Y_test_df=Y_test_df,
        tags=eval_tags, Y_df=Y_train_df
)
numeric_cols = evaluation_old.select_dtypes(include="number").columns
evaluation_old[numeric_cols] = evaluation_old[numeric_cols].map('{:.2f}'.format).astype(np.float64)

evaluation_check = pd.DataFrame({
    'level': ['Total', 'Purpose', 'State', 'Regions', 'Bottom', 'Overall'],
    'metric': 6 * ['mase'],
    'AutoETS': [1.59, 1.32, 1.39, 1.12, 0.98, 1.02],
    'AutoETS/BottomUp': [3.16, 2.28, 1.90, 1.19, 0.98, 1.06],
})

pd.testing.assert_frame_equal(evaluation_old.query("metric == 'mase'")[["level", "metric", "AutoETS", "AutoETS/BottomUp"]].reset_index(drop=True), evaluation_check)
Y_rec_df_with_y = Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how='left')
mase_p = partial(ufl.mase, seasonality=4)

evaluation = evaluate(Y_rec_df_with_y,
         metrics=[mase_p, ufl.rmse],
         tags=eval_tags,
         train_df=Y_train_df)

evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.2f}'.format).astype(np.float64)

evaluation_check = pd.DataFrame({
    'level': ['Total', 'Purpose', 'State', 'Regions', 'Bottom', 'Overall'],
    'metric': 6 * ['mase'],
    'AutoETS': [1.59, 1.32, 1.39, 1.12, 0.98, 1.02],
    'AutoETS/BottomUp': [3.16, 2.28, 1.90, 1.19, 0.98, 1.06],
})

pd.testing.assert_frame_equal(evaluation.query("metric == 'mase'")[["level", "metric", "AutoETS", "AutoETS/BottomUp"]].reset_index(drop=True), evaluation_check)

pd.testing.assert_frame_equal(evaluation_old, evaluation)
# polars

# Test if polars gives equal results to pandas
df_pl = pl.from_pandas(df)
Y_df, S_df, tags = aggregate(df=df_pl, spec=hierarchy_levels)

# Split train/test sets
Y_test_df  = Y_df.group_by('unique_id').tail(8)
Y_train_df = Y_df.group_by('unique_id').head(72)

# Compute base auto-ETS predictions
# Careful identifying correct data freq, this data quarterly 'Q'
fcst = StatsForecast(models=[AutoETS(season_length=4, model='ZZA')], freq='1q', n_jobs=-1)
Y_hat_df_pl = fcst.forecast(df=Y_train_df, h=8, fitted=True)
Y_fitted_df = fcst.forecast_fitted_values()

# Reconcile the base predictions
reconcilers = [BottomUp(),
                MinTrace(method='ols'),
               MinTrace(method='mint_shrink')]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df_pl = hrec.reconcile(Y_hat_df=Y_hat_df_pl,
                          Y_df=Y_fitted_df,
                          S=S_df, tags=tags)

# Evaluate
Y_rec_df_pl_with_y = Y_rec_df_pl.join(Y_test_df, on=['unique_id', 'ds'], how='left')
evaluation_pl = evaluate(Y_rec_df_pl_with_y,
         metrics=[mase_p, ufl.rmse],
         tags=eval_tags,
         train_df=Y_train_df)

evaluation_pl = evaluation_pl.with_columns(pl.col(numeric_cols).round(2).cast(pl.Float64))

# Check if polars gives identical results as pandas
pd.testing.assert_frame_equal(Y_hat_df_pl.to_pandas(), Y_hat_df)
pd.testing.assert_frame_equal(Y_rec_df_pl.to_pandas(), Y_rec_df)
pd.testing.assert_frame_equal(evaluation_pl.to_pandas(), evaluation)
