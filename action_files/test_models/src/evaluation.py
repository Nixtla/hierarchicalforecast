import fire
import pickle
import numpy as np
import pandas as pd

import hierarchicalforecast.evaluation as hfe
from utilsforecast.losses import rmse, mase,scaled_crps
from functools import partial
from typing import Optional

def eval(level: Optional[list[int]] = None) -> pd.DataFrame:
    execution_times = pd.read_csv('data/execution_times.csv')
    models = [f"{x[0]} ({x[1]:.2f} secs)" for x in execution_times.values]

    Y_rec_df = pd.read_csv('data/Y_rec.csv')
    Y_test_df = pd.read_csv('data/Y_test.csv')
    Y_train_df = pd.read_csv('data/Y_train.csv')

    with open('data/tags.pickle', 'rb') as handle:
        tags = pickle.load(handle)

    mase_p = partial(mase, seasonality=4)
    evaluation = hfe.evaluate(
            df=Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how="left"),
            metrics = [rmse, mase_p, scaled_crps],
            level=level,
            tags=tags, 
            train_df=Y_train_df
    )
    numeric_cols = evaluation.select_dtypes(include="number").columns
    evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.3}'.format).astype(np.float64)
    evaluation.columns = ['level', 'metric', 'Base'] + models
    print(evaluation.T)
    evaluation.to_csv('./data/evaluation.csv')

if __name__ == '__main__':
    fire.Fire(eval)