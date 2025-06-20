import pickle
import numpy as np
import pandas as pd

import hierarchicalforecast.evaluation as hfe
from utilsforecast.losses import rmse, mase
from functools import partial

def eval():
    execution_times = pd.read_csv('data/execution_times.csv')
    models = [f"{x[0]} ({x[1]:.2f} secs)" for x in execution_times.values]

    Y_rec_df = pd.read_csv('data/Y_rec.csv')
    Y_test_df = pd.read_csv('data/Y_test.csv')
    Y_train_df = pd.read_csv('data/Y_train.csv')

    with open('data/tags.pickle', 'rb') as handle:
        tags = pickle.load(handle)

    eval_tags = {}
    eval_tags['Total'] = tags['Country']
    eval_tags['State'] = tags['Country/State']
    eval_tags['Regions'] = tags['Country/State/Region']
    eval_tags['Bottom'] = tags['Country/State/Region/Purpose']
    eval_tags['All'] = np.concatenate(list(tags.values()))

    mase_p = partial(mase, seasonality=4)
    evaluation = hfe.evaluate(
            df=Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how="left"),
            metrics = [rmse, mase_p],
            tags=eval_tags, 
            train_df=Y_train_df
    )
    numeric_cols = evaluation.select_dtypes(include="number").columns
    evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.3}'.format).astype(np.float64)
    evaluation.columns = ['level', 'metric', 'Base'] + models
    return evaluation



if __name__ == '__main__':
    evaluation = eval()
    evaluation.to_csv('./data/evaluation.csv')
    print(evaluation.T)
