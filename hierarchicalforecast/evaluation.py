# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/evaluation.ipynb (unless otherwise specified).

__all__ = ['HierarchicalEvaluation']

# Cell
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Cell
class HierarchicalEvaluation:

    def __init__(self, evaluators: List[Callable]):
        self.evaluators = evaluators

    def evaluate(self,
                 Y_h: pd.DataFrame,
                 Y_test: pd.DataFrame,
                 tags: Dict[str, np.ndarray],
                 benchmark: Optional[str] = None):
        """Evaluate hierarchical forecasts.

            Parameters
            ----------
            Y_h: pd.DataFrame
                Forecasts with columns ['ds']
                and models to evaluate.
            Y_test: pd.DataFrame
                True values with columns ['ds', 'y']
            tags: Dict[str, np.ndarray]
                Dictionary of levels.
                Each key is a level and its value
                contains tags associated to that level.
            benchmark: Optional[str]
                Optional benchmark model.
                When passed, the evaluators are scaled by
                the error of this benchark.
                If passed, should be part of `Y_h`.
        """
        drop_cols = ['ds', 'y'] if 'y' in Y_h.columns else ['ds']
        model_names = Y_h.drop(columns=drop_cols, axis=1).columns.to_list()
        fn_names = [fn.__name__ for fn in self.evaluators]
        if benchmark is not None:
            fn_names = [f'{fn_name}-scaled' for fn_name in fn_names]
        tags_ = {'Overall': np.concatenate(list(tags.values()))}
        tags_ = {**tags_, **tags}
        index = pd.MultiIndex.from_product([tags_.keys(), fn_names], names=['level', 'metric'])
        evaluation = pd.DataFrame(columns=model_names, index=index)
        for level, cats in tags_.items():
            Y_h_cats = Y_h.loc[cats]
            y_test_cats = Y_test.loc[cats, 'y'].values
            for i_fn, fn in enumerate(self.evaluators):
                fn_name = fn_names[i_fn]
                for model in model_names:
                    loss = fn(y_test_cats, Y_h_cats[model].values)
                    if benchmark is not None:
                        scale = fn(y_test_cats, Y_h_cats[benchmark].values)
                        if np.isclose(scale, 0., atol=np.finfo(float).eps):
                            scale += np.finfo(float).eps
                            if np.isclose(scale, loss, atol=1e-8):
                                scale = 1.
                        loss /= scale
                    evaluation.loc[(level, fn_name), model] = loss
        return evaluation