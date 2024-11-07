# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/src/evaluation.ipynb.

# %% auto 0
__all__ = ['rel_mse', 'msse', 'scaled_crps', 'energy_score', 'log_score', 'HierarchicalEvaluation']

# %% ../nbs/src/evaluation.ipynb 3
from inspect import signature
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import utilsforecast.processing as ufp

from .utils import pivot, df_constructor
from utilsforecast.compat import DFType
from scipy.stats import multivariate_normal

# %% ../nbs/src/evaluation.ipynb 6
def _metric_protections(y: np.ndarray, y_hat: np.ndarray, 
                        weights: Optional[np.ndarray]) -> None:
    if not ((weights is None) or (np.sum(weights) > 0)):
        raise Exception('Sum of `weights` cannot be 0')
    if not ((weights is None) or (weights.shape == y.shape)):
        raise Exception(
        f'Wrong weight dimension weights.shape {weights.shape}, y.shape {y.shape}')

def mse(y: np.ndarray, y_hat: np.ndarray, 
        weights: Optional[np.ndarray] = None,
        axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Mean Squared Error

    Calculates Mean Squared Error between
    `y` and `y_hat`. MSE measures the relative prediction
    accuracy of a forecasting method by calculating the 
    squared deviation of the prediction and the true
    value at a given time, and averages these devations
    over the length of the series.

    $$ \mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}) = \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} - \hat{y}_{\\tau})^{2} $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `mse`: numpy array, (single value).
    """
    _metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(delta_y[~np.isnan(delta_y)],
                         weights=weights[~np.isnan(delta_y)],
                         axis=axis)
    else:
        mse = np.nanmean(delta_y, axis=axis)
    return mse

def mqloss(y: np.ndarray, y_hat: np.ndarray, 
           quantiles: np.ndarray, 
           weights: Optional[np.ndarray] = None,
           axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Multi-Quantile Loss

    Calculates the Multi-Quantile loss (MQL) between `y` and `y_hat`.
    MQL calculates the average multi-quantile Loss for
    a given set of quantiles, based on the absolute 
    difference between predicted quantiles and observed values.

    $$ \mathrm{MQL}(\\mathbf{y}_{\\tau},[\\mathbf{\hat{y}}^{(q_{1})}_{\\tau}, ... ,\hat{y}^{(q_{n})}_{\\tau}]) = \\frac{1}{n} \\sum_{q_{i}} \mathrm{QL}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}^{(q_{i})}_{\\tau}) $$

    The limit behavior of MQL allows to measure the accuracy 
    of a full predictive distribution $\mathbf{\hat{F}}_{\\tau}$ with 
    the continuous ranked probability score (CRPS). This can be achieved 
    through a numerical integration technique, that discretizes the quantiles 
    and treats the CRPS integral with a left Riemann approximation, averaging over 
    uniformly distanced quantiles.    

    $$ \mathrm{CRPS}(y_{\\tau}, \mathbf{\hat{F}}_{\\tau}) = \int^{1}_{0} \mathrm{QL}(y_{\\tau}, \hat{y}^{(q)}_{\\tau}) dq $$

    **Parameters:**<br>
    `y`: numpy array, Actual values.<br>
    `y_hat`: numpy array, Predicted values.<br>
    `quantiles`: numpy array. Quantiles between 0 and 1, to perform evaluation upon size (n_quantiles).<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>
 
    **Returns:**<br>
    `mqloss`: numpy array, (single value).

    **References:**<br>
    [Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles".](https://www.jstor.org/stable/1913643)<br>
    [James E. Matheson and Robert L. Winkler, "Scoring Rules for Continuous Probability Distributions".](https://www.jstor.org/stable/2629907)
    """
    if weights is None: weights = np.ones(y.shape)
    if (np.sum(quantiles>1)>0 or np.sum(quantiles<0)>0):
        raise Exception('`quantiles` need to be between 0 and 1')
        
    _metric_protections(y, y_hat, weights)
    n_q = len(quantiles)
    
    y_rep  = np.expand_dims(y, axis=-1)
    error  = y_hat - y_rep
    sq     = np.maximum(-error, np.zeros_like(error))
    s1_q   = np.maximum(error, np.zeros_like(error))
    mqloss = (quantiles * sq + (1 - quantiles) * s1_q)
    
    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss  = np.average(mqloss, weights=weights, axis=axis)

    return mqloss

# %% ../nbs/src/evaluation.ipynb 8
def rel_mse(y, y_hat, y_train, mask=None):
    """Relative Mean Squared Error

    Computes Relative mean squared error (RelMSE), as proposed by Hyndman & Koehler (2006)
    as an alternative to percentage errors, to avoid measure unstability.

    $$
    \mathrm{RelMSE}(\\mathbf{y}, \\mathbf{\hat{y}}, \\mathbf{\hat{y}}^{naive1}) =
    \\frac{\mathrm{MSE}(\\mathbf{y}, \\mathbf{\hat{y}})}{\mathrm{MSE}(\\mathbf{y}, \\mathbf{\hat{y}}^{naive1})}
    $$

    **Parameters:**<br>
    `y`: numpy array, Actual values of size (`n_series`, `horizon`).<br>
    `y_hat`: numpy array, Predicted values (`n_series`, `horizon`).<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `loss`: float.    

    **References:**<br>
    - [Hyndman, R. J and Koehler, A. B. (2006).
       "Another look at measures of forecast accuracy",
       International Journal of Forecasting, Volume 22, Issue 4.](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
    - [Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao, Lee Dicker. 
       "Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures. 
       Submitted to the International Journal Forecasting, Working paper available at arxiv.](https://arxiv.org/pdf/2110.13179.pdf)
    """
    if mask is None: 
       mask = np.ones_like(y)
    n_series, horizon = y.shape

    eps = np.finfo(float).eps
    y_naive = np.repeat(y_train[:,[-1]], horizon, axis=1)
    norm = mse(y=y, y_hat=y_naive)
    loss = mse(y=y, y_hat=y_hat, weights=mask)
    loss = loss / (norm + eps)
    return loss

# %% ../nbs/src/evaluation.ipynb 11
def msse(y, y_hat, y_train, mask=None):
    """Mean Squared Scaled Error

    Computes Mean squared scaled error (MSSE), as proposed by Hyndman & Koehler (2006)
    as an alternative to percentage errors, to avoid measure unstability.

    $$
    \\mathrm{MSSE}(\\mathbf{y}, \\mathbf{\\hat{y}}, \\mathbf{y}^{in-sample}) =
    \\frac{\\frac{1}{h} \\sum^{t+h}_{\\tau=t+1} (y_{\\tau} - \\hat{y}_{\\tau})^2}{\\frac{1}{t-1} \\sum^{t}_{\\tau=2} (y_{\\tau} - y_{\\tau-1})^2}
    $$

    where $n$ ($n=$`n`) is the size of the training data, and $h$ is the forecasting horizon ($h=$`horizon`).

    **Parameters:**<br>
    `y`: numpy array, Actual values of size (`n_series`, `horizon`).<br>
    `y_hat`: numpy array, Predicted values (`n_series`, `horizon`).<br>
    `y_train`: numpy array, Predicted values (`n_series`, `n`).<br>
    `mask`: numpy array, Specifies date stamps per serie to consider in loss.<br>

    **Returns:**<br>
    `loss`: float.    

    **References:**<br>
    - [Hyndman, R. J and Koehler, A. B. (2006).
       "Another look at measures of forecast accuracy",
       International Journal of Forecasting, Volume 22, Issue 4.](https://www.sciencedirect.com/science/article/pii/S0169207006000239)<br>
    """
    if mask is None: 
       mask = np.ones_like(y)
    n_series, horizon = y.shape

    eps = np.finfo(float).eps
    y_in_sample_naive = y_train[:, :-1]
    y_in_sample_true = y_train[:, 1:]
    norm = mse(y=y_in_sample_true, y_hat=y_in_sample_naive)
    loss = mse(y=y, y_hat=y_hat, weights=mask)
    loss = loss / (norm + eps)
    return loss

# %% ../nbs/src/evaluation.ipynb 14
def scaled_crps(y, y_hat, quantiles):
    """Scaled Continues Ranked Probability Score

    Calculates a scaled variation of the CRPS, as proposed by Rangapuram (2021),
    to measure the accuracy of predicted quantiles `y_hat` compared to the observation `y`.

    This metric averages percentual weighted absolute deviations as 
    defined by the quantile losses.

    $$
    \mathrm{sCRPS}(\hat{F}_{\\tau}, \mathbf{y}_{\\tau}) = \\frac{2}{N} \sum_{i}
    \int^{1}_{0}
    \\frac{\mathrm{QL}(\hat{F}_{i,\\tau}, y_{i,\\tau})_{q}}{\sum_{i} | y_{i,\\tau} |} dq
    $$

    where $\hat{F}_{\\tau}$ is the an estimated multivariate distribution, and $y_{i,\\tau}$
    are its realizations.

    **Parameters:**<br>
    `y`: numpy array, Actual values of size (`n_series`, `horizon`).<br>
    `y_hat`: numpy array, Predicted quantiles of size (`n_series`, `horizon`, `n_quantiles`).<br>
    `quantiles`: numpy array,(`n_quantiles`). Quantiles to estimate from the distribution of y.<br>

    **Returns:**<br>
    `loss`: float.

    **References:**<br>
    - [Gneiting, Tilmann. (2011). \"Quantiles as optimal point forecasts\". 
    International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207010000063)<br>
    - [Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos, Zhi Chen, Anil Gaba, Ilia Tsetlin, Robert L. Winkler. (2022). 
    \"The M5 uncertainty competition: Results, findings and conclusions\". 
    International Journal of Forecasting.](https://www.sciencedirect.com/science/article/pii/S0169207021001722)<br>
    - [Syama Sundar Rangapuram, Lucien D Werner, Konstantinos Benidis, Pedro Mercado, Jan Gasthaus, Tim Januschowski. (2021). 
    \"End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series\". 
    Proceedings of the 38th International Conference on Machine Learning (ICML).](https://proceedings.mlr.press/v139/rangapuram21a.html)
    """
    eps = np.finfo(float).eps
    norm  = np.sum(np.abs(y))
    loss  = mqloss(y=y, y_hat=y_hat, quantiles=quantiles)
    loss  = 2 * loss * np.sum(np.ones(y.shape)) / (norm + eps)
    return loss

# %% ../nbs/src/evaluation.ipynb 17
def energy_score(y, y_sample1, y_sample2, beta=2):
    """Energy Score

    Calculates Gneiting's Energy Score sample approximation for
    `y` and independent multivariate samples `y_sample1` and `y_sample2`.
    The Energy Score generalizes the CRPS (`beta`=1) in the multivariate setting.

    $$
    \mathrm{ES}(\\mathbf{y}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}') 
    = \\frac{1}{2} \mathbb{E}_{\hat{P}} \\left[ ||\\mathbf{\hat{y}}_{\\tau} - \\mathbf{\hat{y}}_{\\tau}'||^{\\beta} \\right]
    -  \mathbb{E}_{\hat{P}} \\left[ ||\\mathbf{y}_{\\tau} - \\mathbf{\hat{y}}_{\\tau}||^{\\beta} \\right] 
    \quad \\beta \in (0,2]
    $$

    where $\\mathbf{\hat{y}}_{\\tau}, \\mathbf{\hat{y}}_{\\tau}'$ are independent samples drawn from $\hat{P}$.

    **Parameters:**<br>
    `y`: numpy array, Actual values of size (`n_series`, `horizon`).<br>
    `y_sample1`: numpy array, predictive distribution sample of size (`n_series`, `horizon`, `n_samples`).<br>
    `y_sample2`: numpy array, predictive distribution sample of size (`n_series`, `horizon`, `n_samples`).<br>
    `beta`: float in (0,2], defines the energy score's power for the euclidean metric.<br>

    **Returns:**<br>
    `score`: float.

    **References:**<br>
    - [Gneiting, Tilmann, and Adrian E. Raftery. (2007). 
    \"Strictly proper scoring rules, prediction and estimation\". 
    Journal of the American Statistical Association.](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)<br>
    - [Anastasios Panagiotelis, Puwasala Gamakumara, George Athanasopoulos, Rob J. Hyndman. (2022). 
    \"Probabilistic forecast reconciliation: Properties, evaluation and score optimisation\". 
    European Journal of Operational Research.](https://www.sciencedirect.com/science/article/pii/S0377221722006087)    
    """
    if beta>2 or beta<0:
        raise Exception("beta needs to be between 0 and 2.")

    dif1 = (y_sample1 - y_sample2)
    dif2 = (y[:,:,None] - y_sample1)

    term1 = np.linalg.norm(dif1, axis=0) ** beta
    term2 = np.linalg.norm(dif2, axis=0) ** beta

    score = np.mean(term2 - 0.5 * term1)
    return score

# %% ../nbs/src/evaluation.ipynb 19
def log_score(y, y_hat, cov, allow_singular=True):
    """ Log Score.

    One of the simplest multivariate probability scoring rules,
    it evaluates the negative density at the value of the realisation.

    $$
    \mathrm{LS}(\\mathbf{y}_{\\tau}, \\mathbf{P}(\\theta_{\\tau}))
    = - \\log(f(\\mathbf{y}_{\\tau}, \\theta_{\\tau}))
    $$

    where $f$ is the density, $\\mathbf{P}(\\theta_{\\tau})$ is a 
    parametric distribution and $f(\\mathbf{y}_{\\tau}, \\theta_{\\tau})$
    represents its density. 
    For the moment we only support multivariate normal log score.

    $$
    f(\\mathbf{y}_{\\tau}, \\theta_{\\tau}) =
    (2\\pi )^{-k/2}\\det({\\boldsymbol{\Sigma }})^{-1/2}
    \,\\exp \\left(
    -{\\frac {1}{2}}(\mathbf{y}_{\\tau} -\\hat{\mathbf{y}}_{\\tau})^{\!{\mathsf{T}}}
    {\\boldsymbol{\Sigma }}^{-1}
    (\mathbf{y}_{\\tau} -\\hat{\mathbf{y}}_{\\tau})
    \\right)
    $$

    **Parameters:**<br>
    `y`: numpy array, Actual values of size (`n_series`, `horizon`).<br>
    `y_hat`: numpy array, Predicted values (`n_series`, `horizon`).<br>
    `cov`: numpy matrix, Predicted values covariance (`n_series`, `n_series`, `horizon`).<br>
    `allow_singular`: bool=True, if true allows singular covariance.<br>

    **Returns:**<br>
    `score`: float.
    """
    scores = [multivariate_normal.pdf(x=y[:,h], 
                  mean=y_hat[:,h], cov=cov[:,:,h], allow_singular=allow_singular) \
                  for h in range(y.shape[1])]
    score = np.mean(scores)
    return score

# %% ../nbs/src/evaluation.ipynb 23
class HierarchicalEvaluation:
    """Hierarchical Evaluation Class.
    
    You can use your own metrics to evaluate the performance of each level in the structure.
    The metrics receive `y` and `y_hat` as arguments and they are numpy arrays of size `(series, horizon)`.
    Consider, for example, the function `rmse` that calculates the root mean squared error.

    This class facilitates measurements across the hierarchy, defined by the `tags` list.
    See also the [aggregate method](https://nixtla.github.io/hierarchicalforecast/utils.html#aggregate).

    **Parameters:**<br>
    `evaluators`: functions with arguments `y`, `y_hat` (numpy arrays).<br>

    **References:**<br>
    """
    def __init__(self, 
                 evaluators: List[Callable]):
        self.evaluators = evaluators

    def evaluate(self, 
                 Y_hat_df: DFType,
                 Y_test_df: DFType,
                 tags: Dict[str, np.ndarray],
                 Y_df: Optional[DFType] = None,
                 benchmark: Optional[str] = None,
                 id_col: str = "unique_id",
                 time_col: str = "ds", 
                 target_col: str = "y", 
                 ):
        """Hierarchical Evaluation Method.

        **Parameters:**<br>
        `Y_hat_df`: DataFrame, Forecasts indexed by `'unique_id'` with column `'ds'` and models to evaluate.<br>
        `Y_test_df`:  DataFrame, True values with columns `['ds', 'y']`.<br>
        `tags`: np.array, each str key is a level and its value contains tags associated to that level.<br>
        `Y_df`: DataFrame, Training set of base time series with columns `['ds', 'y']` indexed by `unique_id`.<br>
        `benchmark`: str, If passed, evaluators are scaled by the error of this benchark.<br>
        `id_col` : str='unique_id', column that identifies each serie.<br>
        `time_col` : str='ds', column that identifies each timestep, its values can be timestamps or integers.<br>
        `target_col` : str='y', column that contains the target.

        **Returns:**<br>
        `evaluation`: DataFrame with accuracy measurements across hierarchical levels.
        """
        n_series = len(set(Y_hat_df[id_col]))
        h = len(set(Y_hat_df[time_col]))
        if len(Y_hat_df) != n_series * h:
            raise Exception('Y_hat_df should have a forecast for each series and horizon')

        fn_names = [fn.__name__ for fn in self.evaluators]
        has_y_insample = any(['y_insample' in signature(fn).parameters for fn in self.evaluators])
        if has_y_insample and Y_df is None:
            raise Exception('At least one evaluator needs y_insample, please pass `Y_df`')

        if benchmark is not None:
            fn_names = [f'{fn_name}-scaled' for fn_name in fn_names]

        tags_ = {'Overall': np.concatenate(list(tags.values()))}
        tags_ = {**tags_, **tags}

        model_names = list(set(Y_hat_df.columns) - set([time_col, target_col, id_col]))
        evaluation_np = np.empty((len(tags_), len(fn_names), len(model_names)), dtype=np.float64)
        evaluation_index_np = np.empty((len(tags_) * len(fn_names), 2), dtype=object)
        for i_level, (level, cats) in enumerate(tags_.items()):
            mask = ufp.is_in(Y_hat_df[id_col], cats)
            Y_h_cats = ufp.filter_with_mask(Y_hat_df, mask)

            mask = ufp.is_in(Y_test_df[id_col], cats)
            y_test_cats = ufp.filter_with_mask(Y_test_df, mask)[target_col]\
                             .to_numpy()\
                             .reshape(-1, h)

            if has_y_insample and Y_df is not None:
                y_insample = pivot(Y_df, index = id_col, columns = time_col, values = target_col)
                mask = ufp.is_in(y_insample[id_col], cats)
                y_insample = ufp.filter_with_mask(y_insample, mask)
                y_insample = ufp.drop_columns(y_insample, id_col)
                y_insample = y_insample.to_numpy()

            for i_fn, fn in enumerate(self.evaluators):
                if 'y_insample' in signature(fn).parameters:
                    kwargs = {'y_insample': y_insample}
                else:
                    kwargs = {}
                fn_name = fn_names[i_fn]
                for i_model, model in enumerate(model_names):
                    loss = fn(y_test_cats, Y_h_cats[model].to_numpy().reshape(-1, h), **kwargs)
                    if benchmark is not None:
                        scale = fn(y_test_cats, Y_h_cats[benchmark].to_numpy().reshape(-1, h), **kwargs)
                        if np.isclose(scale, 0., atol=np.finfo(float).eps):
                            scale += np.finfo(float).eps
                            if np.isclose(scale, loss, atol=1e-8):
                                scale = 1.
                        loss /= scale

                    evaluation_np[i_level, i_fn, i_model] = loss
                    evaluation_index_np[i_level * len(fn_names) + i_fn, 0] = level
                    evaluation_index_np[i_level * len(fn_names) + i_fn, 1] = fn_name

        evaluation_np = evaluation_np.reshape(-1, len(model_names))
        evaluation = df_constructor(dftype=type(Y_hat_df), 
                                    X=evaluation_index_np, 
                                    columns=["level", "metric"])
        evaluation = ufp.assign_columns(evaluation, model_names, evaluation_np)

        return evaluation
