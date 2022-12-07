# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/methods.ipynb.

# %% auto 0
__all__ = ['BottomUp', 'TopDown', 'MiddleOut', 'MinTrace', 'OptimalCombination', 'ERM']

# %% ../nbs/methods.ipynb 3
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import numpy as np
from numba import njit
from quadprog import solve_qp

# %% ../nbs/methods.ipynb 5
def _reconcile(S: np.ndarray,
               P: np.ndarray,
               W: np.ndarray,
               y_hat: np.ndarray,
               SP: np.ndarray = None,
               level: Optional[List[int]] = None,
               sampler: Optional[Callable] = None):

    # Mean reconciliation
    res = {'mean': np.matmul(S @ P, y_hat)}
    sampler_name = type(sampler).__name__

    # Probabilistic reconciliation
    # TODO: instantiate the samplers after mean reconciliation.
    # separate functions, and add _prob_reconcile -> probabilistic_methods.py
    # both Normality and Bootstrap depend reconciled outputs (P, W).
    # I suggest to do it in `core.HierarchicalForecast.reconcile`
    # after this call `fcsts_model = reconcile_fn(y_hat=y_hat_model, **kwargs)`

    if level is not None and \
        sampler_name in ['Normality', 'Bootstrap', 'PERMBU']:

        if sampler_name == 'Normality':
            res = sampler.get_prediction_levels(P=P, W=W,
                                                res=res, level=level)

        if sampler_name == 'Bootstrap':
            res = sampler.get_prediction_levels(P=P,
                                                res=res, level=level)

        if sampler_name == 'PERMBU':
            res = sampler.get_prediction_levels(res=res, level=level)

    return res

# %% ../nbs/methods.ipynb 7
class BottomUp:
    """Bottom Up Reconciliation Class.
    The most basic hierarchical reconciliation is performed using an Bottom-Up strategy. It was proposed for 
    the first time by Orcutt in 1968.
    The corresponding hierarchical \"projection\" matrix is defined as:
    $$\mathbf{P}_{\\text{BU}} = [\mathbf{0}_{\mathrm{[b],[a]}}\;|\;\mathbf{I}_{\mathrm{[b][b]}}]$$

    **Parameters:**<br>
    None

    **References:**<br>
    - [Orcutt, G.H., Watts, H.W., & Edwards, J.B.(1968). \"Data aggregation and information loss\". The American 
    Economic Review, 58 , 773{787)](http://www.jstor.org/stable/1815532).
    """
    insample = False

    def _get_PW_matrices(self, S, idx_bottom):
        n_hiers, n_bottom = S.shape
        P = np.zeros_like(S, dtype=np.float32)
        P[idx_bottom] = S[idx_bottom]
        P = P.T
        W = np.eye(n_hiers, dtype=np.float32)
        return P, W

    def reconcile(self,
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  idx_bottom: np.ndarray,
                  level: Optional[List[int]] = None,
                  sampler: Optional[Callable] = None):
        """Bottom Up Reconciliation Method.

        **Parameters:**<br>
        `S`: Summing matrix of size (`base`, `bottom`).<br>
        `y_hat`: Forecast values of size (`base`, `horizon`).<br>
        `idx_bottom`: Indices corresponding to the bottom level of `S`, size (`bottom`).<br>
        `level`: float list 0-100, confidence levels for prediction intervals.<br>
        `sampler`: Sampler for prediction intevals, one of Normality(), Bootstrap(), PERMBU().<br>

        **Returns:**<br>
        `y_tilde`: Reconciliated y_hat using the Bottom Up approach.
        """
        self.P, self.W = self._get_PW_matrices(S=S, idx_bottom=idx_bottom)
        return _reconcile(S=S, P=self.P, W=self.W, y_hat=y_hat,
                          level=level, sampler=sampler)

    __call__ = reconcile

# %% ../nbs/methods.ipynb 16
def is_strictly_hierarchical(S: np.ndarray, 
                             tags: Dict[str, np.ndarray]):
    # main idea:
    # if S represents a strictly hierarchical structure
    # the number of paths before the bottom level
    # should be equal to the number of nodes
    # of the previuos level
    levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
    # removing bottom level
    levels_.popitem()
    # making S categorical
    hiers = [np.argmax(S[idx], axis=0) + 1 for _, idx in levels_.items()]
    hiers = np.vstack(hiers)
    paths = np.unique(hiers, axis=1).shape[1] 
    nodes = levels_.popitem()[1].size
    return paths == nodes

# %% ../nbs/methods.ipynb 18
def _get_child_nodes(S: np.ndarray, tags: Dict[str, np.ndarray]):
    level_names = list(tags.keys())
    nodes = OrderedDict()
    for i_level, level in enumerate(level_names[:-1]):
        parent = tags[level]
        child = np.zeros_like(S)
        idx_child = tags[level_names[i_level+1]] 
        child[idx_child] = S[idx_child]
        nodes_level = {}
        for idx_parent_node in parent:
            parent_node = S[idx_parent_node]
            idx_node = child * parent_node.astype(bool)
            idx_node, = np.where(idx_node.sum(axis=1) > 0)
            nodes_level[idx_parent_node] = [idx for idx in idx_child if idx in idx_node]
        nodes[level] = nodes_level
    return nodes        

# %% ../nbs/methods.ipynb 20
def _reconcile_fcst_proportions(S: np.ndarray, y_hat: np.ndarray,
                                tags: Dict[str, np.ndarray],
                                nodes: Dict[str, Dict[int, np.ndarray]],
                                idx_top: int):
    reconciled = np.zeros_like(y_hat)
    reconciled[idx_top] = y_hat[idx_top]
    level_names = list(tags.keys())
    for i_level, level in enumerate(level_names[:-1]):
        nodes_level = nodes[level]
        for idx_parent, idx_childs in nodes_level.items():
            fcst_parent = reconciled[idx_parent]
            childs_sum = y_hat[idx_childs].sum()
            for idx_child in idx_childs:
                reconciled[idx_child] = y_hat[idx_child] * fcst_parent / childs_sum
    return reconciled

# %% ../nbs/methods.ipynb 21
class TopDown:
    """Top Down Reconciliation Class.

    The Top Down hierarchical reconciliation method, distributes the total aggregate predictions and decomposes 
    it down the hierarchy using proportions $\mathbf{p}_{\mathrm{[b]}}$ that can be actual historical values 
    or estimated.

    $$\mathbf{P}=[\mathbf{p}_{\mathrm{[b]}}\;|\;\mathbf{0}_{\mathrm{[b][a,b\;-1]}}]$$
    **Parameters:**<br>
    `method`: One of `forecast_proportions`, `average_proportions` and `proportion_averages`.<br>

    **References:**<br>
    - [CW. Gross (1990). \"Disaggregation methods to expedite product line forecasting\". Journal of Forecasting, 9 , 233–254. 
    doi:10.1002/for.3980090304](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3980090304).<br>
    - [G. Fliedner (1999). \"An investigation of aggregate variable time series forecast strategies with specific subaggregate 
    time series statistical correlation\". Computers and Operations Research, 26 , 1133–1149. 
    doi:10.1016/S0305-0548(99)00017-9](https://doi.org/10.1016/S0305-0548(99)00017-9).
    """
    def __init__(self, 
                 method: str):
        self.method = method
        self.insample = method in ['average_proportions', 'proportion_averages']

    def _get_PW_matrices(self,
                         S: np.ndarray,
                         y_hat: np.ndarray,
                         tags: Dict[str, np.ndarray],
                         y_insample: Optional[np.ndarray] = None):
        if not is_strictly_hierarchical(S, tags):
            raise ValueError('Top down reconciliation requires strictly hierarchical structures.')

        n_hiers, n_bottom = S.shape
        idx_top = int(S.sum(axis=1).argmax())
        levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
        idx_bottom = levels_[list(levels_)[-1]]

        y_top = y_insample[idx_top]
        y_btm = y_insample[idx_bottom]
        if self.method == 'average_proportions':
            prop = np.mean(y_btm / y_top, axis=1)
        elif self.method == 'proportion_averages':
            prop = np.mean(y_btm, axis=1) / np.mean(y_top)
        else:
            raise Exception(f'Unknown method {self.method}')

        P = np.zeros_like(S, np.float64).T #float 64 if prop is too small, happens with wiki2
        P[:, idx_top] = prop
        W = np.eye(n_hiers, dtype=np.float32)
        return P, W
    
    def reconcile(self, 
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  tags: Dict[str, np.ndarray],
                  y_insample: Optional[np.ndarray] = None,
                  level: Optional[List[int]] = None,
                  sampler: Optional[np.ndarray] = None):
        """Top Down Reconciliation Method.

        **Parameters:**<br>
        `S`: Summing matrix of size (`base`, `bottom`).<br>
        `y_hat`: Forecast values of size (`base`, `horizon`).<br>
        `tags`: Each key is a level and each value its `S` indices.<br>
        `y_insample`: Insample values of size (`base`, `insample_size`). Optional for `forecast_proportions` method.<br>
        `idx_bottom`: Indices corresponding to the bottom level of `S`, size (`bottom`).<br>
        `level`: float list 0-100, confidence levels for prediction intervals.<br>
        `sampler`: Sampler for prediction intevals, one of Normality(), Bootstrap(), PERMBU().<br>

        **Returns:**<br>
        `y_tilde`: Reconciliated y_hat using the Top Down approach.
        """
        if self.method == 'forecast_proportions':
            idx_top = int(S.sum(axis=1).argmax())
            levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
            if level is not None:
                warnings.warn('Prediction intervals not implement for `forecast_proportions`')
            nodes = _get_child_nodes(S=S, tags=levels_)
            reconciled = [_reconcile_fcst_proportions(S=S, y_hat=y_hat_[:, None], 
                                                      tags=levels_, 
                                                      nodes=nodes,
                                                      idx_top=idx_top) \
                          for y_hat_ in y_hat.T]
            reconciled = np.hstack(reconciled)
            return {'mean': reconciled}
        else:
            self.P, self.W = self._get_PW_matrices(S=S, y_hat=y_hat, 
                                                   tags=tags, y_insample=y_insample)
            return _reconcile(S=S, P=self.P, W=self.W, y_hat=y_hat,
                            level=level,sampler=sampler)
    __call__ = reconcile

# %% ../nbs/methods.ipynb 27
class MiddleOut:
    """Middle Out Reconciliation Class.

    This method is only available for **strictly hierarchical structures**. It anchors the base predictions 
    in a middle level. The levels above the base predictions use the Bottom-Up approach, while the levels 
    below use a Top-Down.

    **Parameters:**<br>
    `middle_level`: Middle level.<br>
    `top_down_method`: One of `forecast_proportions`, `average_proportions` and `proportion_averages`.<br>

    **References:**<br>
    - [Hyndman, R.J., & Athanasopoulos, G. (2021). \"Forecasting: principles and practice, 3rd edition:
    Chapter 11: Forecasting hierarchical and grouped series.\". OTexts: Melbourne, Australia. OTexts.com/fpp3 
    Accessed on July 2022.](https://otexts.com/fpp3/hierarchical.html)

    """
    def __init__(self, 
                 middle_level: str,
                 top_down_method: str):
        self.middle_level = middle_level
        self.top_down_method = top_down_method 
        self.insample = top_down_method in ['average_proportions', 'proportion_averages']

    def _get_PW_matrices(self):
        raise Exception('Not implemented')

    def reconcile(self, 
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  tags: Dict[str, np.ndarray],
                  y_insample: Optional[np.ndarray] = None):
        """Middle Out Reconciliation Method.

        **Parameters:**<br>
        `S`: Summing matrix of size (`base`, `bottom`).<br>
        `y_hat`: Forecast values of size (`base`, `horizon`).<br>
        `tags`: Each key is a level and each value its `S` indices.<br>
        `y_insample`: Insample values of size (`base`, `insample_size`). Only used for `forecast_proportions`<br>

        **Returns:**<br>
        `y_tilde`: Reconciliated y_hat using the Middle Out approach.
        """
        if not is_strictly_hierarchical(S, tags):
            raise ValueError('Middle out reconciliation requires strictly hierarchical structures.')
        if self.middle_level not in tags.keys():
            raise ValueError('You have to provide a `middle_level` in `tags`.')
        levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
        reconciled = np.full_like(y_hat, fill_value=np.nan)
        cut_nodes = levels_[self.middle_level]
        # bottom up reconciliation
        idxs_bu = []
        for node, idx_node in levels_.items():
            idxs_bu.append(idx_node)
            if node == self.middle_level:
                break
        idxs_bu = np.hstack(idxs_bu)
        #bottom up forecasts
        bu = BottomUp().reconcile(
            S=np.unique(S[idxs_bu], axis=1), 
            y_hat=y_hat[idxs_bu], 
            idx_bottom=np.arange(len(idxs_bu))[-len(cut_nodes):]
        )
        reconciled[idxs_bu] = bu['mean']

        #top down
        child_nodes = _get_child_nodes(S, levels_)
        # parents contains each node in the middle out level
        # as key. The values of each node are the levels that
        # are conected to that node.
        parents = {node: {self.middle_level: np.array([node])} for node in cut_nodes}
        level_names = list(levels_.keys())
        for lv, lv_child in zip(level_names[:-1], level_names[1:]):
            # if lv is not part of the middle out to bottom
            # structure we continue
            if lv not in list(parents.values())[0].keys():
                continue
            for idx_middle_out in parents.keys():
                idxs_parents = parents[idx_middle_out].values()
                complete_idxs_child = []
                for idx_parent, idxs_child in child_nodes[lv].items():
                    if any(idx_parent in val for val in idxs_parents):
                        complete_idxs_child.append(idxs_child)
                parents[idx_middle_out][lv_child] = np.hstack(complete_idxs_child)

        for node, levels_node in parents.items():
            idxs_node = np.hstack(list(levels_node.values()))
            S_node = S[idxs_node]
            S_node = S_node[:,~np.all(S_node == 0, axis=0)]
            counter = 0
            levels_node_ = deepcopy(levels_node)
            for lv_name, idxs_level in levels_node_.items():
                idxs_len = len(idxs_level)
                levels_node_[lv_name] = np.arange(counter, idxs_len + counter)
                counter += idxs_len
            td = TopDown(self.top_down_method).reconcile(
                S=S_node, 
                y_hat=y_hat[idxs_node], 
                y_insample=y_insample[idxs_node] if y_insample is not None else None, 
                tags=levels_node_, 
            )
            reconciled[idxs_node] = td['mean']
        return {'mean': reconciled}
    __call__ = reconcile

# %% ../nbs/methods.ipynb 32
def crossprod(x):
    return x.T @ x

# %% ../nbs/methods.ipynb 33
def cov2corr(cov, return_std=False):
    """ convert covariance matrix to correlation matrix

    **Parameters:**<br>
    `cov`: array_like, 2d covariance matrix.<br>
    `return_std`: bool=False, if True returned std.<br>

    **Returns:**<br>
    `corr`: ndarray (subclass) correlation matrix
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr

# %% ../nbs/methods.ipynb 34
class MinTrace:
    """MinTrace Reconciliation Class.

    This reconciliation algorithm proposed by Wickramasuriya et al. depends on a generalized least squares estimator 
    and an estimator of the covariance matrix of the coherency errors $\mathbf{W}_{h}$. The Min Trace algorithm 
    minimizes the squared errors for the coherent forecasts under an unbiasedness assumption; the solution has a 
    closed form.<br>

    $$\mathbf{P}_{\\text{MinT}}=\\left(\mathbf{S}^{\intercal}\mathbf{W}_{h}\mathbf{S}\\right)^{-1}
    \mathbf{S}^{\intercal}\mathbf{W}^{-1}_{h}$$
    
    **Parameters:**<br>
    `method`: str, one of `ols`, `wls_struct`, `wls_var`, `mint_shrink`, `mint_cov`.<br>
    `nonnegative`: bool, reconciled forecasts should be nonnegative?<br>
    `mint_shr_ridge`: float=2e-8, ridge numeric protection to MinTrace-shr covariance estimator.<br>

    **References:**<br>
    - [Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). \"Optimal forecast reconciliation for
    hierarchical and grouped time series through trace minimization\". Journal of the American Statistical Association, 
    114 , 804–819. doi:10.1080/01621459.2018.1448825.](https://robjhyndman.com/publications/mint/).
    - [Wickramasuriya, S.L., Turlach, B.A. & Hyndman, R.J. (2020). \"Optimal non-negative
    forecast reconciliation". Stat Comput 30, 1167–1182, 
    https://doi.org/10.1007/s11222-020-09930-0](https://robjhyndman.com/publications/nnmint/).
    """
    def __init__(self, 
                 method: str,
                 nonnegative: bool = False,
                 mint_shr_ridge: Optional[float] = 2e-8):
        self.method = method
        self.nonnegative = nonnegative
        self.insample = method in ['wls_var', 'mint_cov', 'mint_shrink']
        if method == 'mint_shrink':
            self.mint_shr_ridge = mint_shr_ridge

    def _get_PW_matrices(self, 
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  y_insample: Optional[np.ndarray] = None,
                  y_hat_insample: Optional[np.ndarray] = None,
                  idx_bottom: Optional[List[int]] = None,):
        # shape residuals_insample (n_hiers, obs)
        res_methods = ['wls_var', 'mint_cov', 'mint_shrink']
        if self.method in res_methods and y_insample is None and y_hat_insample is None:
            raise ValueError(f"For methods {', '.join(res_methods)} you need to pass residuals")
        n_hiers, n_bottom = S.shape
        if self.method == 'ols':
            W = np.eye(n_hiers)
        elif self.method == 'wls_struct':
            W = np.diag(S @ np.ones((n_bottom,)))
        elif self.method in res_methods:
            # Residuals with shape (obs, n_hiers)
            residuals = (y_insample - y_hat_insample).T
            n, _ = residuals.shape

            # Protection: against overfitted model
            residuals_sum = np.sum(residuals, axis=0)
            zero_residual_prc = np.abs(residuals_sum) < 1e-4
            zero_residual_prc = np.mean(zero_residual_prc)
            if zero_residual_prc > .98:
                raise Exception(f'Insample residuals close to 0, zero_residual_prc={zero_residual_prc}. Check `Y_df`')

            # Protection: cases where data is unavailable/nan
            masked_res = np.ma.array(residuals, mask=np.isnan(residuals))
            covm = np.ma.cov(masked_res, rowvar=False, allow_masked=True).data

            if self.method == 'wls_var':
                W = np.diag(np.diag(covm))
            elif self.method == 'mint_cov':
                W = covm
            elif self.method == 'mint_shrink':
                # Schäfer and Strimmer 2005, scale invariant shrinkage
                # lasso or ridge might improve numerical stability but
                # this version follows https://robjhyndman.com/papers/MinT.pdf
                tar = np.diag(np.diag(covm))

                # Protections: constant's correlation set to 0
                # standardized residuals 0 where residual_std=0
                corm, residual_std = cov2corr(covm, return_std=True)
                corm = np.nan_to_num(corm, nan=0.0)
                xs = np.divide(residuals, residual_std, 
                               out=np.zeros_like(residuals), where=residual_std!=0)

                xs = xs[~np.isnan(xs).any(axis=1), :]
                v = (1 / (n * (n - 1))) * (crossprod(xs ** 2) - (1 / n) * (crossprod(xs) ** 2))
                np.fill_diagonal(v, 0)

                # Protection: constant's correlation set to 0
                corapn = cov2corr(tar)
                corapn = np.nan_to_num(corapn, nan=0.0)
                d = (corm - corapn) ** 2
                lmd = v.sum() / d.sum()
                lmd = max(min(lmd, 1), 0)

                # Protection: final ridge diagonal protection
                W = (lmd * tar + (1 - lmd) * covm) + self.mint_shr_ridge
        else:
            raise ValueError(f'Unkown reconciliation method {self.method}')

        eigenvalues, _ = np.linalg.eig(W)
        if any(eigenvalues < 1e-8):
            raise Exception(f'min_trace ({self.method}) needs covariance matrix to be positive definite.')

        else:
            # compute P for free reconciliation
            R = S.T @ np.linalg.pinv(W)
            P = np.linalg.pinv(R @ S) @ R

        return P, W

    def reconcile(self, 
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  y_insample: Optional[np.ndarray] = None,
                  y_hat_insample: Optional[np.ndarray] = None,
                  idx_bottom: Optional[List[int]] = None,
                  level: Optional[List[int]] = None,
                  sampler: Optional[Callable] = None):
        """MinTrace Reconciliation Method.

        **Parameters:**<br>
        `S`: Summing matrix of size (`base`, `bottom`).<br>
        `y_hat`: Forecast values of size (`base`, `horizon`).<br>
        `y_insample`: Insample values of size (`base`, `insample_size`). Only used by `wls_var`, `mint_cov`, `mint_shrink`<br>
        `y_hat_insample`: Insample fitted values of size (`base`, `insample_size`). Only used by `wls_var`, `mint_cov`, `mint_shrink`<br>
        `idx_bottom`: Indices corresponding to the bottom level of `S`, size (`bottom`).<br>
        `level`: float list 0-100, confidence levels for prediction intervals.<br>
        `sampler`: Sampler for prediction intevals, one of Normality(), Bootstrap(), PERMBU().<br>

        **Returns:**<br>
        `y_tilde`: Reconciliated y_hat using the MinTrace approach.
        """
        self.P, self.W = self._get_PW_matrices(S=S, y_hat=y_hat, 
                                               y_insample=y_insample, y_hat_insample=y_hat_insample,
                                               idx_bottom=idx_bottom)        
        if not self.nonnegative:
            return _reconcile(S=S, P=self.P, W=self.W, y_hat=y_hat,
                            level=level, sampler=sampler)

        else:
            _, n_bottom = S.shape
            W_inv = np.linalg.pinv(self.W)
            if level is not None and type(sampler).__name__ in ['Bootstrap', 'PERMBU']:
                raise Exception('nonnegative reconciliation is not compatible with bootstrap forecasts')
            if idx_bottom is None:
                raise Exception('idx_bottom needed for nonnegative reconciliation')
            warnings.warn('Replacing negative forecasts with zero.')
            y_hat = np.copy(y_hat)
            y_hat[y_hat < 0] = 0.
            # Quadratic progamming formulation
            # here we are solving the quadratic programming problem
            # formulated in the origial paper
            # https://robjhyndman.com/publications/nnmint/
            # The library quadprog was chosen
            # based on these benchmarks:
            # https://scaron.info/blog/quadratic-programming-in-python.html
            a = S.T @ W_inv
            G = a @ S
            C = np.eye(n_bottom)
            b = np.zeros(n_bottom)
            # the quadratic programming problem
            # returns the forecasts of the bottom series
            bottom_fcts = np.apply_along_axis(lambda y_hat: solve_qp(G=G, a=a @ y_hat, C=C, b=b)[0], 
                                              axis=0, arr=y_hat)
            if not np.all(bottom_fcts > -1e-8):
                raise Exception('nonnegative optimization failed')
            # remove negative values close to zero
            bottom_fcts = np.clip(np.float32(bottom_fcts), a_min=0, a_max=None)
            y_hat = S @ bottom_fcts
            return BottomUp().reconcile(S=S, y_hat=y_hat, idx_bottom=idx_bottom,
                                        level=level, sampler=sampler)

    __call__ = reconcile

# %% ../nbs/methods.ipynb 42
class OptimalCombination(MinTrace):
    """Optimal Combination Reconciliation Class.

    This reconciliation algorithm was proposed by Hyndman et al. 2011, the method uses generalized least squares 
    estimator using the coherency errors covariance matrix. Consider the covariance of the base forecast 
    $\\textrm{Var}(\epsilon_{h}) = \Sigma_{h}$, the $\mathbf{P}$ matrix of this method is defined by:
    $$ \mathbf{P} = \\left(\mathbf{S}^{\intercal}\Sigma_{h}^{\dagger}\mathbf{S}\\right)^{-1}\mathbf{S}^{\intercal}\Sigma^{\dagger}_{h}$$
    where $\Sigma_{h}^{\dagger}$ denotes the variance pseudo-inverse. The method was later proven equivalent to 
    `MinTrace` variants.

    **Parameters:**<br>
    `method`: str, allowed optimal combination methods: 'ols', 'wls_struct'.<br>
    `nonnegative`: bool, reconciled forecasts should be nonnegative?<br>

    **References:**<br>
    - [Rob J. Hyndman, Roman A. Ahmed, George Athanasopoulos, Han Lin Shang (2010). \"Optimal Combination Forecasts for 
    Hierarchical Time Series\".](https://robjhyndman.com/papers/Hierarchical6.pdf).<br>
    - [Shanika L. Wickramasuriya, George Athanasopoulos and Rob J. Hyndman (2010). \"Optimal Combination Forecasts for 
    Hierarchical Time Series\".](https://robjhyndman.com/papers/MinT.pdf).
    - [Wickramasuriya, S.L., Turlach, B.A. & Hyndman, R.J. (2020). \"Optimal non-negative
    forecast reconciliation". Stat Comput 30, 1167–1182, 
    https://doi.org/10.1007/s11222-020-09930-0](https://robjhyndman.com/publications/nnmint/).
    """
    def __init__(self,
                 method: str,
                 nonnegative: bool = False):
        comb_methods = ['ols', 'wls_struct']
        if method not in comb_methods:
            raise ValueError(f"Optimal Combination class does not support method: \"{method}\"")

        self.method = method
        self.nonnegative = nonnegative
        self.insample = False

# %% ../nbs/methods.ipynb 48
@njit
def lasso(X: np.ndarray, y: np.ndarray, 
          lambda_reg: float, max_iters: int = 1_000,
          tol: float = 1e-4):
    # lasso cyclic coordinate descent
    n, feats = X.shape
    norms = (X ** 2).sum(axis=0)
    beta = np.zeros(feats, dtype=np.float32)
    beta_changes = np.zeros(feats, dtype=np.float32)
    residuals = y.copy()
    
    for it in range(max_iters):
        for i, betai in enumerate(beta):
            # is feature is close to zero, we 
            # continue to the next.
            # in this case is optimal betai= 0
            if abs(norms[i]) < 1e-8:
                continue
            xi = X[:, i]
            #we calculate the normalized derivative
            rho = betai + xi.flatten().dot(residuals) / norms[i] #(norms[i] + 1e-3)
            #soft threshold
            beta[i] = np.sign(rho) * max(np.abs(rho) - lambda_reg * n / norms[i], 0.)#(norms[i] + 1e-3), 0.)
            beta_changes[i] = np.abs(betai - beta[i])
            if beta[i] != betai:
                residuals += (betai - beta[i]) * xi
        if max(beta_changes) < tol:
            break
    #print(it)
    return beta

# %% ../nbs/methods.ipynb 49
class ERM:
    """Optimal Combination Reconciliation Class.

    The Empirical Risk Minimization reconciliation strategy relaxes the unbiasedness assumptions from
    previous reconciliation methods like MinT and optimizes square errors between the reconciled predictions
    and the validation data to obtain an optimal reconciliation matrix P.
    
    The exact solution for $\mathbf{P}$ (`method='closed'`) follows the expression:
    $$\mathbf{P}^{*} = \\left(\mathbf{S}^{\intercal}\mathbf{S}\\right)^{-1}\mathbf{Y}^{\intercal}\hat{\mathbf{Y}}\\left(\hat{\mathbf{Y}}\hat{\mathbf{Y}}\\right)^{-1}$$

    The alternative Lasso regularized $\mathbf{P}$ solution (`method='reg_bu'`) is useful when the observations 
    of validation data is limited or the exact solution has low numerical stability.
    $$\mathbf{P}^{*} = \\text{argmin}_{\mathbf{P}} ||\mathbf{Y}-\mathbf{S} \mathbf{P} \hat{Y} ||^{2}_{2} + \lambda ||\mathbf{P}-\mathbf{P}_{\\text{BU}}||_{1}$$

    **Parameters:**<br>
    `method`: str, one of `closed`, `reg` and `reg_bu`.<br>
    `lambda_reg`: float, l1 regularizer for `reg` and `reg_bu`.<br>

    **References:**<br>
    - [Ben Taieb, S., & Koo, B. (2019). Regularized regression for hierarchical forecasting without 
    unbiasedness conditions. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge 
    Discovery & Data Mining KDD '19 (p. 1337{1347). New York, NY, USA: Association for Computing Machinery.](https://doi.org/10.1145/3292500.3330976).<br>
    """
    def __init__(self,
                 method: str,
                 lambda_reg: float = 1e-2):
        self.method = method
        self.lambda_reg = lambda_reg
        self.insample = True

    def _get_PW_matrices(self, 
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  y_insample: np.ndarray,
                  y_hat_insample: np.ndarray,
                  idx_bottom: np.ndarray):
        n_hiers, n_bottom = S.shape
        # y_hat_insample shape (n_hiers, obs)
        # remove obs with nan values
        nan_idx = np.isnan(y_hat_insample).any(axis=0)
        y_insample = y_insample[:, ~nan_idx]
        y_hat_insample = y_hat_insample[:, ~nan_idx]
        #only using h validation steps to avoid 
        #computational burden
        #print(y_hat.shape)
        h = min(y_hat.shape[1], y_hat_insample.shape[1])
        y_hat_insample = y_hat_insample[:, -h:] # shape (h, n_hiers)
        y_insample = y_insample[:, -h:]
        if self.method == 'closed':
            B = np.linalg.inv(S.T @ S) @ S.T @ y_insample
            B = B.T
            P = np.linalg.pinv(y_hat_insample.T) @ B
            P = P.T
        elif self.method in ['reg', 'reg_bu']:
            X = np.kron(np.array(S, order='F'), np.array(y_hat_insample.T, order='F'))
            Pbu = np.zeros_like(S)
            if self.method == 'reg_bu':
                Pbu[idx_bottom] = S[idx_bottom]
            Pbu = Pbu.T
            Y = y_insample.T.flatten(order='F') - X @ Pbu.T.flatten(order='F')
            if self.lambda_reg is None:
                lambda_reg = np.max(np.abs(X.T.dot(Y)))
            else:
                lambda_reg = self.lambda_reg
            P = lasso(X, Y, lambda_reg)
            P = P + Pbu.T.flatten(order='F')
            P = P.reshape(-1, n_bottom, order='F').T
        else:
            raise ValueError(f'Unkown reconciliation method {self.method}')

        W = np.eye(n_hiers, dtype=np.float32)

        return P, W

    def reconcile(self, 
                  S: np.ndarray,
                  y_hat: np.ndarray,
                  y_insample: np.ndarray,
                  y_hat_insample: np.ndarray,
                  idx_bottom: np.ndarray,
                  level: Optional[List[int]] = None,
                  sampler: Optional[Callable] = None):
        """ERM Reconciliation Method.

        **Parameters:**<br>
        `S`: Summing matrix of size (`base`, `bottom`).<br>
        `y_hat`: Forecast values of size (`base`, `horizon`).<br>
        `y_insample`: Train values of size (`base`, `insample_size`).<br>
        `y_hat_insample`: Insample train predictions of size (`base`, `insample_size`).<br>
        `idx_bottom`: Indices corresponding to the bottom level of `S`, size (`bottom`).<br>
        `level`: float list 0-100, confidence levels for prediction intervals.<br>
        `sampler`: Sampler for prediction intevals, one of Normality(), Bootstrap(), PERMBU().<br>

        **Returns:**<br>
        `y_tilde`: Reconciliated y_hat using the ERM approach.
        """
        self.P, self.W = self._get_PW_matrices(S=S,
                                               y_hat=y_hat,
                                               y_insample=y_insample,
                                               y_hat_insample=y_hat_insample,
                                               idx_bottom=idx_bottom)
        return _reconcile(S=S, P=self.P, W=self.W, y_hat=y_hat,
                          level=level, sampler=sampler)

    __call__ = reconcile
