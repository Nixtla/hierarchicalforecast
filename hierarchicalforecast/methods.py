# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/methods.ipynb.

# %% auto 0
__all__ = ['BottomUp', 'TopDown', 'MiddleOut', 'MinTrace', 'OptimalCombination', 'ERM']

# %% ../nbs/methods.ipynb 2
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
from numba import njit
from scipy.stats import norm
from statsmodels.stats.moment_helpers import cov2corr

# %% ../nbs/methods.ipynb 4
def _reconcile(S: np.ndarray, P: np.ndarray, W: np.ndarray, 
               y_hat: np.ndarray, SP: np.ndarray = None,
               sigmah: Optional[np.ndarray] = None, 
               level: Optional[List[int]] = None):
    if SP is None:
        SP = S @ P
    res = {'mean': np.matmul(SP, y_hat)}
    if sigmah is not None and level is not None:
        #then we calculate prediction intervals
        # we assume normality
        # we have to calculate the "reconciled" sigmah
        # following
        # https://otexts.com/fpp3/rec-prob.html
        R1 = cov2corr(W)
        W_h = [np.diag(sigma) @ R1 @ np.diag(sigma).T for sigma in sigmah.T]
        sigmah = np.hstack([np.sqrt(np.diag(SP @ W @ SP.T))[:, None] for W in W_h])
        res['sigmah'] = sigmah
        # intervals calc
        level = np.asarray(level)
        z = norm.ppf(0.5 + level / 200)
        for zs, lv in zip(z, level):
            res[f'lo-{lv}'] = res['mean'] - zs * sigmah
            res[f'hi-{lv}'] = res['mean'] + zs * sigmah
    return res

# %% ../nbs/methods.ipynb 5
def bottom_up(S: np.ndarray,
              y_hat: np.ndarray,
              idx_bottom: List[int],
              sigmah: Optional[np.ndarray] = None, 
              level: Optional[List[int]] = None):
    n_hiers, n_bottom = S.shape
    P = np.zeros_like(S, dtype=np.float32)
    P[idx_bottom] = S[idx_bottom]
    P = P.T
    W = np.eye(n_hiers, dtype=np.float32)
    return _reconcile(S, P, W, y_hat, sigmah=sigmah, level=level)

# %% ../nbs/methods.ipynb 6
class BottomUp:
    
    def reconcile(
            self,
            S: np.ndarray, # Summing matrix of size (`base`, `bottom`)
            y_hat: np.ndarray, # Forecast values of size (`base`, `horizon`)
            idx_bottom: np.ndarray, # Indices corresponding to the bottom level of `S`, size (`bottom`)
            sigmah: Optional[np.ndarray] = None, # Estimate of the standard deviation of the h-step forecast of size (`base`, `horizon`)
            level: Optional[List[int]] = None, # Levels of probabilistic forecasts
        ):
        return bottom_up(S=S, y_hat=y_hat, 
                         idx_bottom=idx_bottom, sigmah=sigmah, level=level)
    
    __call__ = reconcile

# %% ../nbs/methods.ipynb 15
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

# %% ../nbs/methods.ipynb 17
def _get_child_nodes(S: np.ndarray, tags: Dict[str, np.ndarray]):
    childs = {}
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

# %% ../nbs/methods.ipynb 18
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

# %% ../nbs/methods.ipynb 19
def top_down(S: np.ndarray, 
             y_hat: np.ndarray,
             y_insample: np.ndarray,
             tags: Dict[str, np.ndarray],
             method: str,
             sigmah: Optional[np.ndarray] = None, 
             level: Optional[List[int]] = None):
    if not is_strictly_hierarchical(S, tags):
        raise ValueError('Top down reconciliation requires strictly hierarchical structures.')
    
    n_hiers, n_bottom = S.shape
    idx_top = int(S.sum(axis=1).argmax())
    levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
    idx_bottom = levels_[list(levels_)[-1]]
    
    if method == 'forecast_proportions':
        if sigmah is not None and level is not None:
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
        y_top = y_insample[idx_top]
        y_btm = y_insample[idx_bottom]
        if method == 'average_proportions':
            prop = np.mean(y_btm / y_top, axis=1)
        elif method == 'proportion_averages':
            prop = np.mean(y_btm, axis=1) / np.mean(y_top)
        else:
            raise Exception(f'Unknown method {method}')
    P = np.zeros_like(S, np.float64).T #float 64 if prop is too small, happens with wiki2
    P[:, idx_top] = prop
    W = np.eye(n_hiers, dtype=np.float32)
    return _reconcile(S, P, W, y_hat, sigmah=sigmah, level=level)

# %% ../nbs/methods.ipynb 20
class TopDown:
    
    def __init__(
            self, 
            method: str # One of `forecast_proportions`, `average_proportions` and `proportion_averages`
        ):
        self.method = method
    
    def reconcile(
            self, 
            S: np.ndarray, # Summing matrix of size (`base`, `bottom`)
            y_hat: np.ndarray, # Forecast values of size (`base`, `horizon`)
            y_insample: np.ndarray, # Insample values of size (`base`, `insample_size`)
            tags: Dict[str, np.ndarray], # Each key is a level and each value its `S` indices
            sigmah: Optional[np.ndarray] = None, # Estimate of the standard deviation of the h-step forecast of size (`base`, `horizon`)
            level: Optional[List[int]] = None, # Levels of probabilistic forecasts
        ):
        return top_down(S=S, y_hat=y_hat, 
                        y_insample=y_insample, 
                        tags=tags,
                        method=self.method,
                        sigmah=sigmah, level=level)
    
    __call__ = reconcile

# %% ../nbs/methods.ipynb 27
def middle_out(S: np.ndarray, 
               y_hat: np.ndarray,
               y_insample: np.ndarray,
               tags: Dict[str, np.ndarray],
               middle_level: str,
               top_down_method: str):
    if not is_strictly_hierarchical(S, tags):
        raise ValueError('Middle out reconciliation requires strictly hierarchical structures.')
    if middle_level not in tags.keys():
        raise ValueError('You have to provide a `middle_level` in `tags`.')
    levels_ = dict(sorted(tags.items(), key=lambda x: len(x[1])))
    reconciled = np.full_like(y_hat, fill_value=np.nan)
    cut_nodes = levels_[middle_level]
    # bottom up reconciliation
    idxs_bu = []
    for node, idx_node in levels_.items():
        idxs_bu.append(idx_node)
        if node == middle_level:
            break
    idxs_bu = np.hstack(idxs_bu)
    #bottom up forecasts
    bu = bottom_up(S=np.unique(S[idxs_bu], axis=1), 
                   y_hat=y_hat[idxs_bu], 
                   idx_bottom=np.arange(len(idxs_bu))[-len(cut_nodes):])
    reconciled[idxs_bu] = bu['mean']
    
    #top down
    child_nodes = _get_child_nodes(S, levels_)
    # parents contains each node in the middle out level
    # as key. The values of each node are the levels that
    # are conected to that node.
    parents = {node: {middle_level: np.array([node])} for node in cut_nodes}
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
        td = top_down(S_node, 
                      y_hat[idxs_node], 
                      y_insample[idxs_node], 
                      levels_node_, 
                      method=top_down_method)
        reconciled[idxs_node] = td['mean']
    return {'mean': reconciled}
        

# %% ../nbs/methods.ipynb 28
class MiddleOut:
    
    def __init__(
            self, 
            middle_level: str, # Middle level 
            top_down_method: str # One of `forecast_proportions`, `average_proportions` and `proportion_averages`
        ):
        self.middle_level = middle_level
        self.top_down_method = top_down_method 
    
    def reconcile(
            self, 
            S: np.ndarray, # Summing matrix of size (`base`, `bottom`)
            y_hat: np.ndarray, # Forecast values of size (`base`, `horizon`)
            y_insample: np.ndarray, # Insample values of size (`base`, `insample_size`)
            tags: Dict[str, np.ndarray] # Each key is a level and each value its `S` indices
        ):
        return middle_out(S=S, y_hat=y_hat, 
                          y_insample=y_insample, 
                          tags=tags,
                          middle_level=self.middle_level,
                          top_down_method=self.top_down_method)
    
    __call__ = reconcile

# %% ../nbs/methods.ipynb 34
def crossprod(x):
    return x.T @ x

# %% ../nbs/methods.ipynb 35
def min_trace(S: np.ndarray, 
              y_hat: np.ndarray,
              y_insample: np.ndarray,
              y_hat_insample: np.ndarray,
              method: str,
              sigmah: Optional[np.ndarray] = None,
              level: Optional[List[int]] = None):
    # shape residuals_insample (n_hiers, obs)
    res_methods = ['wls_var', 'mint_cov', 'mint_shrink']
    if method in res_methods and y_insample is None and y_hat_insample is None:
        raise ValueError(f"For methods {', '.join(res_methods)} you need to pass residuals")
    n_hiers, n_bottom = S.shape
    if method == 'ols':
        W = np.eye(n_hiers)
    elif method == 'wls_struct':
        W = np.diag(S @ np.ones((n_bottom,)))
    elif method in res_methods:
        #we need residuals with shape (obs, n_hiers)
        residuals = (y_insample - y_hat_insample).T
        n, _ = residuals.shape
        masked_res = np.ma.array(residuals, mask=np.isnan(residuals))
        covm = np.ma.cov(masked_res, rowvar=False, allow_masked=True).data
        if method == 'wls_var':
            W = np.diag(np.diag(covm))
        elif method == 'mint_cov':
            W = covm
        elif method == 'mint_shrink':
            tar = np.diag(np.diag(covm))
            corm = cov2corr(covm)
            xs = np.divide(residuals, np.sqrt(np.diag(covm)))
            xs = xs[~np.isnan(xs).any(axis=1), :]
            v = (1 / (n * (n - 1))) * (crossprod(xs ** 2) - (1 / n) * (crossprod(xs) ** 2))
            np.fill_diagonal(v, 0)
            corapn = cov2corr(tar)
            d = (corm - corapn) ** 2
            lmd = v.sum() / d.sum()
            lmd = max(min(lmd, 1), 0)
            W = lmd * tar + (1 - lmd) * covm
    else:
        raise ValueError(f'Unkown reconciliation method {method}')
    
    eigenvalues, _ = np.linalg.eig(W)
    if any(eigenvalues < 1e-8):
        raise Exception(f'min_trace ({method}) needs covariance matrix to be positive definite.')
    
    R = S.T @ np.linalg.pinv(W)
    P = np.linalg.pinv(R @ S) @ R
    
    return _reconcile(S, P, W, y_hat, sigmah=sigmah, level=level)

# %% ../nbs/methods.ipynb 36
class MinTrace:
    
    def __init__(
            self, 
            method: str # One of `ols`, `wls_struct`, `wls_var`, `mint_shrink`, `mint_co`
        ):
        self.method = method
        
    def reconcile(
            self, 
            S: np.ndarray, # Summing matrix of size (`base`, `bottom`)
            y_hat: np.ndarray, # Forecast values of size (`base`, `horizon`)
            y_insample: np.ndarray, # Insample values of size (`base`, `insample_size`)
            y_hat_insample: np.ndarray, # Insample forecasts of size (`base`, `insample_size`)
            sigmah: Optional[np.ndarray] = None, # Estimate of the standard deviation of the h-step forecast of size (`base`, `horizon`)
            level: Optional[List[int]] = None, # Levels of probabilistic forecasts
        ):
        return min_trace(S=S, y_hat=y_hat, 
                         y_insample=y_insample,
                         y_hat_insample=y_hat_insample,
                         method=self.method,
                         sigmah=sigmah,
                         level=level)
    
    __call__ = reconcile

# %% ../nbs/methods.ipynb 43
def optimal_combination(S: np.ndarray, 
                        y_hat: np.ndarray,
                        method: str,
                        y_insample: np.ndarray = None,
                        y_hat_insample: np.ndarray = None,
                        sigmah: Optional[np.ndarray] = None, 
                        level: Optional[List[int]] = None):
    
    return min_trace(S=S, y_hat=y_hat, 
                     y_insample=y_insample,
                     y_hat_insample=y_hat_insample,
                     method=method, sigmah=sigmah, level=level)

# %% ../nbs/methods.ipynb 44
class OptimalCombination:
    
    def __init__(
            self, 
            method: str # Allowed Optimal Combination Methods: 'ols', 'wls_struct'
        ):
        comb_methods = ['ols', 'wls_struct']
        if method not in comb_methods:
            raise ValueError(f"Optimal Combination class does not support method: \"{method}\"")
        
        self.method = method
    
    def reconcile(
            self,
            S: np.ndarray, # Summing matrix of size (`base`, `bottom`)
            y_hat: np.ndarray, # Forecast values of size (`base`, `horizon`)
            y_insample: np.ndarray = None, # Insample values of size (`base`, `insample_size`)
            y_hat_insample: np.ndarray = None, # Insample forecasts of size (`base`, `insample_size`)
            sigmah: Optional[np.ndarray] = None, # Estimate of the standard deviation of the h-step forecast of size (`base`, `horizon`)
            level: Optional[List[int]] = None, # Levels of probabilistic forecasts
        ):
        return optimal_combination(S=S, 
                                   y_hat=y_hat, 
                                   y_insample=y_insample, 
                                   y_hat_insample=y_hat_insample, 
                                   method=self.method, sigmah=sigmah, 
                                   level=level)
    
    __call__ = reconcile

# %% ../nbs/methods.ipynb 51
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

# %% ../nbs/methods.ipynb 52
def erm(S: np.ndarray,
        y_hat: np.ndarray,
        y_insample: np.ndarray,
        y_hat_insample: np.ndarray,
        idx_bottom: np.ndarray,
        method: str,
        lambda_reg: float = 1e-3,
        sigmah: Optional[np.ndarray] = None, 
        level: Optional[List[int]] = None):
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
    if method == 'closed':
        B = np.linalg.inv(S.T @ S) @ S.T @ y_insample
        B = B.T
        P = np.linalg.pinv(y_hat_insample.T) @ B
        P = P.T
    elif method in ['reg', 'reg_bu']:
        X = np.kron(np.array(S, order='F'), np.array(y_hat_insample.T, order='F'))
        Pbu = np.zeros_like(S)
        if method == 'reg_bu':
            Pbu[idx_bottom] = S[idx_bottom]
        Pbu = Pbu.T
        Y = y_insample.T.flatten(order='F') - X @ Pbu.T.flatten(order='F')
        if lambda_reg is None:
            lambda_reg = np.max(np.abs(X.T.dot(Y)))
        P = lasso(X, Y, lambda_reg)
        P = P + Pbu.T.flatten(order='F')
        P = P.reshape(-1, n_bottom, order='F').T
    else:
        raise ValueError(f'Unkown reconciliation method {method}')
        
    W = np.eye(n_hiers, dtype=np.float32)
    
    return _reconcile(S, P, W, y_hat, sigmah=sigmah, level=level)

# %% ../nbs/methods.ipynb 53
class ERM:
    
    def __init__(
            self, 
            method: str, # one of `closed`, `reg` and `reg_bu`
            lambda_reg: float = 1e-2 # l1 regularizer for `reg` and `reg_bu`
        ):
        self.method = method
        self.lambda_reg = lambda_reg
        
    def reconcile(
            self, 
            S: np.ndarray, # Summing matrix of size (`base`, `bottom`)
            y_hat: np.ndarray, # Forecast values of size (`base`, `horizon`)
            y_insample: np.ndarray, # Insample values of size (`base`, `insample_size`)
            y_hat_insample: np.ndarray, # Insample forecasts of size (`base`, `insample_size`)
            idx_bottom: np.ndarray, # Indices corresponding to the bottom level of `S`, size (`bottom`)
            sigmah: Optional[np.ndarray] = None, # Estimate of the standard deviation of the h-step forecast of size (`base`, `horizon`)
            level: Optional[List[int]] = None, # Levels of probabilistic forecasts
        ):
        return erm(S=S, y_hat=y_hat, 
                   y_insample=y_insample,
                   y_hat_insample=y_hat_insample,
                   idx_bottom=idx_bottom,
                   method=self.method, lambda_reg=self.lambda_reg,
                   sigmah=sigmah, level=level)
    
    __call__ = reconcile
