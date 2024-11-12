# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/src/core.ipynb.

# %% auto 0
__all__ = ['HierarchicalReconciliation']

# %% ../nbs/src/core.ipynb 4
import re
import time
import copy

from .methods import HReconciler
from .utils import pivot
from inspect import signature
from narwhals.typing import FrameT
from scipy.stats import norm
from scipy import sparse
from typing import Dict, List, Optional

import narwhals as nw
import numpy as np

# %% ../nbs/src/core.ipynb 6
def _build_fn_name(fn) -> str:
    fn_name = type(fn).__name__
    func_params = fn.__dict__

    # Take default parameter out of names
    args_to_remove = ['insample', 'num_threads']
    if not func_params.get('nonnegative', False):
        args_to_remove.append('nonnegative')

    if fn_name == 'MinTrace' and \
        func_params['method']=='mint_shrink':
        if func_params['mint_shr_ridge'] == 2e-8:
            args_to_remove += ['mint_shr_ridge']

    func_params = [f'{name}-{value}' for name, value in func_params.items() if name not in args_to_remove]
    if func_params:
        fn_name += '_' + '_'.join(func_params)
    return fn_name

# %% ../nbs/src/core.ipynb 10
def _reverse_engineer_sigmah(Y_hat_df: FrameT, 
                             y_hat: np.ndarray, 
                             model_name: str,
                             id_col: str = "unique_id",
                             time_col: str = "ds",
                             target_col: str = "y") -> np.ndarray:
    """
    This function assumes that the model creates prediction intervals
    under a normality with the following the Equation:
    $\hat{y}_{t+h} + c \hat{sigma}_{h}$

    In the future, we might deprecate this function in favor of a 
    direct usage of an estimated $\hat{sigma}_{h}$
    """

    drop_cols = [time_col]
    if target_col in Y_hat_df.columns:
        drop_cols.append(target_col)
    if model_name+'-median' in Y_hat_df.columns:
        drop_cols.append(model_name+'-median')
    model_names = Y_hat_df.drop(drop_cols).columns
    pi_model_names = [name for name in model_names if ('-lo' in name or '-hi' in name)]
    pi_model_name = [pi_name for pi_name in pi_model_names if model_name in pi_name]
    pi = len(pi_model_name) > 0

    n_series = len(Y_hat_df[id_col].unique())

    if not pi:
        raise Exception(f'Please include `{model_name}` prediction intervals in `Y_hat_df`')

    pi_col = pi_model_name[0]
    sign = -1 if 'lo' in pi_col else 1
    level_cols = re.findall('[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+', pi_col)
    level_col = float(level_cols[-1])
    z = norm.ppf(0.5 + level_col / 200)
    sigmah = Y_hat_df[pi_col].to_numpy().reshape(n_series,-1)
    sigmah = sign * (sigmah - y_hat) / z

    return sigmah

# %% ../nbs/src/core.ipynb 11
class HierarchicalReconciliation:
    """Hierarchical Reconciliation Class.

    The `core.HierarchicalReconciliation` class allows you to efficiently fit multiple 
    HierarchicaForecast methods for a collection of time series and base predictions stored in 
    pandas DataFrames. The `Y_df` dataframe identifies series and datestamps with the unique_id and ds columns while the
    y column denotes the target time series variable. The `Y_h` dataframe stores the base predictions, 
    example ([AutoARIMA](https://nixtla.github.io/statsforecast/models.html#autoarima), [ETS](https://nixtla.github.io/statsforecast/models.html#autoets), etc.).

    **Parameters:**<br>
    `reconcilers`: A list of instantiated classes of the [reconciliation methods](https://nixtla.github.io/hierarchicalforecast/methods.html) module .<br>

    **References:**<br>
    [Rob J. Hyndman and George Athanasopoulos (2018). \"Forecasting principles and practice, Hierarchical and Grouped Series\".](https://otexts.com/fpp3/hierarchical.html)
    """
    def __init__(self,
                 reconcilers: List[HReconciler]):
        self.reconcilers = reconcilers
        self.orig_reconcilers = copy.deepcopy(reconcilers) # TODO: elegant solution
        self.insample = any([method.insample for method in reconcilers])
    
    def _prepare_fit(self,
                     Y_hat_df: FrameT,
                     S_df: FrameT,
                     Y_df: Optional[FrameT],
                     tags: Dict[str, np.ndarray],
                     level: Optional[List[int]] = None,
                     intervals_method: str = 'normality',
                     sort_df: bool = True,
                     id_col: str = "unique_id",
                     time_col: str = "ds", 
                     target_col: str = "y",                      
                     ):
        """
        Performs preliminary wrangling and protections
        """
        Y_hat_df_ = nw.from_native(Y_hat_df)
        S_df_ = nw.from_native(S_df)
        Y_df_ = None
        self.native_namespace = nw.get_native_namespace(Y_hat_df_)

        #-------------------------------- Match Y_hat/Y/S index order --------------------------------#
        # TODO: This is now a bit slow as we always sort.
        S_df_ = S_df_.with_columns(nw.from_dict({f"{id_col}_id": np.arange(len(S_df_))}, 
                                                native_namespace=self.native_namespace)[f"{id_col}_id"])
        Y_hat_df_ = Y_hat_df_.join(S_df_[[id_col, f"{id_col}_id"]], on=id_col, how='left')
        Y_hat_df_ = Y_hat_df_.sort(by=[f"{id_col}_id", time_col])
        Y_hat_df_ = Y_hat_df_.drop(f"{id_col}_id")

        if Y_df is not None:
            Y_df_ = nw.from_native(Y_df)
            Y_df_ = Y_df_.join(S_df_[[id_col, f"{id_col}_id"]], on=id_col, how='left')
            Y_df_ = Y_df_.sort(by=[f"{id_col}_id", time_col])
            Y_df_ = Y_df_.drop(f"{id_col}_id")

        S_df_ = S_df_.drop(f"{id_col}_id")

        #----------------------------------- Check Input's Validity ----------------------------------#

        # Check input's validity
        if intervals_method not in ['normality', 'bootstrap', 'permbu']:
            raise ValueError(f'Unknown interval method: {intervals_method}')

        if self.insample or (intervals_method in ['bootstrap', 'permbu']):
            if Y_df is None:
                raise Exception('You need to provide `Y_df`.')
        
        # Protect level list
        if (level is not None):
            level_outside_domain = np.any((np.array(level) < 0) | (np.array(level) >= 100 ))
            if level_outside_domain and (intervals_method in ['normality', 'permbu']):
                raise ValueError("Level must be a list containing floating values in the interval [0, 100).")

        # Declare output names
        model_names = list(set(Y_hat_df_.columns) - set([id_col, time_col, target_col]))

        # Ensure numeric columns
        assert Y_hat_df_[model_names].columns == Y_hat_df_[model_names].select(nw.selectors.numeric()).columns, "All models in `Y_hat_df` must contain numeric values."

        # Ensure no null values
        assert not Y_hat_df_[model_names].select(nw.all().is_null().any()).to_numpy().any(), "`Y_hat_df` contains null values. Make sure no column in `Y_hat_df` contains null values."
        
        # TODO: Complete y_hat_insample protection
        model_names = [name for name in model_names if not ('-lo' in name or '-hi' in name or '-median' in name)]        
        if intervals_method in ['bootstrap', 'permbu'] and Y_df_ is not None:
            if not (set(model_names) <= set(Y_df_.columns)):
                raise Exception(f"Check `Y_df` columns, {model_names} must be in `Y_df` columns.")

        # Assert S is an identity matrix at the bottom
        S_np = S_df_.drop(id_col).to_numpy()
        if not np.allclose(S_np[-S_np.shape[1]:], np.eye(S_np.shape[1])):
            raise ValueError(f"The bottom {S_np.shape[1]}x{S_np.shape[1]} part of S must be an identity matrix.")

        # Check Y_hat_df\S_df series difference
        # TODO: this logic should be method specific
        S_diff = set(S_df_[id_col]) - set(Y_hat_df_[id_col])
        Y_hat_diff = set(Y_hat_df_[id_col]) - set(S_df_[id_col])
        if S_diff:
            raise Exception(f'There are unique_ids in S_df that are not in Y_hat_df: {S_diff}')
        if Y_hat_diff:
            raise Exception(f'There are unique_ids in Y_hat_df that are not in S_df: {Y_hat_diff}')

        if Y_df_ is not None:
            Y_diff = set(Y_df_[id_col]) - set(Y_hat_df_[id_col])
            Y_hat_diff = set(Y_hat_df_[id_col]) - set(Y_df_[id_col])
            if Y_diff:
                raise Exception(f'There are unique_ids in Y_df that are not in Y_hat_df: {Y_diff}')
            if Y_hat_diff:
                raise Exception(f'There are unique_ids in Y_hat_df that are not in Y_df: {Y_hat_diff}')

        # Same Y_hat_df/S_df/Y_df's unique_ids. Order is guaranteed by the sort_df flag.
        # TODO: this logic should be method specific
        unique_ids = set(Y_hat_df_[id_col])
        S_df_ = S_df_.filter(nw.col(id_col).is_in(unique_ids))

        return Y_hat_df_, S_df_, Y_df_, model_names

    def _prepare_Y(self, 
                          Y_df: FrameT, 
                          S_df: FrameT, 
                          is_balanced: bool = True,
                          id_col: str = "unique_id",
                          time_col: str = "ds", 
                          target_col: str = "y", 
                          ) -> np.ndarray:
        """
        Prepare Y data.
        """
        if is_balanced:
            Y = Y_df[target_col].to_numpy().reshape(len(S_df), -1)
        else:
            Y_pivot = pivot(Y_df, index=id_col, columns=time_col, values=target_col, sort=True)

            # TODO: check if this is the best way to do it
            pos_in_Y = np.searchsorted(Y_pivot[id_col], S_df[id_col])
            Y_pivot = Y_pivot.drop(id_col)
            Y_pivot = Y_pivot[pos_in_Y]
            Y = Y_pivot.to_numpy()

        # TODO: the result is a Fortran contiguous array, see if we can avoid the below copy
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        return Y


    def reconcile(self, 
                  Y_hat_df: FrameT,
                  S: FrameT,
                  tags: Dict[str, np.ndarray],
                  Y_df: Optional[FrameT] = None,
                  level: Optional[List[int]] = None,
                  intervals_method: str = 'normality',
                  num_samples: int = -1,
                  seed: int = 0,
                  sort_df: bool = True,
                  is_balanced: bool = False,
                  id_col: str = "unique_id",
                  time_col: str = "ds", 
                  target_col: str = "y",                   
        ):
        """Hierarchical Reconciliation Method.

        The `reconcile` method is analogous to SKLearn `fit_predict` method, it 
        applies different reconciliation techniques instantiated in the `reconcilers` list.

        Most reconciliation methods can be described by the following convenient 
        linear algebra notation:

        $$\\tilde{\mathbf{y}}_{[a,b],\\tau} = \mathbf{S}_{[a,b][b]} \mathbf{P}_{[b][a,b]} \hat{\mathbf{y}}_{[a,b],\\tau}$$

        where $a, b$ represent the aggregate and bottom levels, $\mathbf{S}_{[a,b][b]}$ contains
        the hierarchical aggregation constraints, and $\mathbf{P}_{[b][a,b]}$ varies across 
        reconciliation methods. The reconciled predictions are $\\tilde{\mathbf{y}}_{[a,b],\\tau}$, and the 
        base predictions $\hat{\mathbf{y}}_{[a,b],\\tau}$.

        **Parameters:**<br>
        `Y_hat_df`: DataFrame, base forecasts with columns ['unique_id', 'ds'] and models to reconcile.<br>
        `Y_df`: DataFrame, training set of base time series with columns `['unique_id', 'ds', 'y']`.<br>
        If a class of `self.reconciles` receives `y_hat_insample`, `Y_df` must include them as columns.<br>
        `S`: DataFrame with summing matrix of size `(base, bottom)`, see [aggregate method](https://nixtla.github.io/hierarchicalforecast/utils.html#aggregate).<br>
        `tags`: Each key is a level and its value contains tags associated to that level.<br>
        `level`: positive float list [0,100), confidence levels for prediction intervals.<br>
        `intervals_method`: str, method used to calculate prediction intervals, one of `normality`, `bootstrap`, `permbu`.<br>
        `num_samples`: int=-1, if positive return that many probabilistic coherent samples.
        `seed`: int=0, random seed for numpy generator's replicability.<br>
        `sort_df` : bool (default=True), if True, sort `df` by [`unique_id`,`ds`].<br>
        `is_balanced`: bool=False, wether `Y_df` is balanced, set it to True to speed things up if `Y_df` is balanced.<br>
        `id_col` : str='unique_id', column that identifies each serie.<br>
        `time_col` : str='ds', column that identifies each timestep, its values can be timestamps or integers.<br>
        `target_col` : str='y', column that contains the target.        

        **Returns:**<br>
        `Y_tilde_df`: DataFrame, with reconciled predictions.
        """

        # Check input's validity and sort dataframes
        is_pandas_Y_hat_df = nw.dependencies.is_pandas_like_dataframe(Y_hat_df)
        is_pandas_S_df = nw.dependencies.is_pandas_like_dataframe(S)
        if Y_df is not None:
            is_pandas_Y_df = nw.dependencies.is_pandas_like_dataframe(Y_df)

        Y_hat_df, S_df, Y_df, self.model_names = \
                    self._prepare_fit(Y_hat_df=Y_hat_df,
                                      S_df=S,
                                      Y_df=Y_df,
                                      tags=tags,
                                      level=level,
                                      intervals_method=intervals_method,
                                      sort_df=sort_df,
                                      id_col=id_col,
                                      time_col=time_col,
                                      target_col=target_col,                                     
                                      )

        # Initialize reconciler arguments
        reconciler_args = dict(
            idx_bottom=np.arange(len(S_df))[-S_df.shape[1]:],
            tags={key: S_df.with_columns(nw.col(id_col).is_in(val).alias("in_cols"))["in_cols"].to_numpy().nonzero()[0] for key, val in tags.items()},
        )

        any_sparse = any([method.is_sparse_method for method in self.reconcilers])
        if any_sparse:
            if not is_pandas_Y_hat_df or not is_pandas_S_df:
                raise ValueError("You have one or more sparse reconciliation methods. Please convert `S_df` and `Y_hat_df` to a pandas DataFrame.")
            try:
                S_for_sparse = sparse.csr_matrix(S_df.drop(id_col).to_native().sparse.to_coo())                
            except AttributeError:
                S_for_sparse = sparse.csr_matrix(S_df.drop(id_col).to_numpy().astype(np.float64, copy=False))

        if Y_df is not None:
            if any_sparse and not is_pandas_Y_df:
                raise ValueError("You have one or more sparse reconciliation methods. Please convert `Y_df` to a pandas DataFrame.")      
            y_insample = self._prepare_Y(Y_df=Y_df, 
                                         S_df=S_df, 
                                         is_balanced=is_balanced, 
                                         id_col=id_col, 
                                         time_col=time_col, 
                                         target_col=target_col)     
            reconciler_args['y_insample'] = y_insample

        Y_tilde_df = Y_hat_df.clone()
        self.execution_times = {}
        self.level_names = {}
        self.sample_names = {}
        for reconciler, name_copy in zip(self.reconcilers, self.orig_reconcilers):
            reconcile_fn_name = _build_fn_name(name_copy)

            if reconciler.is_sparse_method:
                reconciler_args["S"] = S_for_sparse
            else:
                reconciler_args["S"] = S_df.drop(id_col)\
                                           .to_numpy()\
                                           .astype(np.float64, copy=False)

            has_fitted = 'y_hat_insample' in signature(reconciler.fit_predict).parameters
            has_level = 'level' in signature(reconciler.fit_predict).parameters

            for model_name in self.model_names:
                start = time.time()
                recmodel_name = f'{model_name}/{reconcile_fn_name}'

                # TODO: the below should be method specific
                y_hat = self._prepare_Y(Y_df=Y_hat_df[[id_col, time_col, model_name]], 
                                        S_df=S_df, 
                                        is_balanced=True, 
                                        id_col=id_col, 
                                        time_col=time_col, 
                                        target_col=model_name)
                reconciler_args['y_hat'] = y_hat

                if (self.insample and has_fitted) or intervals_method in ['bootstrap', 'permbu']:
                    y_hat_insample = self._prepare_Y(Y_df=Y_df[[id_col, time_col, model_name]], 
                                         S_df=S_df, 
                                         is_balanced=is_balanced, 
                                         id_col=id_col, 
                                         time_col=time_col, 
                                         target_col=model_name)   
                    reconciler_args['y_hat_insample'] = y_hat_insample

                if has_level and (level is not None):
                    if intervals_method in ['normality', 'permbu']:
                        sigmah = _reverse_engineer_sigmah(Y_hat_df=Y_hat_df,
                                    y_hat=y_hat, model_name=model_name)
                        reconciler_args['sigmah'] = sigmah

                    reconciler_args['intervals_method'] = intervals_method
                    reconciler_args['num_samples'] = 200 # TODO: solve duplicated num_samples
                    reconciler_args['seed'] = seed

                # Mean and Probabilistic reconciliation
                kwargs_ls = [key for key in signature(reconciler.fit_predict).parameters if key in reconciler_args.keys()]
                kwargs = {key: reconciler_args[key] for key in kwargs_ls}
                
                if (level is not None) and (num_samples > 0):
                    # Store reconciler's memory to generate samples
                    reconciler = reconciler.fit(**kwargs)
                    fcsts_model = reconciler.predict(S=reconciler_args['S'], 
                                                     y_hat=reconciler_args['y_hat'], level=level)
                else:
                    # Memory efficient reconciler's fit_predict
                    fcsts_model = reconciler(**kwargs, level=level)

                # Parse final outputs
                Y_tilde_df = nw.concat([Y_tilde_df, nw.from_dict({recmodel_name: fcsts_model['mean'].flatten()}, native_namespace=self.native_namespace)], how="horizontal")

                if intervals_method in ['bootstrap', 'normality', 'permbu'] and level is not None:
                    level.sort()
                    lo_names = [f'{recmodel_name}-lo-{lv}' for lv in reversed(level)]
                    hi_names = [f'{recmodel_name}-hi-{lv}' for lv in level]
                    self.level_names[recmodel_name] = lo_names + hi_names
                    sorted_quantiles = np.reshape(fcsts_model['quantiles'], (len(Y_tilde_df), -1))
                    y_tilde = dict(zip(self.level_names[recmodel_name], sorted_quantiles.T))
                    Y_tilde_df = nw.concat([Y_tilde_df, nw.from_dict(y_tilde, native_namespace=self.native_namespace)], how="horizontal")

                    if num_samples > 0:
                        samples = reconciler.sample(num_samples=num_samples)
                        self.sample_names[recmodel_name] = [f'{recmodel_name}-sample-{i}' for i in range(num_samples)]
                        samples = np.reshape(samples, (len(Y_tilde_df),-1))        
                        y_tilde = dict(zip(self.sample_names[recmodel_name], samples.T))
                        Y_tilde_df = nw.concat([Y_tilde_df, nw.from_dict(y_tilde, native_namespace=self.native_namespace)], how="horizontal")
                      

                end = time.time()
                self.execution_times[f'{model_name}/{reconcile_fn_name}'] = (end - start)

        return Y_tilde_df.to_native()

    def bootstrap_reconcile(self,
                            Y_hat_df: FrameT,
                            S_df: FrameT,
                            tags: Dict[str, np.ndarray],
                            Y_df: Optional[FrameT] = None,
                            level: Optional[List[int]] = None,
                            intervals_method: str = 'normality',
                            num_samples: int = -1,
                            num_seeds: int = 1,
                            sort_df: bool = True):
        """Bootstraped Hierarchical Reconciliation Method.

        Applies N times, based on different random seeds, the `reconcile` method 
        for the different reconciliation techniques instantiated in the `reconcilers` list. 

        **Parameters:**<br>
        `Y_hat_df`: DataFrame, base forecasts with columns ['unique_id', 'ds'] and models to reconcile.<br>
        `Y_df`: DataFrame, training set of base time series with columns `['unique_id', 'ds', 'y']`.<br>
        If a class of `self.reconciles` receives `y_hat_insample`, `Y_df` must include them as columns.<br>
        `S`: DataFrame with summing matrix of size `(base, bottom)`, see [aggregate method](https://nixtla.github.io/hierarchicalforecast/utils.html#aggregate).<br>
        `tags`: Each key is a level and its value contains tags associated to that level.<br>
        `level`: positive float list [0,100), confidence levels for prediction intervals.<br>
        `intervals_method`: str, method used to calculate prediction intervals, one of `normality`, `bootstrap`, `permbu`.<br>
        `num_samples`: int=-1, if positive return that many probabilistic coherent samples.
        `num_seeds`: int=1, random seed for numpy generator's replicability.<br>
        `sort_df` : deprecated.<br>

        **Returns:**<br>
        `Y_bootstrap_df`: DataFrame, with bootstraped reconciled predictions.
        """

        # Check input's validity and sort dataframes
        Y_hat_df, S_df, Y_df, self.model_names = \
                    self._prepare_fit(Y_hat_df=Y_hat_df,
                                      S_df=S_df,
                                      Y_df=Y_df,
                                      tags=tags,
                                      intervals_method=intervals_method,
                                      sort_df=sort_df)

        # Bootstrap reconciled predictions
        Y_tilde_list = []
        for seed in range(num_seeds):
            Y_tilde_df = self.reconcile(Y_hat_df=Y_hat_df,
                                        S=S_df,
                                        tags=tags,
                                        Y_df=Y_df,
                                        level=level,
                                        intervals_method=intervals_method,
                                        num_samples=num_samples,
                                        seed=seed,
                                        sort_df=False)
            Y_tilde_df_ = nw.from_native(Y_tilde_df)
            Y_tilde_df_ = Y_tilde_df_.with_columns(nw.lit(seed).alias('seed'))

            # TODO: fix broken recmodel_names
            if seed==0:
                first_columns = Y_tilde_df_.columns
            Y_tilde_df_ = Y_tilde_df_.rename({col: first_columns[i] for i, col in enumerate(first_columns)})
            Y_tilde_list.append(Y_tilde_df_)

        Y_bootstrap_df = nw.concat(Y_tilde_list, how="vertical")

        return Y_bootstrap_df.to_native()
