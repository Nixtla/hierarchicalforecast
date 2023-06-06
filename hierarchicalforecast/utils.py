# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% auto 0
__all__ = ['HierarchicalPlot']

# %% ../nbs/utils.ipynb 2
import sys
import timeit
from itertools import chain
from typing import Callable, Dict, List, Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

plt.rcParams['font.family'] = 'serif'

# %% ../nbs/utils.ipynb 4
class CodeTimer:
    def __init__(self, name=None, verbose=True):
        self.name = " '"  + name + "'" if name else ''
        self.verbose = verbose

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start)
        if self.verbose:
            print('Code block' + self.name + \
                  ' took:\t{0:.5f}'.format(self.took) + ' seconds')

# %% ../nbs/utils.ipynb 5
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

# %% ../nbs/utils.ipynb 6
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

# %% ../nbs/utils.ipynb 7
# convert levels to output quantile names
def level_to_outputs(level:Iterable[int]):
    """ Converts list of levels into output names matching StatsForecast and NeuralForecast methods.

    **Parameters:**<br>
    `level`: int list [0,100]. Probability levels for prediction intervals.<br>

    **Returns:**<br>
    `output_names`: str list. String list with output column names.
    """
    qs = sum([[50-l/2, 50+l/2] for l in level], [])
    output_names = sum([[f'-lo-{l}', f'-hi-{l}'] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles]) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, '-median')
    
    return quantiles, output_names

# convert quantiles to output quantile names
def quantiles_to_outputs(quantiles:Iterable[float]):
    """Converts list of quantiles into output names matching StatsForecast and NeuralForecast methods.

    **Parameters:**<br>
    `quantiles`: float list [0., 1.]. Alternative to level, quantiles to estimate from y distribution.<br>

    **Returns:**<br>
    `output_names`: str list. String list with output column names.
    """
    output_names = []
    for q in quantiles:
        if q<.50:
            output_names.append(f'-lo-{np.round(100-200*q,2)}')
        elif q>.50:
            output_names.append(f'-hi-{np.round(100-200*(1-q),2)}')
        else:
            output_names.append('-median')
    return quantiles, output_names

# %% ../nbs/utils.ipynb 8
# given input array of sample forecasts and inptut quantiles/levels, 
# output a Pandas Dataframe with columns of quantile predictions
def samples_to_quantiles_df(samples:np.ndarray, 
                            unique_ids:Iterable[str], 
                            dates:Iterable, 
                            quantiles:Optional[Iterable[float]] = None,
                            level:Optional[Iterable[int]] = None, 
                            model_name:Optional[str] = "model"):
    """ Transform Samples into HierarchicalForecast input.
    Auxiliary function to create compatible HierarchicalForecast input Y_hat_df dataframe.

    **Parameters:**<br>
    `samples`: numpy array. Samples from forecast distribution of shape [n_series, n_samples, horizon].<br>
    `unique_ids`: string list. Unique identifiers for each time series.<br>
    `dates`: datetime list. List of forecast dates.<br>
    `quantiles`: float list in [0., 1.]. Alternative to level, quantiles to estimate from y distribution.<br>
    `level`: int list in [0,100]. Probability levels for prediction intervals.<br>
    `model_name`: string. Name of forecasting model.<br>

    **Returns:**<br>
    `quantiles`: float list in [0., 1.]. quantiles to estimate from y distribution .<br>
    `Y_hat_df`: pd.DataFrame. With base quantile forecasts with columns ds and models to reconcile indexed by unique_id.
    """
    
    # Get the shape of the array
    n_series, n_samples, horizon = samples.shape

    assert n_series == len(unique_ids)
    assert horizon == len(dates)
    assert (quantiles is not None) ^ (level is not None)  #check exactly one of quantiles/levels has been input

    #create initial dictionary
    forecasts_mean = np.mean(samples, axis=1).flatten()
    unique_ids = np.repeat(unique_ids, horizon)
    ds = np.tile(dates, n_series)
    data = pd.DataFrame({"unique_id":unique_ids, "ds":ds, model_name:forecasts_mean})

    #create quantiles and quantile names
    quantiles, quantile_names = level_to_outputs(level) if level is not None else quantiles_to_outputs(quantiles)
    percentiles = [quantile * 100 for quantile in quantiles]
    col_names = np.array([model_name + quantile_name for quantile_name in quantile_names])
    
    #add quantiles to dataframe
    forecasts_quantiles = np.percentile(samples, percentiles, axis=1)

    forecasts_quantiles = np.transpose(forecasts_quantiles, (1,2,0)) # [Q,H,N] -> [N,H,Q]
    forecasts_quantiles = forecasts_quantiles.reshape(-1,len(quantiles))

    df = pd.DataFrame(data=forecasts_quantiles, 
                      columns=col_names)
    
    return quantiles, pd.concat([data,df], axis=1).set_index('unique_id')

# %% ../nbs/utils.ipynb 11
def _to_summing_matrix(S_df: pd.DataFrame):
    """Transforms the DataFrame `df` of hierarchies to a summing matrix S."""
    categories = [S_df[col].unique() for col in S_df.columns]
    cat_sizes = [len(cats) for cats in categories]
    idx_bottom = np.argmax(cat_sizes)
    cats_bottom = categories[idx_bottom]
    encoder = OneHotEncoder(categories=categories, sparse=False, dtype=np.float32)
    S = encoder.fit_transform(S_df).T
    S = pd.DataFrame(S, index=chain(*categories), columns=cats_bottom)
    tags = dict(zip(S_df.columns, categories))
    return S, tags

# %% ../nbs/utils.ipynb 12
def aggregate_before(df: pd.DataFrame,
              spec: List[List[str]],
              agg_fn: Callable = np.sum):
    """Utils Aggregation Function.

    Aggregates bottom level series contained in the pd.DataFrame `df` according 
    to levels defined in the `spec` list applying the `agg_fn` (sum, mean).<br>

    **Parameters:**<br>
    `df`: pd.DataFrame with columns `['ds', 'y']` and columns to aggregate.<br>
    `spec`: List of levels. Each element of the list contains a list of columns of `df` to aggregate.<br>
    `agg_fn`: Function used to aggregate `'y'`.<br>

    **Returns:**<br>
    `Y_df, S, tags`: tuple with hierarchically structured series `Y_df` ($\mathbf{y}_{[a,b]}$),
    summing matrix `S`, and hierarchical aggregation indexes `tags`.
    """
    max_len_idx = np.argmax([len(hier) for hier in spec])
    bottom_comb = spec[max_len_idx]
    df_hiers = []
    for hier in spec:
        df_hier = df.groupby(hier + ['ds'])['y'].apply(agg_fn).reset_index()
        df_hier['unique_id'] = df_hier[hier].agg('/'.join, axis=1)
        if hier == bottom_comb:
            bottom_hier = df_hier['unique_id'].unique()
        df_hiers.append(df_hier)
    df_hiers = pd.concat(df_hiers)
    S_df = df_hiers[['unique_id'] + bottom_comb].drop_duplicates().reset_index(drop=True)
    S_df = S_df.set_index('unique_id')
    S_df = S_df.fillna('agg')
    hiers_cols = []
    for hier in spec:
        hier_col = '/'.join(hier)
        S_df[hier_col] = S_df[hier].agg('/'.join, axis=1)
        hiers_cols.append(hier_col)
    Y_df = df_hiers[['unique_id', 'ds', 'y']].set_index('unique_id')
    
    # Aggregations constraints S definition
    S, tags = _to_summing_matrix(S_df.loc[bottom_hier, hiers_cols])
    return Y_df, S, tags

# %% ../nbs/utils.ipynb 13
def numpy_balance(*arrs):
    """
    Fast NumPy implementation of balance function.
    The function creates all the interactions between
    the NumPy arrays provided.
    **Parameters:**<br>
    `arrs`: NumPy arrays.<br>
    **Returns:**<br>
    `out`: NumPy array.<br>
    """
    N = len(arrs)
    out =  np.transpose(np.meshgrid(*arrs, indexing='ij'),
                        np.roll(np.arange(N + 1), -1)).reshape(-1, N)
    return out

def _to_summing_dataframe(df: pd.DataFrame,
                          spec: List[List[str]]):
    #------------------------------- Wrangling -----------------------------#
    # Keep unique levels, preserving first aparison order
    all_levels = list(chain.from_iterable(spec))
    all_levels = [*dict.fromkeys(all_levels)]

    # Create hierarchical labels
    S_df = df[all_levels].copy()
    S_df = S_df.drop_duplicates()

    max_len_idx = np.argmax([len(hier) for hier in spec])
    bottom_comb = spec[max_len_idx]
    hiers_cols = []
    df = df.copy()
    for hier in spec:
        if hier == bottom_comb:
            hier_col = 'unique_id'
            bottom_col = '/'.join(hier)
            df['unique_id'] = df[hier].agg('/'.join, axis=1)
        else:
            hier_col = '/'.join(hier) 
        S_df[hier_col] = S_df[hier].agg('/'.join, axis=1)
        hiers_cols.append(hier_col)
    S_df = S_df.sort_values(by=bottom_comb)
    S_df = S_df[hiers_cols]

    #------------------------------- Encoding ------------------------------#
    # One hot encode only aggregate levels
    # TODO: option to only operate with sparse matrices
    bottom_ids = list(S_df.unique_id)
    del S_df['unique_id']
    categories = [S_df[col].unique() for col in S_df.columns]
    tags = dict(zip(S_df.columns, categories))
    tags[bottom_col] = bottom_ids

    encoder = OneHotEncoder(categories=categories,
                            sparse=False, dtype=np.float32)
    S = encoder.fit_transform(S_df).T
    S = np.concatenate([S, np.eye(len(bottom_ids), dtype=np.float32)], axis=0)
    S_df = pd.DataFrame(S, columns=bottom_ids,
                        index=list(chain(*categories))+bottom_ids)

    # Match index ordering of S_df and collapse df to Y_bottom_df
    Y_bottom_df = df.copy()
    Y_bottom_df = Y_bottom_df.groupby(['unique_id', 'ds'])['y'].sum().reset_index()
    Y_bottom_df.unique_id = Y_bottom_df.unique_id.astype('category')
    Y_bottom_df.unique_id = Y_bottom_df.unique_id.cat.set_categories(S_df.columns)
    return Y_bottom_df, S_df, tags

def aggregate(df: pd.DataFrame,
              spec: List[List[str]],
              is_balanced: bool=False):
    """ Utils Aggregation Function.
    Aggregates bottom level series contained in the pd.DataFrame `df` according 
    to levels defined in the `spec` list applying the `agg_fn` (sum, mean).
    **Parameters:**<br>
    `df`: pd.DataFrame with columns `['ds', 'y']` and columns to aggregate.<br>
    `spec`: List of levels. Each element of the list contains a list of columns of `df` to aggregate.<br>
    `is_balanced`: bool=False, wether `Y_bottom_df` is balanced, if not we balance.<br>
    **Returns:**<br>
    `Y_df, S_df, tags`: tuple with hierarchically structured series `Y_df` ($\mathbf{y}_{[a,b]}$),
    summing dataframe `S_df`, and hierarchical aggregation indexes `tags`.
    """
    
    #Ensure no null values
    if df.isnull().values.any():
        raise Exception('`df` contains null values')
            
    #-------------------------------- Wrangling --------------------------------#
    # constraints S_df and collapsed Y_bottom_df with 'unique_id'
    Y_bottom_df, S_df, tags = _to_summing_dataframe(df=df, spec=spec)

    # Create balanced/sorted dataset for numpy aggregation (nan=0)
    # TODO: investigate potential memory speed tradeoff
    if not is_balanced:
        dates         = Y_bottom_df['ds'].unique()
        balanced_prod = numpy_balance(S_df.columns, dates)
        balanced_df   = pd.DataFrame(balanced_prod, columns=['unique_id', 'ds'])
        balanced_df['ds'] = balanced_df['ds'].astype(Y_bottom_df['ds'].dtype)

        Y_bottom_df.set_index(['unique_id', 'ds'], inplace=True)
        balanced_df.set_index(['unique_id', 'ds'], inplace=True)
        balanced_df   = balanced_df.merge(Y_bottom_df[['y']],
                                          how='left', left_on=['unique_id', 'ds'],
                                          right_index=True).reset_index()
        Y_bottom_df.reset_index(inplace=True)
    else:
        dates       = Y_bottom_df['ds'].unique()
        balanced_df = Y_bottom_df.copy()

    #------------------------------- Aggregation -------------------------------#
    n_agg = S_df.shape[0] - S_df.shape[1]
    Agg = S_df.values[:n_agg, :]
    y_bottom = balanced_df.y.values

    y_bottom = y_bottom.reshape(len(S_df.columns), len(dates))
    y_bottom_mask = np.isnan(y_bottom)
    y_agg = Agg @ np.nan_to_num(y_bottom)
    y_agg_mask = Agg @ y_bottom_mask

    # Create long format hierarchical dataframe
    y_agg = y_agg.flatten()
    y_agg[y_agg_mask.flatten() > 1] = np.nan
    y_bottom = y_bottom.flatten()
    Y_df = pd.DataFrame(dict(
                unique_id = np.repeat(S_df.index, len(dates)),
                ds = np.tile(dates, len(S_df.index)),
                y = np.concatenate([y_agg, y_bottom], axis=0)))
    Y_df = Y_df.set_index('unique_id').dropna()
    return Y_df, S_df, tags

# %% ../nbs/utils.ipynb 22
class HierarchicalPlot:
    """ Hierarchical Plot

    This class contains a collection of matplotlib visualization methods, suited for small
    to medium sized hierarchical series.

    **Parameters:**<br>
    `S`: pd.DataFrame with summing matrix of size `(base, bottom)`, see [aggregate function](https://nixtla.github.io/hierarchicalforecast/utils.html#aggregate).<br>
    `tags`: np.ndarray, with hierarchical aggregation indexes, where 
        each key is a level and its value contains tags associated to that level.<br><br>
    """
    def __init__(self,
                 S: pd.DataFrame,
                 tags: Dict[str, np.ndarray]):
        self.S = S
        self.tags = tags

    def plot_summing_matrix(self):
        """ Summation Constraints plot
        
        This method simply plots the hierarchical aggregation
        constraints matrix $\mathbf{S}$.
        """
        plt.figure(num=1, figsize=(4, 6), dpi=80, facecolor='w')
        plt.spy(self.S)
        plt.show()
        plt.close()

    def plot_series(self,
                    series: str,
                    Y_df: Optional[pd.DataFrame] = None,
                    models: Optional[List[str]] = None,
                    level: Optional[List[int]] = None):
        """ Single Series plot

        **Parameters:**<br>
        `series`: str, string identifying the `'unique_id'` any-level series to plot.<br>
        `Y_df`: pd.DataFrame, hierarchically structured series ($\mathbf{y}_{[a,b]}$). 
                It contains columns `['unique_id', 'ds', 'y']`, it may have `'models'`.<br>
        `models`: List[str], string identifying filtering model columns.
        `level`: float list 0-100, confidence levels for prediction intervals available in `Y_df`.<br>

        **Returns:**<br>
        Single series plot with filtered models and prediction interval level.<br><br>
        """
        if series not in self.S.index:
            raise Exception(f'time series {series} not found')
        fig, ax = plt.subplots(1, 1, figsize = (20, 7))
        df_plot = Y_df.loc[series].set_index('ds')
        cols = models if models is not None else df_plot.columns
        cols_wo_levels = [col for col in cols if ('-lo-' not in col and '-hi-' not in col)]
        cmap = plt.cm.get_cmap("tab10", 10)
        cmap = [cmap(i) for i in range(10)][:len(cols_wo_levels)]
        cmap_dict = dict(zip(cols_wo_levels, cmap))
        for col in cols_wo_levels:
            ax.plot(df_plot[col], linewidth=2, label=col, color=cmap_dict[col])
            if level is not None and col != 'y':
                for lv in level:
                    if f'{col}-lo-{lv}' not in df_plot.columns:
                        # if model
                        # doesnt have levels
                        continue
                    ax.fill_between(
                        df_plot.dropna().index, 
                        df_plot[f'{col}-lo-{lv}'].dropna().values, 
                        df_plot[f'{col}-hi-{lv}'].dropna().values,
                        alpha=-lv/100 + 1,
                        color=cmap_dict[col],
                        label=f'{col}_level_{lv}'
                    )
        ax.set_title(f'{series} Forecast', fontsize=22)
        ax.set_xlabel('Timestamp [t]', fontsize=20)
        ax.legend(prop={'size': 15})
        ax.grid()
        ax.xaxis.set_major_locator(
            plt.MaxNLocator(min(max(len(df_plot) // 10, 1), 10))
        )
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(20)
                    
    def plot_hierarchically_linked_series(self,
                                          bottom_series: str,
                                          Y_df: Optional[pd.DataFrame] = None,
                                          models: Optional[List[str]] = None,
                                          level: Optional[List[int]] = None):
        """ Hierarchically Linked Series plot

        **Parameters:**<br>
        `bottom_series`: str, string identifying the `'unique_id'` bottom-level series to plot.<br>
        `Y_df`: pd.DataFrame, hierarchically structured series ($\mathbf{y}_{[a,b]}$). 
                It contains columns ['unique_id', 'ds', 'y'] and models. <br>
        `models`: List[str], string identifying filtering model columns.
        `level`: float list 0-100, confidence levels for prediction intervals available in `Y_df`.<br>

        **Returns:**<br>
        Collection of hierarchilly linked series plots associated with the `bottom_series`
        and filtered models and prediction interval level.<br><br>
        """
        if bottom_series not in self.S.columns:
            raise Exception(f'bottom time series {bottom_series} not found')
        linked_series = self.S[bottom_series].loc[lambda x: x == 1.].index
        fig, axs = plt.subplots(len(linked_series), 1, figsize=(20, 2 * len(linked_series)))
        cols = models if models is not None else Y_df.drop(['ds'], axis=1)
        cols_wo_levels = [col for col in cols if ('-lo-' not in col and '-hi-' not in col)]
        cmap = plt.cm.get_cmap("tab10", 10)
        cmap = [cmap(i) for i in range(10)][:len(cols_wo_levels)]
        cmap_dict = dict(zip(cols_wo_levels, cmap))
        for idx, series in enumerate(linked_series):
            df_plot = Y_df.loc[[series]].set_index('ds')
            for col in cols_wo_levels:
                axs[idx].plot(df_plot[col], linewidth=2, label=col, color=cmap_dict[col])
                if level is not None and col != 'y':
                    for lv in level:
                        if f'{col}-lo-{lv}' not in df_plot.columns:
                            # if model
                            # doesnt have levels
                            continue
                        axs[idx].fill_between(
                            df_plot.dropna().index, 
                            df_plot[f'{col}-lo-{lv}'].dropna().values, 
                            df_plot[f'{col}-hi-{lv}'].dropna().values,
                            alpha=-lv/100 + 1,
                            color=cmap_dict[col],
                            label=f'{col}_level_{lv}'
                        )
            axs[idx].set_title(f'{series}', fontsize=10)
            axs[idx].grid()
            axs[idx].get_xaxis().label.set_visible(False)
            axs[idx].legend().set_visible(False)
            axs[idx].xaxis.set_major_locator(
                plt.MaxNLocator(min(max(len(df_plot) // 10, 1), 10))
            )
            for label in (axs[idx].get_xticklabels() + axs[idx].get_yticklabels()):
                label.set_fontsize(10)
        plt.subplots_adjust(hspace=0.4)
        handles, labels = axs[0].get_legend_handles_labels()
        kwargs = dict(loc='lower center', 
                      prop={'size': 10}, 
                      bbox_to_anchor=(0, 0.05, 1, 1))
        if sys.version_info.minor > 7:
            kwargs['ncols'] = np.max([2, np.ceil(len(labels) / 2)])
        fig.legend(handles, labels, **kwargs)

    def plot_hierarchical_predictions_gap(self,
                                          Y_df: pd.DataFrame,
                                          models: Optional[List[str]] = None,
                                          xlabel: Optional=None,
                                          ylabel: Optional=None,
                                          ):
        """ Hierarchically Predictions Gap plot

        **Parameters:**<br>
        `Y_df`: pd.DataFrame, hierarchically structured series ($\mathbf{y}_{[a,b]}$). 
                It contains columns ['unique_id', 'ds', 'y'] and models. <br>
        `models`: List[str], string identifying filtering model columns.
        `xlabel`: str, string for the plot's x axis label.
        `ylable`: str, string for the plot's y axis label.

        **Returns:**<br>
        Plots of aggregated predictions at different levels of the hierarchical structure.
        The aggregation is performed according to the tag levels see 
        [aggregate function](https://nixtla.github.io/hierarchicalforecast/utils.html).<br><br>
        """
        # Parse predictions dataframe
        horizon_dates = Y_df['ds'].unique()
        cols = models if models is not None else Y_df.drop(['ds', 'y'], axis=1).columns
        
        # Plot predictions across tag levels
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if 'y' in Y_df.columns:
            idx_top = self.S.sum(axis=1).idxmax()
            y_plot = Y_df.loc[idx_top].y.values
            plt.plot(horizon_dates, y_plot, label='True')

        ys = []
        for tag in self.tags:
            y_plot = sum([Y_df[cols].loc[Y_df.index == idx].values \
                          for idx in self.tags[tag]])
            plt.plot(horizon_dates, y_plot, label=f'Level: {tag}')
            
            ys.append(y_plot[:,None])

        plt.title('Predictions Accumulated Difference')
        if ylabel is not None:
            plt.ylabel(ylabel)
        if xlabel is not None:
            plt.xlabel(xlabel)

        plt.legend()
        plt.grid()
        plt.show()
