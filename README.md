# Nixtla &nbsp;[![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlacommunity/shared_invite/zt-1pmhan9j5-F54XR20edHk0UtYAPcW4KQ)

<div align="center">
<img src="https://raw.githubusercontent.com/Nixtla/neuralforecast/main/nbs/imgs_indx/logo_mid.png">
<h1 align="center">Hierarchical Forecast 👑</h1>
<h3 align="center">Probabilistic hierarchical forecasting with statistical and econometric methods</h3>
    
[![CI](https://github.com/Nixtla/hierarchicalforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Nixtla/hierarchicalforecast/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/hierarchicalforecast)](https://pypi.org/project/hierarchicalforecast/)
[![PyPi](https://img.shields.io/pypi/v/hierarchicalforecast?color=blue)](https://pypi.org/project/hierarchicalforecast/)
[![conda-nixtla](https://img.shields.io/conda/vn/conda-forge/hierarchicalforecast?color=seagreen&label=conda)](https://anaconda.org/conda-forge/hierarchicalforecast)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Nixtla/hierarchicalforecast/blob/main/LICENSE)
    
**HierarchicalForecast** offers a collection of cross-sectional and temporal reconciliation methods, including `BottomUp`, `TopDown`, `MiddleOut`, `MinTrace` and `ERM`, as well as probabilistic coherent prediction methods such as `Normality`, `Bootstrap`, and `PERMBU`.
</div>

## 📚 Intro
A vast amount of time series datasets are organized into structures with different levels or hierarchies of aggregation. Examples include cross-sectional aggregations such as categories, brands, or geographical groupings, or temporal aggregations such as weeks, months or years. Coherent forecasts across levels are necessary for consistent decision-making and planning. Hierachical Forecast offers different reconciliation methods that render coherent forecasts across cross-sectional and temporal hierachies. 

## 🎊 Features 

* Classic reconciliation methods:
    - `BottomUp`: Simple addition to the upper levels.
    - `TopDown`: Distributes the top levels forecasts trough the hierarchies.
* Alternative reconciliation methods:
    - `MiddleOut`: It anchors the base predictions in a middle level. The levels above the base predictions use the bottom-up approach, while the levels below use a top-down.
    - `MinTrace`: Minimizes the total forecast variance of the space of coherent forecasts, with the Minimum Trace reconciliation.
    - `ERM`: Optimizes the reconciliation matrix minimizing an L1 regularized objective.
* Probabilistic coherent methods:
    - `Normality`: Uses MinTrace variance-covariance closed form matrix under a normality assumption.
    - `Bootstrap`: Generates distribution of hierarchically reconciled predictions using Gamakumara's bootstrap approach.
    - `PERMBU`: Reconciles independent sample predictions by reinjecting multivariate dependence with estimated rank permutation copulas, and performing a Bottom-Up aggregation.
* Temporal reconciliation methods:
    - All reconciliation methods (except for the insample methods) are available to use with temporal hierarchies too.

Missing something? Please open an issue here or write us in [![Slack](https://img.shields.io/badge/Slack-4A154B?&logo=slack&logoColor=white)](https://join.slack.com/t/nixtlaworkspace/shared_invite/zt-135dssye9-fWTzMpv2WBthq8NK0Yvu6A)

## 📖 Why? 

**Short**: We want to contribute to the ML field by providing reliable baselines and benchmarks for hierarchical forecasting task in industry and academia. Here's the complete [paper](https://arxiv.org/abs/2207.03517).

**Verbose**: `HierarchicalForecast` integrates publicly available processed datasets, evaluation metrics, and a curated set of standard statistical baselines. In this library we provide usage examples and references to extensive experiments where we showcase the baseline's use and evaluate the accuracy of their predictions. With this work, we hope to contribute to Machine Learning forecasting by bridging the gap to statistical and econometric modeling, as well as providing tools for the development of novel hierarchical forecasting algorithms rooted in a thorough comparison of these well-established models. We intend to continue maintaining and increasing the repository, promoting collaboration across the forecasting community.

## 💻 Installation

We recommend using `uv` as Python package manager, for which you can find installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/). You can then install `HierarchicalForecast` with:

```python
uv pip install hierarchicalforecast
```

Alternatively, you can use the Python package index [pip](https://pypi.org) directly:

```python
pip install hierarchicalforecast
```


## 🧬 How to use

The following example needs `statsforecast` and `datasetsforecast` as additional packages. If not installed, install it via your preferred method, e.g. `pip install statsforecast datasetsforecast`.
The `datasetsforecast` library allows us to download hierarhical datasets and we will use `statsforecast` to compute the base forecasts to be reconciled.

You can open a complete example in Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nixtla/hierarchicalforecast/blob/main/nbs/examples/TourismSmall.ipynb)

Minimal Example:
```python
# !pip install -U numba statsforecast datasetsforecast
import numpy as np
import pandas as pd

#obtain hierarchical dataset
from datasetsforecast.hierarchical import HierarchicalData

# compute base forecast no coherent
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive

#obtain hierarchical reconciliation methods and evaluation
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.evaluation import evaluate
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
from utilsforecast.losses import mse

# Load TourismSmall dataset
Y_df, S_df, tags = HierarchicalData.load('./data', 'TourismSmall')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
S_df = S_df.reset_index(names="unique_id")

#split train/test sets
Y_test_df  = Y_df.groupby('unique_id').tail(4)
Y_train_df = Y_df.drop(Y_test_df.index)

# Compute base auto-ARIMA predictions
fcst = StatsForecast(models=[AutoARIMA(season_length=4), Naive()],
                     freq='QE', n_jobs=-1)
Y_hat_df = fcst.forecast(df=Y_train_df, h=4)

# Reconcile the base predictions
reconcilers = [
    BottomUp(),
    TopDown(method='forecast_proportions'),
    MiddleOut(middle_level='Country/Purpose/State',
              top_down_method='forecast_proportions')
]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_train_df,
                          S_df=S_df, tags=tags)
```

### Evaluation
Assumes you have a test dataframe.

```python
df = Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'])
evaluation = evaluate(df = df,
                      tags = tags,
                      metrics = [mse],
                      benchmark = "Naive")
```

## 📖 Documentation
Here is a link to the [documentation](https://nixtlaverse.nixtla.io/hierarchicalforecast/index.html).

## 📃 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE) file for details.

## 🏟 HTS projects
In the R ecosystem, we recommend checking out [fable](http://fable.tidyverts.org/), and the now-retired [hts](https://github.com/earowang/hts).
In Python we want to acknowledge the following libraries [hiere2e](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021), [hierts](https://github.com/elephaint/hierts), [sktime](https://github.com/sktime/sktime-tutorial-pydata-berlin-2022), [darts](https://github.com/unit8co/darts), [pyhts](https://github.com/AngelPone/pyhts), [scikit-hts](https://github.com/carlomazzaferro/scikit-hts).

## 📚 References and Acknowledgements
This work is highly influenced by the fantastic work of previous contributors and other scholars who previously proposed the reconciliation methods presented here. We want to highlight the work of Rob Hyndman, George Athanasopoulos, Shanika L. Wickramasuriya, Souhaib Ben Taieb, and Bonsoo Koo. For a full reference link, please visit the Reference section of this [paper](https://arxiv.org/pdf/2207.03517.pdf).
We encourage users to explore this [literature review](https://otexts.com/fpp3/hierarchical-reading.html).

## 🙏 How to cite
If you enjoy or benefit from using these Python implementations, a citation to this [hierarchical forecasting reference paper](https://arxiv.org/abs/2207.03517) will be greatly appreciated.
```bibtex
@article{olivares2024hierarchicalforecastreferenceframeworkhierarchical,
      title={HierarchicalForecast: A Reference Framework for Hierarchical Forecasting in Python}, 
      author={Kin G. Olivares and Azul Garza and David Luo and Cristian Challú and Max Mergenthaler and Souhaib Ben Taieb and Shanika L. Wickramasuriya and Artur Dubrawski},
      year={2024},
      eprint={2207.03517},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2207.03517}, 
}
```
