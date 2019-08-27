import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.base import clone

def get_numeric_cols(X, cat_cols):
    if cat_cols is None:
        return "all"
    elif len(cat_cols) < X.shape[1]:
        if (isinstance(X, pd.DataFrame)) and (isinstance(cat_cols[0], str)):
            numeric_cols = list(set(X.columns).difference(cat_cols))
        else:
            numeric_cols = np.array(list(set(range(X.shape[1])).difference(cat_cols)))

def safe_column_indexing(X, columns):
    if columns is None:
        return
    if isinstance(columns, str) and (columns == "all"):
        return X
    columns = np.array(columns)
    columns_type = "string" if isinstance(columns[0], str) else "int"

    if isinstance(X, pd.DataFrame):
        if columns_type == "string":
            return X.loc[:, columns]
        else:
            return X.iloc[:, columns]
    else:
        return X[:, columns]

def _get_params(trial, params):
    param_dict = {}
    for (key, value) in params.items():
        if "optuna.distributions" in str(type(value)):
            param_dict[key] = trial._suggest(key, value)
        else:
            param_dict[key] = params[key]
    return param_dict

def _init_pipeline(pipeline, pipe_params, trial):
    if pipeline is not None:
        final_pipeline = clone(pipeline)
        if pipe_params is not None:
            pipe_params = _get_params(trial, pipe_params)
            final_pipeline.set_params(**pipe_params)
        return final_pipeline
    return None

def contains_nan(X):
    if isinstance(X, pd.DataFrame):
        return pd.isnull(X).values.any()
    elif sp.issparse(X):
        return pd.isnull(X.data).any()
    else:
        return pd.isnull(X).any() #numpy
