from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder, BinaryEncoder, CatBoostEncoder, TargetEncoder, WOEEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.base import clone
from sklearn.utils import safe_indexing

import scipy.sparse as sp
from .utils import *

def add_to_pipe(pipe, name, step, cols=None, cols_name=None,
                remainder="passthrough", n_jobs=1):
    if step is None:
        return pipe
    if cols is not None:
        step = ColumnTransformer(transformers=[(name, step, cols)],
                                 remainder=remainder,
                                 sparse_threshold=0,
                                 n_jobs=n_jobs)

        name = cols_name if cols_name is not None else name
    if pipe is None:
        pipe_res = Pipeline([(name, step)])
    else:
        pipe_res = clone(pipe)
        step_names = [step[0] for step in pipe_res.steps]
        if name not in step_names:
            pipe_res.steps.append([name, step])
    return pipe_res

def CatImputer(X, cat_cols, tags, trial, random_state):
    if (contains_nan(safe_column_indexing(X, cat_cols))) and (tags["handles categorical nan"] == False):
        imputer = SimpleImputer(strategy="constant", fill_value="unknown")
    return imputer

def CatEncoder(X, cat_cols, tags, objective_type, trial, n_classes, random_state):
    if tags["handles categorical"] == False:
        large_threshold = 6
        #TODO: handle numpy arrays with categorical?
        #TODO: handle multiclass / Regression
        large_cardinal_cats = [col for col in X[cat_cols].columns if X[col].nunique() > large_threshold]
        small_cardinal_cats = [col for col in X[cat_cols].columns if X[col].nunique() <= large_threshold]

        enc_pipe = None
        cat_enc_types = ["binary", "catboost", "target"]

        if small_cardinal_cats is not None:
            enc_pipe = add_to_pipe(enc_pipe, "ohe", OneHotEncoder(cols=small_cardinal_cats, drop_invariant=True))

        if large_cardinal_cats is not None:
            if (objective_type == "classification" and n_classes == 1):
                cat_enc_types.append("woe")

            cat_enc_type = trial.suggest_categorical("cat_enc_type", cat_enc_types)

            if cat_enc_type == "binary":
                # mapping = get_mapping(X, large_cardinal_cats)
                enc = BinaryEncoder(cols=large_cardinal_cats,
                                    # mapping=mapping
                                    )

            elif cat_enc_type == "woe":
                enc = WOEEncoder(cols=large_cardinal_cats, drop_invariant=True)

            elif cat_enc_type == "target":
                min_samples_leaf = 10  # TODO: calculate percentage or something else
                enc = TargetEncoder(min_samples_leaf=min_samples_leaf,
                                    cols=large_cardinal_cats)

            else: # catboost
                enc = CatBoostEncoder(cols=large_cardinal_cats,
                                      random_state=random_state)  # TODO: replace SEED
                # TODO: permute to the dataset beforehand

            enc_pipe = add_to_pipe(enc_pipe, cat_enc_type + "_encoder", enc)
    return enc_pipe

def NumericImputer(X, numeric_cols, trial, tags):
    imputer = None
    if (contains_nan(safe_column_indexing(X, numeric_cols))):
        if (not sp.issparse(X) and tags["handles numeric nan"] == False) or\
                (sp.issparse(X) and tags["nan value when sparse"] != np.nan):
            imputer = SimpleImputer(strategy="median", add_indicator=True)
    return imputer

def Scaler(X, numeric_cols, trial, tags):
    scaler = None

    center = True
    scaler_types = ["robust", "standard", "minmax", "maxabs"]
    if sp.issparse(X):
        scaler_types.remove("minmax")
        center = False

    if (numeric_cols is not None) and tags["sensitive to feature scaling"]:
        scaler_type = trial.suggest_categorical("scaler", scaler_types)
        if scaler_type == "standard":
            scaler = StandardScaler(with_mean=center)
        elif scaler_type == "robust":
            scaler = RobustScaler(with_centering=center)
        elif scaler_type == "maxabs":
            scaler = MaxAbsScaler()
        else:  # minmax
            scaler = MinMaxScaler()
    return scaler

# if numeric_transforms is not None:
#     if cat_cols is None:
#         numeric_pipe = add_to_pipe(numeric_pipe, "numeric_pipe", numeric_transforms)
#     else:
#         numeric_pipe = add_to_pipe(numeric_pipe, "numeric_pipe", numeric_transforms,
#                                    cols=numeric_cols, cols_name="numeric_transforms", remainder="drop")
#
# pipeline = None

# if cat_pipe is not None and numeric_pipe is not None:
#     pipeline = add_to_pipe(pipeline, "cat_num_pipes", FeatureUnion([("cat", cat_pipe), ("numeric", numeric_pipe)]))
# elif cat_pipe is not None:
#     pipeline = cat_pipe
# elif numeric_pipe is not None:
#     pipeline = numeric_pipe

#pipline = before, default, after
#cat_imp, cat_enc, num_imp, num_scale


#1. add first step
#2. for cat:
#       add all steps
#   for numeric:
#       add all steps
#   join whatever is there (Union)

#TODO: move before objective
# ## get numeric columns
# numeric_cols = None
# if cat_cols is None:
#     if isinstance(X, pd.DataFrame):
#         numeric_cols = X.columns
#     else:
#         numeric_cols = list(range(X.shape[1]))
# else:
#     if len(cat_cols) < X.shape[1]:
#         if (isinstance(X, pd.DataFrame)) and (isinstance(cat_cols[0], str)):
#             numeric_cols = list(set(X.columns).difference(cat_cols))
#         else:
#             numeric_cols = np.array(list(set(range(X.shape[1])).difference(cat_cols)))
#
# if (cat_cols is not None):
#     if tags["handles categorical"] == False:
#
#         # TODO: fix category encoders dealing with string labels in y
#         cat_enc_types = ["binary", "catboost", "woe", "target"]
#         large_threshold = 6
#         large_cardinal_cats = [col for col in X[cat_cols].columns if X[col].nunique() > large_threshold]
#         small_cardinal_cats = [col for col in X[cat_cols].columns if X[col].nunique() <= large_threshold]
#         if small_cardinal_cats is not None:
#             cat_transforms = add_to_pipe(cat_transforms, "ohe",
#                                          OneHotEncoder(cols=small_cardinal_cats, drop_invariant=True))
#         if large_cardinal_cats is not None:
#             if (self.objective_type == "classification" and len(self.y_stats) > 2): #multiclass
#                 cat_enc_types = ["binary"]
#             cat_enc_type = trial.suggest_categorical("cat_enc_type", cat_enc_types)
#             if cat_enc_type == "binary":
#                 # mapping = get_mapping(X, large_cardinal_cats)
#                 enc = BinaryEncoder(cols=large_cardinal_cats,
#                                     drop_invariant=True,
#                                     # mapping=mapping
#                                     )
#             elif cat_enc_type == "woe":  # slow!
#                 enc = WOEEncoder(cols=large_cardinal_cats, drop_invariant=True)
#             elif cat_enc_type == "target":
#                 min_samples_leaf = 10  # TODO: calculate percentage or something else
#                 enc = TargetEncoder(min_samples_leaf=min_samples_leaf,
#                                     cols=large_cardinal_cats,
#                                     drop_invariant=True)
#             else: # catboost
#                 enc = CatBoostEncoder(cols=large_cardinal_cats,
#                                       drop_invariant=True,
#                                       random_state=random_state)  # TODO: replace SEED
#                 # TODO: permute to the dataset beforehand
#             cat_transforms = add_to_pipe(cat_transforms, cat_enc_type + "_encoder", enc)
#             cat_transform_name = "cat_encoder"
#     #TODO: move contains_nan before Objective
#     elif (contains_nan(X[cat_cols])) and (tags["handles categorical nan"] == False):
#         cat_transforms = ("imputer", SimpleImputer(strategy="constant", fill_value="unknown"))
#         cat_transform_name = "imputer"
#
# if cat_transforms is not None:
#     if numeric_cols is None:
#         cat_pipe = add_to_pipe(cat_pipe, cat_transform_name, cat_transforms)
#     else:
#         cat_pipe = add_to_pipe(cat_pipe, cat_transform_name, cat_transforms, cols=cat_cols,
#                                cols_name="cat_transforms", remainder="drop")
#
# if numeric_cols is not None:
#     if cat_cols is None:
#         X_numeric = X
#     elif isinstance(numeric_cols[0], str):
#         X_numeric = X[numeric_cols]
#     else:
#         col_indices = np.asarray([index for (index, name) in enumerate(X.columns) if name in numeric_cols])
#         X_numeric = X[:, col_indices]
#
# numeric_pipe = None
# numeric_transforms = None
# if (contains_nan(X_numeric)):
#     if (sp.issparse(X) and tags["nan value when sparse"] != np.nan) or \
#             (not sp.issparse(X) and tags["handles numeric nan"] == False):
#         imputer = SimpleImputer(strategy="median", add_indicator=True)
#         numeric_transforms = add_to_pipe(numeric_transforms, "imputer", imputer)
#
# center = True
# scaler_types = ["robust", "standard", "minmax", "maxabs"]
# if issparse(X):
#     scaler_types.remove("minmax")
#     center = False
#
# if (numeric_cols is not None) and tags["sensitive to feature scaling"]:
#     scaler_type = trial.suggest_categorical("scaler", scaler_types)
#     if scaler_type == "standard":
#         scaler = StandardScaler(with_mean=center)
#     elif scaler_type == "robust":
#         scaler = RobustScaler(with_centering=center)
#     elif scaler_type == "maxabs":
#         scaler = MaxAbsScaler()
#     else:  # minmax
#         scaler = MinMaxScaler()
#     numeric_transforms = add_to_pipe(numeric_transforms, scaler_type + "_scaler", scaler)
#
# if numeric_transforms is not None:
#     if cat_cols is None:
#         numeric_pipe = add_to_pipe(numeric_pipe, "numeric_pipe", numeric_transforms)
#     else:
#         numeric_pipe = add_to_pipe(numeric_pipe, "numeric_pipe", numeric_transforms,
#                                    cols=numeric_cols, cols_name="numeric_transforms", remainder="drop")
#
# pipeline = None
#
# if cat_pipe is not None and numeric_pipe is not None:
#     pipeline = add_to_pipe(pipeline, "cat_num_pipes", FeatureUnion([("cat", cat_pipe), ("numeric", numeric_pipe)]))
# elif cat_pipe is not None:
#     pipeline = cat_pipe
# elif numeric_pipe is not None:
#     pipeline = numeric_pipe