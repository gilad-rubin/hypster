from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder, BinaryEncoder, CatBoostEncoder, TargetEncoder, WOEEncoder
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

def CatEncoder(X, cat_cols, tags, estimator_name, objective_type, trial, n_classes, random_state):
    if tags["handles categorical"] == False:
        large_threshold = 6
        #TODO: handle numpy arrays with categorical?
        #TODO: handle multiclass / Regression
        if isinstance(X, pd.DataFrame) and isinstance(cat_cols[0], str):
            large_cardinal_cats = [col for col in X[cat_cols].columns if X[col].nunique() > large_threshold]
            small_cardinal_cats = [col for col in X[cat_cols].columns if X[col].nunique() <= large_threshold]
        elif isinstance(X, pd.DataFrame):
            large_cardinal_cats = [col for col in cat_cols if len(np.unique(X.iloc[:,col])) > large_threshold]
            small_cardinal_cats = [col for col in cat_cols if len(np.unique(X.iloc[:,col])) <= large_threshold]
        else:
            large_cardinal_cats = [col for col in cat_cols if len(np.unique(X[:,col])) > large_threshold]
            small_cardinal_cats = [col for col in cat_cols if len(np.unique(X[:,col])) <= large_threshold]

        enc_pipe = None
        cat_enc_types = ["target", "catboost"] #TODO: add "binary" and fix error that kiils terminal

        if small_cardinal_cats is not None:
            enc_pipe = add_to_pipe(enc_pipe, "ohe", OneHotEncoder(cols=small_cardinal_cats, drop_invariant=True))

        if large_cardinal_cats is not None:
            if (objective_type == "classification" and n_classes == 1):
                cat_enc_types.append("woe")

            cat_enc_type = trial.suggest_categorical(estimator_name + " cat_enc_type", cat_enc_types)

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

def Scaler(X, numeric_cols, trial, estimator_name, tags):
    scaler = None

    center = True
    #scaler_types = ["robust", "standard", "minmax", "maxabs"]
    scaler_types = ["standard", "minmax", "maxabs"]
    if sp.issparse(X):
        scaler_types.remove("minmax")
        center = False

    if (numeric_cols is not None) and tags["sensitive to feature scaling"]:
        scaler_type = trial.suggest_categorical(estimator_name + " scaler", scaler_types)
        if scaler_type == "standard":
            scaler = StandardScaler(with_mean=center)
        elif scaler_type == "robust":
            scaler = RobustScaler(with_centering=center)
        elif scaler_type == "maxabs":
            scaler = MaxAbsScaler()
        else:  # minmax
            scaler = MinMaxScaler()
    return scaler