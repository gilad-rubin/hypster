from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics

import optuna
from optuna.visualization import plot_intermediate_values

import numpy as np
import xgboost as xgb
import lightgbm as lgb

from catboost import CatBoost
from catboost import Pool

# Get Dataset
dataset = fetch_20newsgroups(data_home="news", subset='all', shuffle = False, remove=['headers', 'footers', 'quotes'])
word_count = [len(x.split()) for x in dataset.data]
df = pd.DataFrame({"text" : dataset.data,
                   "subclass_index" : dataset.target,
                   "word_count" : word_count})
df.sort_values("word_count", inplace=True)
df = df[df["word_count"] > 30]
target_prefix = 'sci'
df["subclass_name"] = [dataset.target_names[class_num] for class_num in df["subclass_index"]]
df["binary_label"] = np.where(df["subclass_name"].str.startswith(target_prefix), 1, 0)
df["subclass_name_vs_other"] = np.where(df["binary_label"]==1, df["subclass_name"], "other")
df = df[["text", "subclass_name", "binary_label", "subclass_name_vs_other", "word_count"]]

# Train-Test Split
df_train, df_test, y_train, y_test = train_test_split(df, df["binary_label"],
                                                    test_size = 0.33,
                                                    random_state = SEED,
                                                    stratify = df["binary_label"])
#because Pandas....
df_train = df_train.copy()
df_test = df_test.copy()

# Build Pipeline
pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2),
                                               min_df=5,
                                               use_idf=False,
                                               smooth_idf=False,
                                               strip_accents="unicode",
                                               encoding = "latin1",
                                               stop_words="english",
                                               sublinear_tf=True,
                                               norm="l2"))
                    ])
#%%
X_train = pipeline.fit_transform(df_train["text"], y_train)
X_test = pipeline.transform(df_test["text"])