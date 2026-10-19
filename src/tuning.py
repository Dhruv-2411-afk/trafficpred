from __future__ import annotations
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

def tune_tree_model(X: pd.DataFrame, y: pd.Series, kind: str, random_state: int = 42):
    candidates = {"random_forest": (RandomForestRegressor(random_state=random_state, n_jobs=-1), {"n_estimators":[200,400], "max_depth":[8,16,None], "min_samples_leaf":[1,3,6]}),
                  "gradient_boosting": (GradientBoostingRegressor(random_state=random_state), {"n_estimators":[100,250], "max_depth":[2,3,5], "learning_rate":[.03,.1], "subsample":[.7,1.]}),
                  "histgb": (HistGradientBoostingRegressor(random_state=random_state), {"max_iter":[100,250], "max_leaf_nodes":[15,31,63], "learning_rate":[.03,.1], "l2_regularization":[0,.1,1]})}
    model, grid = candidates[kind]
    return RandomizedSearchCV(model, grid, n_iter=min(12, len(grid)*3), cv=TimeSeriesSplit(4), scoring="neg_mean_absolute_error", random_state=random_state, n_jobs=-1).fit(X, y)
