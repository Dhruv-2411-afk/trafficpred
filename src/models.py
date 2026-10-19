from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    temporal = [x for x in ["hour", "day", "weekday", "month", "weekend", "rush_hour"] if x in columns]
    traffic = temporal + [x for x in ["avg_speed", "avg_occupancy", "percent_observed"] if x in columns]
    lag = traffic + [x for x in columns if "lag_" in x or "roll_" in x]
    probability = lag + [x for x in ["arrival_rate_lambda", "interarrival_mean", "interarrival_std", "poisson_congestion_probability"] if x in columns]
    queue = probability + [x for x in ["traffic_intensity", "estimated_queue_length", "estimated_waiting_time", "historical_congestion_probability"] if x in columns]
    return {"A_temporal": temporal, "B_traffic": traffic, "C_lag_rolling": lag, "D_probability": probability, "E_queueing": queue, "F_full": queue}

def estimators(random_state: int = 42) -> dict[str, object]:
    pre = lambda scale=False: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())] if scale else [("imputer", SimpleImputer(strategy="median"))])
    return {"linear": Pipeline([("prep", pre(True)), ("model", LinearRegression())]),
            "ridge": Pipeline([("prep", pre(True)), ("model", Ridge(alpha=1.0))]),
            "lasso": Pipeline([("prep", pre(True)), ("model", Lasso(alpha=.05, max_iter=10000))]),
            "random_forest": Pipeline([("prep", pre()), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=random_state, n_jobs=-1))]),
            "gradient_boosting": Pipeline([("prep", pre()), ("model", GradientBoostingRegressor(random_state=random_state))]),
            "xgboost_fallback_histgb": Pipeline([("prep", pre()), ("model", HistGradientBoostingRegressor(random_state=random_state))])}

def chronological_split(df: pd.DataFrame, train_ratio: float = .7, validation_ratio: float = .15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.sort_values("timestamp").reset_index(drop=True); n = len(d); a, b = int(n*train_ratio), int(n*(train_ratio+validation_ratio))
    return d.iloc[:a], d.iloc[a:b], d.iloc[b:]

def train_experiments(df: pd.DataFrame, target: str, random_state: int, model_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.evaluation import metrics
    target_col = f"target_{target}"; data = df.dropna(subset=[target_col]); train, _, test = chronological_split(data)
    rows, predictions = [], []
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for exp, features in feature_sets(list(data.columns)).items():
        if not features: continue
        for name, estimator in estimators(random_state).items():
            estimator.fit(train[features], train[target_col]); pred = estimator.predict(test[features])
            row = {"experiment": exp, "model": name, "target": target, "features": len(features), **metrics(test[target_col], pred)}
            rows.append(row)
            predictions.append(pd.DataFrame({"timestamp": test.timestamp, "station_id": test.station_id, "actual": test[target_col], "predicted": pred, "experiment": exp, "model": name}))
            joblib.dump({"pipeline": estimator, "features": features, "target": target}, Path(model_dir)/f"{target}_{exp}_{name}.joblib")
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)
