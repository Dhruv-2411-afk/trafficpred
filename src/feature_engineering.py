from __future__ import annotations
import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame, horizon: int = 1, windows: list[int] | None = None,
                  congestion_speed: float = 45, congestion_occupancy: float = .12) -> pd.DataFrame:
    windows = windows or [3, 6, 12]
    d = df.sort_values(["station_id", "timestamp"]).copy()
    d["hour"], d["day"], d["weekday"], d["month"] = d.timestamp.dt.hour, d.timestamp.dt.day, d.timestamp.dt.weekday, d.timestamp.dt.month
    d["weekend"] = (d.weekday >= 5).astype(int)
    d["rush_hour"] = d.hour.isin([7, 8, 9, 16, 17, 18]).astype(int)
    d["truck_flow"] = np.nan  # PeMS station aggregate does not identify trucks; preserved as documented unavailable.
    g = d.groupby("station_id", group_keys=False)
    for col in ["flow", "avg_speed", "avg_occupancy"]:
        d[f"{col}_lag_1"] = g[col].shift(1)
        for w in windows:
            d[f"{col}_roll_mean_{w}"] = g[col].transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
            d[f"{col}_roll_std_{w}"] = g[col].transform(lambda x: x.shift(1).rolling(w, min_periods=w).std())
    d["congested"] = ((d.avg_speed < congestion_speed) | (d.avg_occupancy > congestion_occupancy)).astype(int)
    d["target_flow"] = g.flow.shift(-horizon)
    d["target_speed"] = g.avg_speed.shift(-horizon)
    return d
