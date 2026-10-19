from __future__ import annotations
import pandas as pd

def select_stations_and_aggregate(df: pd.DataFrame, station_limit: int = 15, frequency: str = "1h") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select high-observation stations from data, then aggregate to hourly values."""
    quality = (df.groupby("station_id").agg(observation_count=("timestamp", "size"),
               traffic_volume=("flow", "sum"), speed=("avg_speed", "mean"),
               occupancy=("avg_occupancy", "mean"), missing_percent=("percent_observed", lambda x: 100 - x.mean()))
               .sort_values("observation_count", ascending=False))
    selected = quality.head(station_limit).index
    d = df[df.station_id.isin(selected)].copy().set_index("timestamp")
    hourly = (d.groupby("station_id").resample(frequency).agg(
        flow=("flow", "sum"), avg_speed=("avg_speed", "mean"), avg_occupancy=("avg_occupancy", "mean"),
        percent_observed=("percent_observed", "mean"), samples=("samples", "sum"))
        .reset_index().dropna(subset=["flow", "avg_speed"]))
    return hourly.sort_values(["station_id", "timestamp"]), quality
