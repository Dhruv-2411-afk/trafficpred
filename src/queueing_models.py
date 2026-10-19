from __future__ import annotations
import numpy as np
import pandas as pd

def add_queueing_features(df: pd.DataFrame, service_quantile: float = .9) -> pd.DataFrame:
    d = df.sort_values(["station_id", "timestamp"]).copy()
    # Proxy: historical upper-quantile observed flow, not a measured discharge/service rate.
    proxy = d.groupby("station_id").flow.transform(lambda x: x.shift(1).expanding(min_periods=24).quantile(service_quantile))
    d["service_rate_proxy_mu"] = proxy
    d["traffic_intensity"] = (d.arrival_rate_lambda / proxy).clip(upper=.999)
    rho = d.traffic_intensity
    d["estimated_queue_length"] = rho / (1 - rho)
    d["estimated_waiting_time"] = d.estimated_queue_length / d.arrival_rate_lambda.replace(0, np.nan)
    d["historical_congestion_probability"] = d.groupby("station_id").congested.transform(lambda x: x.shift(1).expanding(min_periods=12).mean())
    return d
