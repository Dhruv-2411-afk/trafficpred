from __future__ import annotations
import numpy as np
import pandas as pd

def monte_carlo(df: pd.DataFrame, simulations: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    rng=np.random.default_rng(random_state); d=df.dropna(subset=["arrival_rate_lambda", "service_rate_proxy_mu"]).copy()
    d["simulated_mean_count"]=[rng.poisson(x, simulations).mean() for x in d.arrival_rate_lambda]
    d["simulation_congestion_probability"]=[np.mean(rng.poisson(l, simulations) > m) for l,m in zip(d.arrival_rate_lambda,d.service_rate_proxy_mu)]
    return d[["timestamp","station_id","flow","simulated_mean_count","simulation_congestion_probability"]]
