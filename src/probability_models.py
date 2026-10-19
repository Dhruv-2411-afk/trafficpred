from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def add_probability_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.sort_values(["station_id", "timestamp"]).copy()
    # Hourly flow represents arrivals/hour; all estimates are shifted so they are historical.
    g = d.groupby("station_id", group_keys=False)
    d["arrival_rate_lambda"] = g.flow.transform(lambda x: x.shift(1).rolling(12, min_periods=6).mean())
    d["interarrival_mean"] = 3600 / d.arrival_rate_lambda.replace(0, np.nan)
    d["interarrival_std"] = g.interarrival_mean.transform(lambda x: x.shift(1).rolling(12, min_periods=6).std())
    d["poisson_congestion_probability"] = 1 - stats.poisson.cdf(d.arrival_rate_lambda * 1.1, d.arrival_rate_lambda)
    summary = {}
    for station, x in d.groupby("station_id"):
        counts = x.flow.dropna()
        if len(counts) > 10:
            lam = counts.mean()
            # Discrete KS is descriptive only; Monte Carlo/chi-square preferable for final inference.
            summary[str(station)] = {"lambda": float(lam), "variance_to_mean": float(counts.var()/lam),
                                     "poisson_ks_stat": float(stats.kstest(counts, "poisson", args=(lam,)).statistic)}
    return d, summary
