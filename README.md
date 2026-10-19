# Traffic Congestion as a Probability Problem

A reproducible research project testing whether historical probabilistic-arrival and queueing proxies improve chronological traffic-flow (or speed) prediction from official Caltrans PeMS District 7 station 5-minute data.

## Data and verified format

Only raw official PeMS `d07_text_station_5min_*.txt.gz` files are used. The supplied January 2026 file was inspected before implementation: it is headerless with 52 comma-separated fields. The parser maps its station-level fields (`timestamp`, station ID, district, flow, average occupancy, average speed) and preserves lane blocks. No station is invented: the pipeline selects the configured number of stations with the most valid observations and writes their statistics.

Put supplied files in `data/raw/` (they are intentionally not committed), create an environment, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
```

Set all paths, station count, dates, horizon, congestion thresholds, chronological split, seed (42), and simulation count in `config/config.yaml`.

## Research design

The parser streams compressed files in chunks, rejects malformed timestamps/IDs and impossible negative flow or speed, aggregates selected stations to hourly measurements, and builds **strictly past-looking** lags and rolling summaries. Targets are station-specific future values at the configured horizon. Splits are chronological (70/15/15); there is no random shuffle.

Experiments cumulatively compare temporal; traffic; lag/rolling; probability; queueing; and full features across Linear, Ridge, Lasso, Random Forest, Gradient Boosting, and a HistGradientBoosting fallback labelled `xgboost_fallback_histgb`. Metrics include MAE, MSE, RMSE, R², adjusted R² (feature-count approximation is documented in code), and guarded MAPE. Saved test prediction errors can be compared with a paired Wilcoxon signed-rank test only when aligned.

Arrival rate λ is a trailing 12-hour mean flow. Poisson diagnostics report variance-to-mean and a descriptive KS statistic; these are diagnostics, not proof that traffic is Poisson. The implied exponential inter-arrival mean is 3600/λ seconds. The service rate μ is **not measured**: it is explicitly a past expanding 90th-percentile flow proxy. Therefore M/M/1 quantities (ρ, queue length, wait) are labelled approximations, constrained below instability, and must not be presented as physical queue measurements. Truck flow is unavailable in this station aggregate and remains missing rather than fabricated.

Monte Carlo samples Poisson arrivals from the historical λ (default 10,000 draws) and reports simulated count and proxy-capacity exceedance probabilities. Core figures, model metrics, station summary, probability diagnostics, simulations, predictions, processed data, and joblib models are persisted below `reports/`, `data/processed/`, and `models/`.

## Limitations

PeMS detector health, station coverage, lane configuration, incidents, weather, spatial correlations, and non-Poisson arrivals can materially affect results. Aggregate flow is not individual vehicle arrival telemetry. Probability and queueing features are predictive proxies; they do not establish causality or validate an M/M/1 physical model. Inspect diagnostic outputs and compare against conventional features before making the core contribution claim.
