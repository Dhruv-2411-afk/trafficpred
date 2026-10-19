"""Run the reproducible PeMS congestion research pipeline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, yaml
from src.data_loader import load_station_data
from src.preprocessing import select_stations_and_aggregate
from src.feature_engineering import make_features
from src.probability_models import add_probability_features
from src.queueing_models import add_queueing_features
from src.models import train_experiments
from src.simulation import monte_carlo
from src.visualization import save_core_figures

def main(config_path: str) -> None:
    cfg=yaml.safe_load(Path(config_path).read_text()); paths=cfg["paths"]
    for p in paths.values(): Path(p).mkdir(parents=True,exist_ok=True)
    raw=load_station_data(paths["raw"], cfg["data"]["chunksize"])
    hourly, station_stats=select_stations_and_aggregate(raw,cfg["data"]["station_limit"],cfg["data"]["aggregation"])
    f=cfg["features"]
    featured=make_features(hourly,f["prediction_horizon_hours"],f["rolling_windows"],f["congestion_speed_mph"],f["congestion_occupancy"])
    featured, probability_summary=add_probability_features(featured)
    featured=add_queueing_features(featured,f["service_rate_quantile"])
    processed=Path(paths["processed"]); featured.to_parquet(processed/"hourly_features.parquet",index=False); station_stats.to_csv(Path(paths["results"])/"station_statistics.csv")
    metrics,predictions=train_experiments(featured,cfg["experiment"]["target"],cfg["experiment"]["random_state"],paths["models"])
    metrics.to_csv(Path(paths["results"])/"model_metrics.csv",index=False); predictions.to_csv(Path(paths["results"])/"test_predictions.csv",index=False)
    simulation=monte_carlo(featured,cfg["experiment"]["simulation_count"],cfg["experiment"]["random_state"]); simulation.to_csv(Path(paths["results"])/"monte_carlo.csv",index=False)
    Path(paths["results"]).joinpath("probability_diagnostics.json").write_text(json.dumps(probability_summary,indent=2))
    save_core_figures(featured,metrics,paths["figures"],predictions,simulation)
    joblib.dump(cfg,Path(paths["models"])/"run_config.joblib")
    print(f"Completed: {len(featured):,} hourly rows; outputs under reports/ and models/.")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="config/config.yaml"); main(parser.parse_args().config)
