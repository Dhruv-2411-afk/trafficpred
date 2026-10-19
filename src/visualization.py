from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def save_core_figures(df: pd.DataFrame, metrics: pd.DataFrame, output: str | Path,
                      predictions: pd.DataFrame | None = None, simulation: pd.DataFrame | None = None) -> None:
    out=Path(output); out.mkdir(parents=True, exist_ok=True); sns.set_theme(style="whitegrid")
    sample=df.sort_values("timestamp").groupby("station_id").head(500)
    fig,ax=plt.subplots(3,1,figsize=(13,9),sharex=True)
    for a,c in zip(ax,["flow","avg_speed","avg_occupancy"]): a.plot(sample.timestamp,sample[c]); a.set_ylabel(c)
    fig.tight_layout(); fig.savefig(out/"flow_speed_occupancy_timeseries.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5)); sns.heatmap(df.pivot_table(index="weekday",columns="hour",values="flow",aggfunc="mean"),ax=ax,cmap="mako"); fig.savefig(out/"traffic_heatmap.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); sns.barplot(metrics.sort_values("RMSE").head(20),x="RMSE",y="model",hue="experiment",ax=ax); fig.tight_layout(); fig.savefig(out/"model_comparison.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4)); sns.lineplot(data=df,x="timestamp",y="arrival_rate_lambda",hue="station_id",legend=False,ax=ax); fig.savefig(out/"lambda_timeseries.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(df.arrival_rate_lambda.dropna(),stat="density",ax=ax); fig.savefig(out/"lambda_distribution.png",dpi=160); plt.close(fig)
    x=df.flow.dropna(); lam=x.mean(); fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(x,stat="density",bins=30,ax=ax); z=range(int(x.quantile(.01)),int(x.quantile(.99))+1); ax.plot(z,stats.poisson.pmf(z,lam),label="Poisson theoretical"); ax.legend(); fig.savefig(out/"poisson_observed_vs_theoretical.png",dpi=160); plt.close(fig)
    ia=df.interarrival_mean.dropna(); fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(ia,stat="density",ax=ax); ax.plot(sorted(ia),stats.expon.pdf(sorted(ia),scale=ia.mean()),label="Exponential fit"); ax.legend(); fig.savefig(out/"interarrival_exponential_fit.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); stats.probplot(ia,dist="expon",sparams=(0,ia.mean()),plot=ax); fig.savefig(out/"interarrival_qq.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4)); df.plot(x="timestamp",y=["estimated_queue_length","estimated_waiting_time"],ax=ax); fig.savefig(out/"queue_length_waiting_time.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4)); df.plot(x="timestamp",y=["poisson_congestion_probability","historical_congestion_probability"],ax=ax); fig.savefig(out/"congestion_probability.png",dpi=160); plt.close(fig)
    if predictions is not None and not predictions.empty:
        p=predictions[predictions.model==predictions.model.iloc[0]]; fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(p.actual,p.predicted,s=4); ax.set(xlabel="Actual",ylabel="Predicted"); fig.savefig(out/"actual_vs_predicted.png",dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4)); sns.histplot(p.actual-p.predicted,ax=ax); fig.savefig(out/"residuals.png",dpi=160); plt.close(fig)
    if simulation is not None and not simulation.empty:
        fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(simulation.flow,simulation.simulated_mean_count,s=4); ax.set(xlabel="Observed count",ylabel="Simulated mean count"); fig.savefig(out/"monte_carlo_results.png",dpi=160); plt.close(fig)
