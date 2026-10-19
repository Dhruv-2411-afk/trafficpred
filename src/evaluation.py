from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred); n = len(y)
    r2 = r2_score(y, pred); p = 1 # exact adjusted R2 needs feature count, stored separately if needed
    return {"MAE": mean_absolute_error(y,pred), "MSE": mean_squared_error(y,pred), "RMSE": mean_squared_error(y,pred)**.5, "R2": r2, "Adjusted_R2": 1-(1-r2)*(n-1)/max(n-p-1,1), "MAPE": np.mean(np.abs((y-pred)/np.maximum(np.abs(y),1)))*100}

def paired_comparison(base, enhanced) -> dict:
    a, b = np.abs(np.asarray(base)), np.abs(np.asarray(enhanced))
    if len(a) != len(b) or len(a) < 10: return {"test": "not run", "reason": "aligned prediction errors required"}
    stat, p = wilcoxon(a, b, alternative="greater")
    return {"test":"one-sided Wilcoxon signed-rank", "n":len(a), "statistic":float(stat), "p_value":float(p), "note":"Interpret only with aligned chronological test predictions."}
