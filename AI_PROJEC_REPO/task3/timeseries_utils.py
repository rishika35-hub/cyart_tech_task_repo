# timeseries_utils.py

import numpy as np
import pandas as pd
import numexpr as ne
from numba import njit

def generate_time_series(length=100000, seed=42):
    np.random.seed(seed)
    return np.cumsum(np.random.randn(length))

def moving_average_numpy(series, window):
    weights = np.ones(window) / window
    return np.convolve(series, weights, mode='valid')

def moving_average_pandas(series, window):
    return series.rolling(window=window).mean().dropna().values

def moving_average_numexpr(series, window):
    """Simulates a moving average using explicit loop and NumExpr evaluation."""
    cumsum = np.cumsum(series, dtype=float)
    cumsum[window:] = ne.evaluate("cumsum[window:] - cumsum[:-window]")
    return cumsum[window - 1:] / window

@njit
def moving_average_numba(series, window):
    result = np.empty(len(series) - window + 1)
    for i in range(len(result)):
        result[i] = np.mean(series[i:i+window])
    return result
