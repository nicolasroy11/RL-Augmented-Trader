from typing import List
import numpy as np

from traderplus.db.data_store import Tick

def extract_features(ticks: List[Tick]):
    prices = np.array([tick.price for tick in ticks])
    rsi_5 = np.array([tick.rsi_5 for tick in ticks])
    rsi_7 = np.array([tick.rsi_7 for tick in ticks])
    rsi_12 = np.array([tick.rsi_12 for tick in ticks])
    ema_short = np.array([tick.ema_short for tick in ticks])
    ema_mid = np.array([tick.ema_mid for tick in ticks])
    ema_long = np.array([tick.ema_long for tick in ticks])
    ema_xlong = np.array([tick.ema_xlong for tick in ticks])
    macd_line = np.array([tick.macd_line for tick in ticks])
    macd_hist = np.array([tick.macd_hist for tick in ticks])
    macd_signal = np.array([tick.macd_signal for tick in ticks])
    return {
        "price": prices,
        "rsi_5": rsi_5,
        "rsi_7": rsi_7,
        "rsi_12": rsi_12,
        "ema_short": ema_short,
        "ema_mid": ema_mid,
        "ema_long": ema_long,
        "ema_xlong": ema_xlong,
        "macd_line": macd_line,
        "macd_hist": macd_hist,
        "macd_signal": macd_signal
    }


def calculate_min_max(features_dict):
    min_vals = {}
    max_vals = {}
    for feature, values in features_dict.items():
        if feature.startswith("rsi"):
            # RSI already in 0-100 range, normalize by dividing by 100
            min_vals[feature] = 0.0
            max_vals[feature] = 100.0
        else:
            min_vals[feature] = np.min(values)
            max_vals[feature] = np.max(values)
    return min_vals, max_vals


def normalize_tick(tick, min_vals, max_vals):
    features = []
    for feature in min_vals.keys():
        val = getattr(tick, feature)
        # Normalize RSI by dividing by 100
        if feature.startswith("rsi"):
            norm_val = val / 100.0
        else:
            norm_val = (val - min_vals[feature]) / (max_vals[feature] - min_vals[feature] + 1e-8)
        features.append(norm_val)
    return np.array(features)
