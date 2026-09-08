# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import timedelta
from src.logger import get_logger
from src.utils import haversine_km

logger = get_logger("featurize")

def preprocess_times(df: pd.DataFrame):
    """Upewnij się, że kolumna 'time' jest datetime UTC."""
    if 'time' not in df.columns:
        raise ValueError("Brak kolumny 'time' w dataframe")
    if np.issubdtype(df['time'].dtype, np.number):
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    else:
        df['time'] = pd.to_datetime(df['time'], utc=True)
    return df

def create_time_indexed_samples(df_events: pd.DataFrame, start=None, end=None, freq='H'):
    """Zwraca timeline (czasowe ts) pokrywające okres z df_events."""
    if start is None:
        start = df_events['time'].min().floor('H')
    if end is None:
        end = df_events['time'].max().ceil('H')
    timeline = pd.DataFrame({"ts": pd.date_range(start, end, freq=freq)})
    return timeline

def aggregate_features_for_timeline(df_events: pd.DataFrame, timeline: pd.DataFrame,
                                   window_hours=24, min_mag=4.0):
    times = df_events['time'].to_numpy()
    mags = df_events['magnitude'].to_numpy()
    counts = []
    maxs = []
    means = []
    time_since = []
    for ts in timeline['ts']:
        window_start = ts - pd.Timedelta(hours=window_hours)
        mask = (times > window_start) & (times <= ts)
        window_mags = mags[mask]
        counts.append(int(window_mags.size))
        maxs.append(float(window_mags.max()) if window_mags.size>0 else 0.0)
        means.append(float(window_mags.mean()) if window_mags.size>0 else 0.0)
        past = times[times <= ts]
        if past.size == 0:
            time_since.append(np.nan)
        else:
            time_since.append((ts - past[-1]) / np.timedelta64(1, 'h'))
    X = pd.DataFrame({
        "ts": timeline['ts'],
        "count_24h": counts,
        "max_mag_24h": maxs,
        "mean_mag_24h": means,
        "hours_since_last": time_since,
        "hour": timeline['ts'].dt.hour,
        "dayofweek": timeline['ts'].dt.dayofweek
    })
    X = X.fillna(-1)
    return X
