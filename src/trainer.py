# src/trainer.py
import os
import sys
import math
import json
import time
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# try optional imports
try:
    import lightgbm as lgb
except Exception as e:
    lgb = None

import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

# --- project-local imports (put these in src/) ---
# from config import *
# from utils import haversine_km  # if needed

# Minimal config (you can move to config.py)
DATA_CSV = os.path.join("data", "earthquakes_1990_2023.csv")
MODEL_OUT = os.path.join("models", "model_lgbm.pkl")
PRED_OUT = os.path.join("results", "predictions.csv")

SEED = 42
TARGET_HOURS = 24            # predict if >=1 event in next 24h
MIN_MAG = 4.0                # event magnitude threshold for label
REGION_CENTER = (36.7783, -119.4179)  # example region default (California)
RADIUS_KM = 150              # region radius for assignment

# logging
logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(ch)

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def safe_read_csv(path):
    if not os.path.exists(path):
        logger.error(f"CSV not found: {path}")
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.exception("Failed to read CSV")
        raise
    return df

# minimal haversine for assigning events to region
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def assign_region(df, center, radius_km):
    latc, lonc = center
    dists = df.apply(lambda r: haversine_km(latc, lonc, r['latitude'], r['longitude']), axis=1)
    return df.loc[dists <= radius_km].copy()

def make_time_features(df):
    # df must have 'time' as datetime
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    return df

def create_samples(df_events, horizon_hours=TARGET_HOURS, min_mag=MIN_MAG):
    """
    Create time-indexed samples (one per hour) with features:
    - count_last_24h
    - max_mag_last_24h
    - mean_mag_last_24h
    - time_since_last_event_hours
    target: 1 if in next horizon_hours exists event with mag>=min_mag
    """
    # ensure events sorted by time
    df_events = df_events.sort_values("time").reset_index(drop=True)
    # create hourly timeline spanning events
    start = df_events['time'].min().floor('H')
    end = df_events['time'].max().ceil('H')
    timeline = pd.DataFrame({"ts": pd.date_range(start, end, freq='H')})
    # For speed, we use numpy-based sliding aggregations
    times = df_events['time'].to_numpy()
    mags = df_events['magnitude'].to_numpy()

    # helper to compute aggregates for each timeline ts
    counts = []
    maxs = []
    means = []
    t_since = []
    targets = []

    event_idx = 0
    n = len(times)
    for ts in timeline['ts']:
        window_start = ts - pd.Timedelta(hours=24)
        # mask indices in last 24h
        mask = (times > window_start) & (times <= ts)
        window_mags = mags[mask]
        counts.append(int(window_mags.size))
        maxs.append(float(window_mags.max()) if window_mags.size>0 else 0.0)
        means.append(float(window_mags.mean()) if window_mags.size>0 else 0.0)
        # time since last event
        past = times[times <= ts]
        if past.size==0:
            t_since.append(np.nan)
        else:
            t_since.append( (ts - past[-1]) / np.timedelta64(1,'h') )
        # target: any event in (ts, ts + horizon] with mag>=min_mag
        future_mask = (times > ts) & (times <= ts + pd.Timedelta(hours=horizon_hours))
        t_mags = mags[future_mask]
        targets.append(1 if (t_mags.size>0 and t_mags.max()>=min_mag) else 0)

    X = pd.DataFrame({
        "ts": timeline['ts'],
        "count_24h": counts,
        "max_mag_24h": maxs,
        "mean_mag_24h": means,
        "hours_since_last": t_since,
        "hour": timeline['ts'].dt.hour,
        "dayofweek": timeline['ts'].dt.dayofweek
    })
    y = pd.Series(targets, name='target')
    # drop initial rows where hours_since_last is NaN if you want
    X = X.fillna(-1)
    return X, y

def train_lgbm(X_train, y_train, X_val, y_val, use_gpu=True, seed=SEED):
    if lgb is None:
        raise RuntimeError("lightgbm not installed; please pip install lightgbm")

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": seed,
        "feature_pre_filter": False
    }

    # try GPU if requested
    if use_gpu:
        # two possible devices: 'gpu' for newer builds; also check for 'gpu' support
        try:
            params["device"] = "gpu"
            logger.info("Attempting to use LightGBM GPU")
        except Exception:
            logger.warning("GPU params not applied; will use CPU")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    evals_result = {}
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=50
    )
    return bst, evals_result

def main():
    seed_everything(SEED)
    logger.info("Starting training pipeline")
    df = safe_read_csv(DATA_CSV)

    # basic expected columns
    required = {'time','latitude','longitude','magnitude'}
    if not required.issubset(set(df.columns)):
        logger.error("CSV missing required columns: need time, latitude, longitude, magnitude")
        raise RuntimeError("Bad CSV")

    # parse time column (support ms timestamps or ISO strings)
    try:
        # try numeric ms timestamps
        if np.issubdtype(df['time'].dtype, np.number):
            df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        else:
            df['time'] = pd.to_datetime(df['time'], utc=True)
    except Exception:
        logger.exception("Failed to parse time column")
        raise

    # assign region (example: use one region center; in full pipeline loop over REGIONS)
    df_region = assign_region(df, REGION_CENTER, RADIUS_KM)
    if df_region.empty:
        logger.warning("No events in region; exiting")
        return

    X, y = create_samples(df_region, horizon_hours=TARGET_HOURS, min_mag=MIN_MAG)
    # time-based split: last 20% timeline as val
    split_idx = int(len(X)*0.8)
    X_train, X_val = X.iloc[:split_idx].drop(columns=['ts']), X.iloc[split_idx:].drop(columns=['ts'])
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}")

    # train
    use_gpu = True
    try:
        bst, evals = train_lgbm(X_train, y_train, X_val, y_val, use_gpu=use_gpu, seed=SEED)
    except Exception as e:
        logger.exception("GPU training failed, retrying on CPU")
        bst, evals = train_lgbm(X_train, y_train, X_val, y_val, use_gpu=False, seed=SEED)

    # evaluate
    y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    auc_score = roc_auc_score(y_val, y_pred)
    ap = average_precision_score(y_val, y_pred)
    logger.info(f"VAL ROC-AUC: {auc_score:.4f}, AP: {ap:.4f}")

    # save model and preds
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    import joblib
    joblib.dump(bst, MODEL_OUT)
    logger.info(f"Saved model to {MODEL_OUT}")

    os.makedirs(os.path.dirname(PRED_OUT), exist_ok=True)
    df_out = X_val.copy()
    df_out['target'] = y_val.values
    df_out['pred'] = y_pred
    df_out.to_csv(PRED_OUT, index=False)
    logger.info(f"Saved predictions to {PRED_OUT}")

if __name__ == "__main__":
    main()
