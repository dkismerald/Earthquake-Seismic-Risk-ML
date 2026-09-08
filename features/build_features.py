# features/build_features.py
# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import timedelta

from config import (
    HISTORY_WINDOW_HOURS,
    PREDICTION_HORIZON_HOURS,
    MIN_MAGNITUDE,
    REGIONS,
)
from src.utils import haversine_km

# ================= PERFORMANCE CONSTRAINTS =================
MAX_EVENTS_PER_REGION = 50_000
MAX_SAMPLES_PER_REGION = 20_000
SAMPLE_STRIDE = 10
MIN_EVENT_GAP_MINUTES = 5
# ===========================================================

logger = logging.getLogger("features")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)


def _ensure_datetime_utc(df, time_col="time"):
    if time_col not in df.columns:
        raise ValueError(f"Brak kolumny '{time_col}' w DF")

    try:
        if pd.api.types.is_numeric_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], unit="ms", utc=True)
        else:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
    except Exception:
        logger.exception("Błąd parsowania czasu – fallback")
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if df[time_col].dt.tz is None:
            df[time_col] = df[time_col].dt.tz_localize("UTC")

    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize("UTC")
    else:
        df[time_col] = df[time_col].dt.tz_convert("UTC")

    return df


def build_dataset(csv_path, min_past_events=5):
    logger.info("Wczytywanie CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    if "magnitude" not in df.columns and "magnitudo" in df.columns:
        df = df.rename(columns={"magnitudo": "magnitude"})
    if "depth_km" not in df.columns and "depth" in df.columns:
        df = df.rename(columns={"depth": "depth_km"})

    df = _ensure_datetime_utc(df, time_col="time")
    df = df.sort_values("time").reset_index(drop=True)

    samples = []
    logger.info("Budowanie próbek dla %d regionów", len(REGIONS))

    for region_name, (latc, lonc, _) in tqdm(REGIONS.items(), desc="regions"):
        lat_min, lat_max = latc - 3.0, latc + 3.0
        lon_min, lon_max = lonc - 3.0, lonc + 3.0

        region_df = df[
            (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        ].copy()

        if region_df.empty:
            continue

        # HARD LIMIT INPUT
        region_df = region_df.sort_values("time").tail(MAX_EVENTS_PER_REGION).reset_index(drop=True)

        times = region_df["time"]
        mags = region_df["magnitude"]
        depths = region_df.get("depth_km", pd.Series(0.0, index=region_df.index))

        last_sample_time = None
        region_sample_count = 0

        it = range(0, len(region_df), SAMPLE_STRIDE)
        for i in tqdm(it, desc=f"samples:{region_name}", leave=False):
            if region_sample_count >= MAX_SAMPLES_PER_REGION:
                break

            t = times.iat[i]

            if last_sample_time is not None:
                gap_min = (t - last_sample_time) / pd.Timedelta(minutes=1)
                if gap_min < MIN_EVENT_GAP_MINUTES:
                    continue

            window_start = t - pd.Timedelta(hours=HISTORY_WINDOW_HOURS)
            past_mask = (times > window_start) & (times <= t)
            past_mags = mags[past_mask]

            if past_mags.shape[0] < min_past_events:
                continue

            count_24h = int(((times > (t - pd.Timedelta(hours=24))) & (times <= t)).sum())
            count_72h = int(past_mags.shape[0])

            samples.append({
                "region": region_name,
                "ts": t.isoformat(),
                "count_24h": count_24h,
                "count_72h": count_72h,
                "max_mag_72h": float(past_mags.max()),
                "mean_mag_72h": float(past_mags.mean()),
                "mean_depth_72h": float(depths[past_mask].mean()),
                "min_depth_72h": float(depths[past_mask].min()),
                "hours_since_last": float(
                    (t - times[past_mask].max()) / pd.Timedelta(hours=1)
                ),
                "label": int(
                    (mags[(times > t) &
                          (times <= t + pd.Timedelta(hours=PREDICTION_HORIZON_HOURS))] >= MIN_MAGNITUDE).any()
                ),
            })

            last_sample_time = t
            region_sample_count += 1

        logger.info(
            "Region %s → samples: %d",
            region_name,
            region_sample_count,
        )

    logger.info("Utworzono próbek łącznie: %d", len(samples))
    return pd.DataFrame(samples)
