from collections import defaultdict
import requests
import pandas as pd
from config import *

API_URL = "https://data.api.xweather.com/earthquakes"

def build_live_features():
    features = {}
    t = pd.Timestamp.utcnow()

    for name, (_, _, loc) in REGIONS.items():
        params = {
            "p": loc,
            "limit": 200,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }

        r = requests.get(API_URL, params=params, timeout=10)
        data = r.json().get("response", [])

        if not data:
            continue

        df = pd.DataFrame([e['report'] for e in data])
        df['time'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.sort_values('time', ascending=False).reset_index(drop=True)

        window_start_72h = t - pd.Timedelta(hours=HISTORY_WINDOW_HOURS)
        window_start_24h = t - pd.Timedelta(hours=24)

        past_72h = df[df['time'] > window_start_72h]

        if past_72h.empty:
            continue

        past_24h = past_72h[past_72h['time'] > window_start_24h]

        mags_72h = past_72h['mag']
        depths_72h = past_72h['depthKM']

        last_time = past_72h['time'].max()
        hours_since_last = (t - last_time) / pd.Timedelta(hours=1)

        features[name] = {
            "count_24h": len(past_24h),
            "count_72h": len(past_72h),
            "max_mag_72h": mags_72h.max(),
            "mean_mag_72h": mags_72h.mean(),
            "mean_depth_72h": depths_72h.mean(),
            "min_depth_72h": depths_72h.min(),
            "hours_since_last": float(hours_since_last)
        }

    return features