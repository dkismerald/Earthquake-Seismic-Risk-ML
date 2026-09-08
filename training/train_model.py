# -*- coding: utf-8 -*-
import json
import os
from src.logger import get_logger
from src.data_loader import read_csv_chunks
from config import DATA_CSV, REGION_MODELS_JSON, REGIONS2, RADIUS_KM, MIN_MAGNITUDE
from src.utils import within_radius
import pandas as pd

logger = get_logger("train_model")

def main():
    if not os.path.exists(DATA_CSV):
        logger.error("CSV brak: %s", DATA_CSV)
        return
    stats = {name: {"events": 0, "times": []} for name in REGIONS2}
    processed = 0
    for i, chunk in enumerate(read_csv_chunks(DATA_CSV, chunksize=200_000)):
        processed += len(chunk)
        # parse time column defensively
        try:
            if pd.api.types.is_numeric_dtype(chunk['time']):
                chunk['time'] = pd.to_datetime(chunk['time'], unit='ms', utc=True)
            else:
                chunk['time'] = pd.to_datetime(chunk['time'], utc=True)
        except Exception:
            logger.exception("Błąd parsowania czasu w chunku")
            continue
        # filter by magnitude
        if 'magnitudo' in chunk.columns:
            chunk = chunk[chunk['magnitudo'] >= MIN_MAGNITUDE]
        else:
            logger.error("Brak kolumny 'magnitude' w CSV")
            return
        # iterate regions
        for name, (latc, lonc, _) in REGIONS2.items():
            try:
                mask = chunk.apply(lambda r: within_radius(r['latitude'], r['longitude'], latc, lonc, RADIUS_KM), axis=1)
            except Exception:
                logger.exception("Błąd podczas obliczania mask dla regionu %s", name)
                mask = pd.Series([False]*len(chunk))
            local = chunk[mask]
            stats[name]['events'] += len(local)
            stats[name]['times'].extend(local['time'].tolist())
        if i % 5 == 0:
            logger.info("Processed rows: %d", processed)
    models = {}
    for name, data in stats.items():
        if len(data['times']) < 10:
            continue
        span_hours = (max(data['times']) - min(data['times'])).total_seconds() / 3600
        rate = data['events'] / span_hours if span_hours > 0 else 0.0
        lat, lon, _ = REGIONS2[name]
        models[name] = {
            "center": [lat, lon],
            "radius_km": RADIUS_KM,
            "min_magnitude": MIN_MAGNITUDE,
            "event_rate_per_hour": rate,
            "total_events": int(data['events']),
            "time_span_hours": span_hours
        }
    os.makedirs(os.path.dirname(REGION_MODELS_JSON) or ".", exist_ok=True)
    with open(REGION_MODELS_JSON, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=4)
    logger.info("Models created for: %s", list(models.keys()))

if __name__ == "__main__":
    main()
