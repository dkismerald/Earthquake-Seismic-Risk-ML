# -*- coding: utf-8 -*-
import csv
from predictor.region_predictor import predict_regions
from streaming.live_checker import latest_event_per_region
from config import REGIONS, PREDICTIONS_CSV
from src.utils import haversine_km, bearing_direction
from src.logger import get_logger
logger = get_logger("export")

def main():
    preds = predict_regions()
    events = latest_event_per_region()
    rows = []
    for region, p_raw in preds.items():
        ev = events.get(region)
        mag = None
        dist = None
        direction = None
        if ev:
            lat, lon, _ = REGIONS[region]
            dist = haversine_km(lat, lon, ev["latitude"], ev["longitude"])
            direction = bearing_direction(lat, lon, ev["latitude"], ev["longitude"])
            mag = ev.get("magnitude")
        # compute risk as in main
        from main import risk_index, magnitude_boost
        risk = risk_index(p_raw) * magnitude_boost(mag)
        risk = min(risk, 1.0)
        rows.append([region, p_raw, risk, ("YES" if ev else "NO"), mag, dist, direction])
    try:
        with open(PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["region","raw_prob","risk","live","mag","dist_km","direction"])
            writer.writerows(rows)
        logger.info("Saved %s", PREDICTIONS_CSV)
    except Exception:
        logger.exception("Failed to write predictions CSV")

if __name__ == "__main__":
    main()
