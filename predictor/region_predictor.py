# -*- coding: utf-8 -*-
import json
import os
from config import REGION_MODELS_JSON, GLOBAL_MODEL, TARGET_HOURS
from src.predict import load_global_model, heuristic_prob
from src.logger import get_logger

logger = get_logger("region_predictor")

def _load_region_models(path=REGION_MODELS_JSON):
    if not os.path.exists(path):
        logger.warning("Region models JSON not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to read region models JSON")
        return {}

def predict_regions():
    """
    Zwraca dict: region -> raw_probability (surowa wartość)
    - jeśli istnieje model globalny -> można użyć (ale prostszy pipeline używa historycznego rate)
    """
    regions = _load_region_models()
    preds = {}
    # try global model if exists
    model = load_global_model(GLOBAL_MODEL)
    for region, entry in regions.items():
        try:
            # fallback: use history-based Poisson approx
            rate = entry.get("event_rate_per_hour", 0.0)
            p = heuristic_prob(rate, TARGET_HOURS)
            preds[region] = float(p)
        except Exception:
            preds[region] = 0.0
    # If global model exists and you have precomputed features, you could override here.
    # For now we keep this simple and defensible: historica -> poisson probability.
    return preds
