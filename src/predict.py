# -*- coding: utf-8 -*-
import os
import pickle
import math
from src.logger import get_logger
from config import GLOBAL_MODEL, TARGET_HOURS

logger = get_logger("predict")

def load_global_model(path=GLOBAL_MODEL):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded global model: %s", path)
            return model
        except Exception:
            logger.exception("Fail to load model")
            raise
    return None

def heuristic_prob(rate_per_hour, horizon_hours=TARGET_HOURS):
    # Poisson survival -> P(at least one) = 1 - exp(-rate * horizon)
    return 1.0 - math.exp(-rate_per_hour * horizon_hours)

def region_score_from_history(region_entry):
    # region_entry expected fields: event_rate_per_hour
    return heuristic_prob(region_entry.get("event_rate_per_hour", 0.0))

def predict_using_model_for_region(model, features_row):
    # model expects 2D array
    try:
        pred = model.predict(features_row.reshape(1, -1))
        return float(pred[0])
    except Exception:
        logger.exception("Model predict error")
        return 0.0
