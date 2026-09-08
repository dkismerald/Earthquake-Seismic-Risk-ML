import json
import os
import pickle
import numpy as np
from sklearn.linear_model import SGDRegressor

REGION_FILE = "region_models.json"
MODEL_FILE = "global_earthquake_model.pkl"
STATE_FILE = "trained_regions.json"

# -------------------------
# Load data
# -------------------------

with open(REGION_FILE, "r") as f:
    regions = json.load(f)

# Load or initialize trained-regions state
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
        trained_regions = set(json.load(f))
else:
    trained_regions = set()

# Load or initialize models
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    model_initialized = True
else:
    model = SGDRegressor()
    model_initialized = False

new_regions = 0

# -------------------------
# Incremental training
# -------------------------

for name, data in regions.items():
    if name in trained_regions:
        continue  # already processed

    X = np.array([[
        data["event_rate_per_hour"],
        data["total_events"],
        data["time_span_hours"],
        data["min_magnitude"],
        data["radius_km"]
    ]])

    y = np.array([data["event_rate_per_hour"]])

    if not model_initialized:
        model.partial_fit(X, y)
        model_initialized = True
    else:
        model.partial_fit(X, y)

    trained_regions.add(name)
    new_regions += 1
    print(f"Trained on region: {name}")

# -------------------------
# Save state
# -------------------------

if new_regions > 0:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(STATE_FILE, "w") as f:
        json.dump(sorted(list(trained_regions)), f, indent=2)

print(f"Training complete. New regions added: {new_regions}")
