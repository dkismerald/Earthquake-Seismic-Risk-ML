import joblib
import pandas as pd
from streaming.live_feature_builder import build_live_features

model = joblib.load("model/model_lgbm.pkl")

def predict_regions():
    features = build_live_features()
    print("Raw features dict:", features)  # See the full dict for each region
    X = pd.DataFrame.from_dict(features, orient="index")
    print("X columns:", list(X.columns))  # This will show exactly which 6 are present
    print("X shape:", X.shape)
    probs = model.predict_proba(X)[:, 1]
    return dict(zip(X.index, probs))