# training/train_ml_model.py
# -*- coding: utf-8 -*-

import sys
import logging
from pathlib import Path

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from features.build_features import build_dataset
from config import SEED, USE_GPU, GLOBAL_MODEL, PREDICTIONS_CSV


# ---------------- logging ----------------
logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s trainer: %(message)s")
    )
    logger.addHandler(handler)


# ---------------- gpu probing ----------------
def try_gpu_training():
    """
    Returns True only if LightGBM REALLY supports GPU at runtime.
    """
    if not USE_GPU:
        return False

    try:
        logger.info("Sprawdzanie GPU (LightGBM)...")
        test_model = lgb.LGBMClassifier(
            n_estimators=1,
            device="gpu"
        )
        X_dummy = np.random.rand(10, 3)
        y_dummy = np.random.randint(0, 2, size=10)
        test_model.fit(X_dummy, y_dummy)
        logger.info("GPU działa poprawnie.")
        return True
    except Exception as e:
        logger.warning("GPU niedostępne lub wadliwe: %s", str(e))
        return False


# ---------------- training ----------------
def train():
    np.random.seed(SEED)

    logger.info("Start treningu ML")
    logger.info("Seed: %d", SEED)

    logger.info("Budowanie datasetu...")
    df = build_dataset("../data/earthquakes_1990_2023.csv")

    if df is None or df.empty:
        logger.error("Dataset pusty - przerywam")
        return

    logger.info("Dataset: %d próbek | %d kolumn", df.shape[0], df.shape[1])

    X = df.drop(columns=["region", "ts", "label"])
    y = df["label"].astype(np.int8)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=SEED
    )

    logger.info("Train: %d | Val: %d", len(X_train), len(X_val))

    params = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        n_estimators=600,
        seed=SEED,
        verbosity=-1
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]

    # --------- GPU if REALLY available ----------
    if try_gpu_training():
        try:
            logger.info("Trening na GPU...")
            model = lgb.LGBMClassifier(**params, device="gpu")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=callbacks
            )
        except Exception as e:
            logger.warning("GPU zawiodło w trakcie treningu: %s", str(e))
            logger.info("Przełączam na CPU...")
            model = None
    else:
        model = None

    # --------- CPU fallback (GUARANTEED) ----------
    if model is None:
        logger.info("Trening na CPU...")
        model = lgb.LGBMClassifier(**params, device="cpu")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=callbacks
        )

    logger.info("Trening zakończony")

    # ---------------- evaluation ----------------
    y_pred = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_pred)
    ap = average_precision_score(y_val, y_pred)

    logger.info("Walidacja:")
    logger.info("ROC AUC: %.4f", auc)
    logger.info("Average Precision: %.4f", ap)

    # ---------------- save outputs ----------------
    Path(GLOBAL_MODEL).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, GLOBAL_MODEL)
    logger.info("Model zapisany: %s", GLOBAL_MODEL)

    Path(PREDICTIONS_CSV).parent.mkdir(parents=True, exist_ok=True)
    out = X_val.copy()
    out["label"] = y_val.values
    out["pred"] = y_pred
    out.to_csv(PREDICTIONS_CSV, index=False)
    logger.info("Predykcje zapisane: %s", PREDICTIONS_CSV)


if __name__ == "__main__":
    train()
