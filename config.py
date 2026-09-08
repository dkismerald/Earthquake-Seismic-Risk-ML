import random
import numpy as np
CLIENT_ID = "OW5A7GNPerjPrTgF86UvH"
CLIENT_SECRET = "9KHozqzlAefI6AY9s5iGK1rFX5c8zrrF8w4MtCKL"

BASE_URL = "https://data.api.xweather.com"
# -*- coding: utf-8 -*-
"""
Konfiguracja projektu - wartości domyślne.
Edytuj REGIONS i zmienne środowiskowe według potrzeby.
"""
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
DATA_CSV = "data/earthquakes_1990_2023.csv"
REGION_MODELS_JSON = "model/region_models.json"
GLOBAL_MODEL = "model/model_lgbm.pkl"
PREDICTIONS_CSV = "results/predictions.csv"

# Prediction task
TARGET_HOURS = 24         # horyzont predykcji (np. 24)
MIN_MAGNITUDE = 4.0       # minimalna magnituda rozpatrywana jako 'event'

# Region radius (kilometry) - domyślnie dla skryptów treningowych
RADIUS_KM = 150

# Polling / runtime
DEFAULT_POLL_INTERVAL = 60
DEFAULT_DURATION = 3600

# Reproducibility
RANDOM_SEED = SEED
# ============================================================
# Parametry czasowe (kluczowe dla ML)
# ============================================================

# ile godzin wstecz model "patrzy"
HISTORY_WINDOW_HOURS = 72

# na ile godzin w przyszłość robimy predykcję
PREDICTION_HORIZON_HOURS = 24

USE_GPU = False  # jeżeli LightGBM nie ma GPU -> automatyczny fallback
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
MAX_DEPTH = 6

# REGIONS - przyklad; dopasuj/rozszerz zgodnie z wymaganiami
REGIONS = {
    "Tokyo_JP": (35.6895, 139.6917, "tokyo,jp"),
    "Santiago_CL": (-33.4489, -70.6693, "santiago,cl"),
    "Lima_PE": (-12.0464, -77.0428, "lima,pe"),
    "Jakarta_ID": (-6.2088, 106.8456, "jakarta,id"),
    "Manila_PH": (14.5995, 120.9842, "manila,ph"),
    "Taipei_TW": (25.0330, 121.5654, "taipei,tw"),
    "Tehran_IR": (35.6892, 51.3890, "tehran,ir"),
    "Kathmandu_NP": (27.7172, 85.3240, "kathmandu,np"),
    "SanFrancisco_US": (37.7749, -122.4194, "san francisco,ca"),
    "LosAngeles_US": (34.0522, -118.2437, "los angeles,ca"),
    "KermadecIslands_NZ": (-29.2500, -177.9167, "kermadec islands,nz"),
    "MontePatria_CL": (-30.6970, -70.9530, "monte patria,cl"),
    "Mendi_PG": (-6.1476, 143.6559, "mendi,pg"),
    "Hengchun_TW": (21.9580, 120.7793, "hengchun,tw"),
    "Adak_AK": (51.8800, -176.6581, "adak,ak"),
    "HappyValley_AK": (59.7100, -151.3150, "happy valley,ak"),
    "SeveroKurilsk_RU": (50.6789, 156.1275, "severo-kurilsk,ru"),
    "Labuan_ID": (5.2831, 115.2308, "labuan,id"),
    "Ohonua_TO": (-21.3333, -175.6667, "ohonua,to"),
    "VillaElCarmen_NI": (11.9760, -86.2860, "villa el carmen,ni"),
    "MacquarieIsland_AU": (-54.6200, 158.8500, "macquarie island,au"),
}

# Regions to models (lat, lon)
REGIONS2 = {
    "Nakatonbetsu_JP": (45.0846, 142.1755, "nakatonbetsu,jp"),
    "Honshu_JP": (36.2048, 138.2529, "honshu,jp"),
    "California_US": (36.7783, -119.4179, "california,ca"),
    "Chile": (-35.6751, -71.5430, "chile,"),
    "Tokyo_JP": (35.6895, 139.6917, "tokyo,jp"),
    "Santiago_CL": (-33.4489, -70.6693, "santiago,cl"),
    "Lima_PE": (-12.0464, -77.0428, "lima,pe"),
    "Jakarta_ID": (-6.2088, 106.8456, "jakarta,id"),
    "Manila_PH": (14.5995, 120.9842, "manila,ph"),
    "Taipei_TW": (25.0330, 121.5654, "taipei,tw"),
    "Tehran_IR": (35.6892, 51.3890, "tehran,ir"),
    "Kathmandu_NP": (27.7172, 85.3240, "kathmandu,np"),
    "SanFrancisco_US": (37.7749, -122.4194, "san francisco,ca"),
    "LosAngeles_US": (34.0522, -118.2437, "los angeles,ca"),
    "MexicoCity_MX": (19.4326, -99.1332, "mexico city,mx"),
    "Anchorage_US": (61.2181, -149.9003, "anchorage,ak"),
    "Wellington_NZ": (-41.2865, 174.7762, "wellington,nz"),
    "Naples_IT": (40.8518, 14.2681, "naples,it"),
    "Istanbul_TR": (41.0082, 28.9784, "istanbul,tr"),
    "Athens_GR": (37.9838, 23.7275, "athens,gr"),
    "Quito_EC": (-0.1807, -78.4678, "quito,ec"),
    "LaPaz_BO": (-16.4897, -68.1193, "la paz,bo"),
    "PortAuPrince_HT": (18.5944, -72.3074, "port-au-prince,ht"),
    "Reykjavik_IS": (64.1466, -21.9426, "reykjavik,is"),
    "AddisAbaba_ET": (8.9806, 38.7578, "addis ababa,et"),
    "KermadecIslands_NZ": (-29.2500, -177.9167, "kermadec islands,nz"),
    "MontePatria_CL": (-30.6970, -70.9530, "monte patria,cl"),
    "Mendi_PG": (-6.1476, 143.6559, "mendi,pg"),
    "Hengchun_TW": (21.9580, 120.7793, "hengchun,tw"),
    "CarlsbergRidge_IN": (3.0000, 60.0000, "carlsberg ridge,indian ocean"),
    "Adak_AK": (51.8800, -176.6581, "adak,ak"),
    "HappyValley_AK": (59.7100, -151.3150, "happy valley,ak"),
    "SeveroKurilsk_RU": (50.6789, 156.1275, "severo-kurilsk,ru"),
    "Labuan_ID": (5.2831, 115.2308, "labuan,id"),
    "Ohonua_TO": (-21.3333, -175.6667, "ohonua,to"),
    "VillaElCarmen_NI": (11.9760, -86.2860, "villa el carmen,ni"),
    "MacquarieIsland_AU": (-54.6200, 158.8500, "macquarie island,au"),
}

REGIONS3 = {
    # Japan / Pacific Ring of Fire
    "Tokyo_JP": (35.6895, 139.6917, "tokyo,jp"),
    "Honshu_JP": (36.2048, 138.2529, "honshu,jp"),

    # US West Coast
    "California_US": (36.7783, -119.4179, "california,ca"),
    "SanFrancisco_US": (37.7749, -122.4194, "san francisco,ca"),

    # South America subduction zone
    "Chile": (-35.6751, -71.5430, "chile,"),
    "Santiago_CL": (-33.4489, -70.6693, "santiago,cl"),

    # Southeast Asia / Sunda plate
    "Jakarta_ID": (-6.2088, 106.8456, "jakarta,id"),

    # Philippines subduction zone
    "Manila_PH": (14.5995, 120.9842, "manila,ph"),

    # New Zealand plate boundary
    "Wellington_NZ": (-41.2865, 174.7762, "wellington,nz"),

    # Middle East collision zone
    "Istanbul_TR": (41.0082, 28.9784, "istanbul,tr"),
}