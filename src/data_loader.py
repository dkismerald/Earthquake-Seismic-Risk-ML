# -*- coding: utf-8 -*-
import os
import pandas as pd
from typing import Iterator
from src.logger import get_logger

logger = get_logger("data_loader")

def read_csv_chunks(path: str, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    if not os.path.exists(path):
        logger.error("CSV not found: %s", path)
        raise FileNotFoundError(path)
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            yield chunk
    except Exception:
        logger.exception("Failed to stream CSV")
        raise
