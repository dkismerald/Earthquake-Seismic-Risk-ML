# -*- coding: utf-8 -*-
import logging
import sys

def get_logger(name: str = "project", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    return logger
