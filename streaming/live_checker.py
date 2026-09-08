# -*- coding: utf-8 -*-
import requests
from config import REGIONS, CLIENT_ID, CLIENT_SECRET
from src.logger import get_logger

logger = get_logger("live_checker")

API_URL = "https://data.api.xweather.com/earthquakes/closest"

def _fetch_latest_event(api_location):
    params = {
        "p": api_location,
        "limit": 1,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        resp = data.get("response", [])
        if not resp:
            return None
        e = resp[0]
        # defensive extraction
        loc = e.get("loc", {})
        report = e.get("report", {}) or {}
        lat = loc.get("lat")
        lon = loc.get("long")
        mag = report.get("mag")
        ts = report.get("dateTimeISO")
        if lat is None or lon is None:
            return None
        return {"latitude": float(lat), "longitude": float(lon), "magnitude": (float(mag) if mag is not None else None), "timestamp": ts}
    except Exception:
        logger.exception("Error fetching latest event for %s", api_location)
        return None

def check_region_live(api_location):
    return _fetch_latest_event(api_location) is not None

def check_all_regions_live():
    live_status = {}
    for name, (_, _, api_loc) in REGIONS.items():
        try:
            live_status[name] = check_region_live(api_loc)
        except Exception:
            live_status[name] = False
    return live_status

def latest_event_per_region():
    events = {}
    for name, (_, _, api_loc) in REGIONS.items():
        try:
            ev = _fetch_latest_event(api_loc)
            if ev:
                events[name] = ev
        except Exception:
            continue
    return events
