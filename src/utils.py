# -*- coding: utf-8 -*-
import math

def haversine_km(lat1, lon1, lat2, lon2):
    """Zwraca odległość w km pomiędzy dwoma punktami (współrzędne stopnie)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def bearing_direction(lat1, lon1, lat2, lon2):
    """Zwraca przybliżony kierunek (N, NE, E, ...) z punktu1 do punktu2."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.cos(math.radians(lon2 - lon1))
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int(round(bearing / 45)) % 8]

def within_radius(lat, lon, center_lat, center_lon, radius_km):
    return haversine_km(lat, lon, center_lat, center_lon) <= radius_km
