"""
Utility functions for Seismic Fractal Analysis.
"""

import numpy as np


def geographic_to_metric(coordinates: np.ndarray) -> np.ndarray:
    """
    Transform WGS84 coordinates (lon, lat, depth) to local Euclidean metric coordinates (km).

    Args:
        coordinates: Numpy array of shape (N, 3) containing [longitude, latitude, depth].

    Returns:
        Numpy array of shape (N, 3) containing [x_km, y_km, z_km].
    """
    coords_metric = coordinates.copy()
    mean_latitude = np.mean(coordinates[:, 1])

    # WGS84 approximation
    km_per_degree_lat = 111.1
    km_per_degree_lon = 111.1 * np.cos(np.radians(mean_latitude))

    min_lon = np.min(coordinates[:, 0])
    min_lat = np.min(coordinates[:, 1])

    # Transform to km relative to minimum corner
    coords_metric[:, 0] = (coordinates[:, 0] - min_lon) * km_per_degree_lon
    coords_metric[:, 1] = (coordinates[:, 1] - min_lat) * km_per_degree_lat
    # Depth is already in km, so we leave column 2 as is

    return coords_metric


def normalize_coordinates(metric_coords: np.ndarray) -> np.ndarray:
    """
    Normalize metric coordinates to unit cube [0, 1]^3 while preserving aspect ratio.

    Args:
        metric_coords: Array of shape (N, 3) in km.

    Returns:
        Normalized array of shape (N, 3).
    """
    minima = np.min(metric_coords, axis=0)
    maxima = np.max(metric_coords, axis=0)

    # Use the largest dimension to scale everything, preserving aspect ratio
    max_range = np.max(maxima - minima)

    # Avoid division by zero
    if max_range == 0:
        return np.zeros_like(metric_coords)

    normalized = (metric_coords - minima) / max_range
    return normalized
