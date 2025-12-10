"""
Advanced synthetic validation geometries for SFA.
Generates known fractal structures for algorithm validation.
"""

import numpy as np
from typing import Tuple

__all__ = [
    "generate_cantor_dust",
    "generate_sierpinski_carpet",
    "generate_fractal_line",
    "generate_fractal_plane",
    "generate_etas_synthetic",
]


def generate_cantor_dust(
    n_iterations: int = 5,
    dimension: int = 2
) -> np.ndarray:
    """
    Generate Cantor dust in d dimensions.
    
    Theoretical D = ln(2^d) / ln(3) ≈ 0.631 (d=1), 1.262 (d=2)
    
    Args:
        n_iterations: Number of subdivision iterations
        dimension: Spatial dimension
        
    Returns:
        Point cloud (N, dimension)
    """
    points = np.array([[0.5] * dimension])
    
    for _ in range(n_iterations):
        new_points = []
        for point in points:
            for offset in [0, 2/3]:
                new_point = point.copy()
                new_point[0] = point[0] / 3 + offset / 3
                new_points.append(new_point)
        points = np.array(new_points)
    
    return points


def generate_sierpinski_carpet(n_iterations: int = 4) -> np.ndarray:
    """
    Generate 2D Sierpinski carpet.
    
    Theoretical D = ln(8) / ln(3) ≈ 1.893
    
    Args:
        n_iterations: Subdivision iterations
        
    Returns:
        2D point cloud
    """
    size = 3 ** n_iterations
    carpet = np.ones((size, size), dtype=bool)
    
    for i in range(n_iterations):
        step = 3 ** i
        for x in range(0, size, 3 * step):
            for y in range(0, size, 3 * step):
                carpet[x + step:x + 2*step, y + step:y + 2*step] = False
    
    coords = np.argwhere(carpet).astype(float)
    coords /= size  # Normalize to [0, 1]
    
    return coords


def generate_fractal_line(
    n_points: int = 1000,
    noise_level: float = 0.01
) -> np.ndarray:
    """
    1D line embedded in 3D (D ≈ 1.0).
    
    Args:
        n_points: Number of points
        noise_level: Gaussian noise std
        
    Returns:
        3D point cloud with D₂ ≈ 1.0
    """
    t = np.linspace(0, 10, n_points)
    x = t + np.random.normal(0, noise_level, n_points)
    y = np.random.normal(0, noise_level, n_points)
    z = np.random.normal(0, noise_level, n_points)
    
    return np.column_stack([x, y, z])


def generate_fractal_plane(
    n_points: int = 1000,
    noise_level: float = 0.01
) -> np.ndarray:
    """
    2D plane embedded in 3D (D ≈ 2.0).
    
    Args:
        n_points: Number of points
        noise_level: Out-of-plane noise
        
    Returns:
        3D point cloud with D₂ ≈ 2.0
    """
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = np.random.normal(0, noise_level, n_points)
    
    return np.column_stack([x, y, z])


def generate_etas_synthetic(
    n_points: int = 5000,
    background_rate: float = 0.1,
    productivity: float = 0.05,
    clustering_strength: float = 1.2,
    duration_days: int = 365
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic seismicity using ETAS model.
    
    Epidemic-Type Aftershock Sequence model for testing
    temporal clustering detection.
    
    Args:
        n_points: Target number of events
        background_rate: Background seismicity rate
        productivity: Aftershock productivity
        clustering_strength: Clustering exponent
        duration_days: Simulation duration
        
    Returns:
        (times, magnitudes, coordinates)
    """
    # Simple ETAS implementation
    times = []
    mags = []
    locs = []
    
    # Background events
    n_background = int(background_rate * duration_days)
    bg_times = np.sort(np.random.uniform(0, duration_days, n_background))
    bg_mags = np.random.exponential(1.0, n_background) + 3.0
    bg_locs = np.random.uniform(-10, 10, (n_background, 2))
    
    times.extend(bg_times)
    mags.extend(bg_mags)
    locs.extend(bg_locs)
    
    # Triggered events
    for i, (t, m, loc) in enumerate(zip(bg_times, bg_mags, bg_locs)):
        # Number of aftershocks
        n_aftershocks = np.random.poisson(productivity * 10**(m - 5))
        
        # Aftershock times (Omori decay)
        as_times = t + np.random.pareto(clustering_strength, n_aftershocks)
        as_times = as_times[as_times < duration_days]
        
        # Aftershock magnitudes (G-R)
        as_mags = np.random.exponential(1.0, len(as_times)) + 2.5
        
        # Aftershock locations (clustered around mainshock)
        as_locs = loc + np.random.normal(0, 0.5, (len(as_times), 2))
        
        times.extend(as_times)
        mags.extend(as_mags)
        locs.extend(as_locs)
    
    # Convert to arrays
    times = np.array(times)
    mags = np.array(mags)
    coords_2d = np.array(locs)
    
    # Add depth dimension (uniform for simplicity)
    depths = np.random.uniform(0, 30, len(times))
    coords = np.column_stack([coords_2d, depths])
    
    # Sort by time
    sort_idx = np.argsort(times)
    
    return times[sort_idx], mags[sort_idx], coords[sort_idx]
