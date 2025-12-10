"""
Multifractal analysis module for Seismic Fractal Analysis.
Implements Rényi dimension spectrum D_q for q ∈ [-5, 5].
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict

__all__ = [
    "MultifractalAnalyzer",
    "compute_renyi_spectrum",
    "compute_generalized_dimensions",
]


class MultifractalAnalyzer:
    """
    Compute multifractal spectrum using box-counting and partition methods.
    
    Implements Rényi dimensions:
        D_q = lim_{r→0} [ln(I_q(r)) / ((q-1) ln(r))]
    where I_q(r) = Σ p_i^q for boxes of size r.
    
    Special cases:
        - q = 0: Capacity dimension (box-counting)
        - q = 1: Information dimension
        - q = 2: Correlation dimension (Grassberger-Procaccia)
    """
    
    @staticmethod
    def compute_renyi_spectrum(
        coordinates: np.ndarray,
        q_values: np.ndarray = np.linspace(-5, 5, 51),
        r_min: float = 0.01,
        r_max: float = 0.5,
        n_radii: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Rényi dimension spectrum D_q.
        
        Args:
            coordinates: (N, d) point cloud
            q_values: Array of q values
            r_min: Minimum radius
            r_max: Maximum radius  
            n_radii: Number of radius values
            
        Returns:
            (q_values, D_q_values)
        """
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
        D_q = np.zeros(len(q_values))
        
        N = len(coordinates)
        d = coordinates.shape[1]  # Dimension of space
        
        for idx, q in enumerate(q_values):
            I_q_values = []
            
            for r in radii:
                # BOX-COUNTING METHOD (not ball-counting)
                # Discretize space into grid of hypercubes size r
                
                # Get bounding box
                min_coords = coordinates.min(axis=0)
                max_coords = coordinates.max(axis=0)
                
                # Create box indices for each point (floor division)
                box_indices = np.floor((coordinates - min_coords) / r).astype(int)
                
                # Convert box indices to unique identifiers
                # (each row is d-dimensional box index)
                box_ids = [tuple(idx) for idx in box_indices]
                
                # Count points per box
                from collections import Counter
                box_counts = Counter(box_ids)
                
                # Number of occupied boxes
                N_boxes = len(box_counts)
                
                # Probabilities: p_i = n_i / N_total
                counts_array = np.array(list(box_counts.values()))
                probs = counts_array / N
                
                # Compute I_q based on q value
                if abs(q) < 1e-6:
                    # q=0: Capacity dimension = number of occupied boxes
                    I_q = N_boxes
                elif abs(q - 1.0) < 1e-6:
                    # q=1: Information dimension via Shannon entropy
                    # I_1 = exp(H) where H = -sum(p_i * log(p_i))
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    I_q = np.exp(entropy)
                else:
                    # General Rényi: I_q = sum(p_i^q)
                    I_q = np.sum(probs ** q)
                
                I_q_values.append(max(I_q, 1e-10))  # Avoid log(0)
            
            # Convert to array and take log
            I_q_array = np.array(I_q_values)
            log_r = np.log(radii)
            log_I = np.log(I_q_array)
            
            # Linear fit to get slope
            try:
                slope = np.polyfit(log_r, log_I, 1)[0]
            except:
                D_q[idx] = np.nan
                continue
            
            # Convert slope to fractal dimension
            if abs(q - 1.0) < 1e-6:
                # For q=1 (information): I_1 ~ r^D_1, so D_1 = slope
                D_q[idx] = slope
            elif abs(q) < 1e-6:
                # For q=0 (capacity): N_boxes ~ r^(-D_0), so D_0 = -slope
                D_q[idx] = -slope
            else:
                # General case: I_q ~ r^((q-1)*D_q), so D_q = slope/(q-1)
                D_q[idx] = slope / (q - 1)
        
        return q_values, D_q


def compute_generalized_dimensions(
    coordinates: np.ndarray,
    q_values: np.ndarray = np.array([-2, -1, 0, 1, 2, 3])
) -> Dict[float, float]:
    """
    Compute generalized dimensions for specific q values.
    
    Args:
        coordinates: Point cloud
        q_values: Array of q values to compute
        
    Returns:
        Dictionary mapping q → D_q
    """
    analyzer = MultifractalAnalyzer()
    q_full, D_q_full = analyzer.compute_renyi_spectrum(coordinates, q_values)
    
    return {float(q): float(D) for q, D in zip(q_full, D_q_full)}
