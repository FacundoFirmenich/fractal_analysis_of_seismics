"""
Seismic Fractal Analysis (SFA)
==============================

A comprehensive framework for fractal dimension estimation of seismic catalogs.

References:
    - Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors.
      Physical Review Letters, 50(5), 346-349.
    - Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM.
      Bulletin of the Earthquake Research Institute, 43, 237-239.
"""

__version__ = "2.0.0"
__author__ = "SFA Development Team"
__license__ = "GPL-3.0"

from .core import FractalDimensionEstimator, GeodeticTransformer, SyntheticValidator
from .stats import (
    SeismicityAnalysis,
    SpatialStatisticalAnalysis,
    BayesianRobustness,
    StatisticalInference,
)
from .data import SeismicDataAcquisition, PanAmericanPresets
from .vis import FractalPlotter, StyleManager, AdvancedPlotters

__all__ = [
    "FractalDimensionEstimator",
    "GeodeticTransformer",
    "SyntheticValidator",
    "SeismicityAnalysis",
    "SpatialStatisticalAnalysis",
    "BayesianRobustness",
    "StatisticalInference",
    "SeismicDataAcquisition",
    "PanAmericanPresets",
    "FractalPlotter",
    "StyleManager",
    "AdvancedPlotters",
]