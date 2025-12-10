# -*- coding: utf-8 -*-
"""
UNIT TESTS: Core Fractal Dimension Estimation
==============================================
Comprehensive unit tests for sfa/core.py module.
Tests individual functions in isolation with mocks.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestFractalDimensionEstimator:
    """Unit tests for FractalDimensionEstimator class."""
    
    @pytest.fixture
    def estimator(self):
        """Fixture: Create estimator instance."""
        from sfa.core import FractalDimensionEstimator
        return FractalDimensionEstimator()
    
    def test_initialization(self, estimator):
        """Test: Estimator initializes correctly."""
        assert estimator is not None
        assert hasattr(estimator, 'compute_gp_dimension')
    
    def test_theil_sen_slope_basic(self, estimator):
        """Test: Theil-Sen regression on perfect linear data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # slope = 2
        
        slope = estimator._theil_sen_slope(x, y)
        
        assert np.isclose(slope, 2.0, atol=0.01)
    
    def test_theil_sen_slope_with_noise(self, estimator):
        """Test: Theil-Sen robust to outliers."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * x + np.random.normal(0, 0.1, 100)
        y[50] = 100  # Outlier
        
        slope = estimator._theil_sen_slope(x, y)
        
        assert 2.4 < slope < 2.6  # Should be robust to outlier
    
    def test_compute_gp_dimension_1d(self, estimator):
        """Test: D₂ ≈ 1.0 for 1D line."""
        np.random.seed(42)
        n = 2000
        coords = np.column_stack([
            np.linspace(0, 10, n),
            np.random.normal(0, 0.01, n),
            np.random.normal(0, 0.01, n)
        ])
        
        d2, sem = estimator.compute_gp_dimension(coords, bootstrap_iterations=10)
        
        assert 0.9 < d2 < 1.1
        assert sem < 0.1
    
    def test_compute_gp_dimension_2d(self, estimator):
        """Test: D₂ ≈ 2.0 for 2D plane."""
        np.random.seed(42)
        n = 2000
        coords = np.column_stack([
            np.random.uniform(0, 10, n),
            np.random.uniform(0, 10, n),
            np.random.normal(0, 0.01, n)
        ])
        
        d2, sem = estimator.compute_gp_dimension(coords, bootstrap_iterations=10)
        
        assert 1.8 < d2 < 2.2
        assert sem < 0.1
    
    def test_compute_gp_dimension_insufficient_data(self, estimator):
        """Test: Raises error with too few points."""
        coords = np.random.rand(50, 3)
        
        with pytest.raises(ValueError):
            estimator.compute_gp_dimension(coords)
    
    def test_bootstrap_reproducibility(self, estimator):
        """Test: Fixed seed gives reproducible results."""
        np.random.seed(42)
        coords = np.random.rand(1000, 3)
        
        d2_1, _ = estimator.compute_gp_dimension(coords, bootstrap_iterations=5, random_seed=42)
        d2_2, _ = estimator.compute_gp_dimension(coords, bootstrap_iterations=5, random_seed=42)
        
        assert np.isclose(d2_1, d2_2, atol=1e-10)


class TestDataNormalization:
    """Unit tests for coordinate normalization."""
    
    def test_normalize_preserves_aspect_ratio(self):
        """Test: Normalization preserves geometric aspect ratio."""
        from sfa.utils import normalize_coordinates
        
        coords = np.array([
            [0, 0, 0],
            [10, 5, 2]
        ])
        
        normalized = normalize_coordinates(coords)
        
        # Check range is [0, 1]
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
        # Check aspect ratio preserved (max range same for all dims)
        assert normalized[1, 0] == 1.0  # Longest dimension spans [0,1]
    
    def test_normalize_handles_zero_range(self):
        """Test: Handles degenerate case of zero range."""
        from sfa.utils import normalize_coordinates
        
        coords = np.array([
            [5, 5, 5],
            [5, 5, 5]
        ])
        
        # Should not raise, should return valid array
        normalized = normalize_coordinates(coords)
        assert normalized.shape == coords.shape


class TestStatisticalMethods:
    """Unit tests for statistical inference functions."""
    
    def test_bootstrap_confidence_interval(self):
        """Test: Bootstrap CI contains true mean."""
        from sfa.stats import bootstrap_confidence_interval
        
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)
        
        lower, upper = bootstrap_confidence_interval(data, n_iterations=100, confidence=0.95)
        
        assert lower < 10 < upper
        assert upper - lower < 1.0  # Reasonable width
    
    def test_hedges_g_effect_size(self):
        """Test: Hedges' g calculation."""
        from sfa.stats import hedges_g_effect_size
        
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        
        g = hedges_g_effect_size(group1, group2)
        
        assert g > 2.0  # Large effect size expected
        assert not np.isnan(g)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
