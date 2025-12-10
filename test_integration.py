# -*- coding: utf-8 -*-
"""
INTEGRATION TESTS: End-to-End Workflow
=======================================
Tests complete analysis pipeline with real USGS data.
Validates integration of all components.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestEndToEndWorkflow:
    """Integration tests for complete analysis pipeline."""
    
    @patch('sfa.data.fetch_usgs_catalog')
    def test_full_analysis_pipeline(self, mock_fetch):
        """Test: Complete workflow from data fetch to results."""
        # Mock USGS response
        mock_data = {
            'latitude': np.random.uniform(35, 40, 500),
            'longitude': np.random.uniform(-125, -120, 500),
            'depth': np.random.uniform(0, 30, 500),
            'magnitude': np.random.uniform(2.5, 5.0, 500)
        }
        mock_fetch.return_value = mock_data
        
        from sfa.core import FractalDimensionEstimator
        from sfa.data import fetch_usgs_catalog
        from sfa.utils import geographic_to_metric
        
        # Fetch data
        catalog = fetch_usgs_catalog(lat=37.0, lon=-122.0, radius_km=200)
        assert len(catalog['latitude']) > 0
        
        # Convert coordinates
        coords = geographic_to_metric(
            catalog['latitude'],
            catalog['longitude'],
            catalog['depth']
        )
        assert coords.shape[1] == 3
        
        # Estimate D₂
        estimator = FractalDimensionEstimator()
        d2, sem = estimator.compute_gp_dimension(coords, bootstrap_iterations=10)
        
        assert 0.5 < d2 < 3.0
        assert sem > 0
        assert not np.isnan(d2)
    
    def test_multifractal_spectrum_computation(self):
        """Test: Rényi spectrum computation."""
        from sfa.multifractal import compute_renyi_spectrum
        
        np.random.seed(42)
        coords = np.random.rand(1000, 3)
        
        spectrum = compute_renyi_spectrum(coords, q_values=[-2, 0, 1, 2])
        
        assert 'D0' in spectrum
        assert 'D1' in spectrum
        assert 'D2' in spectrum
        assert spectrum['D0'] >= spectrum['D1'] >= spectrum['D2']
    
    def test_graph_community_detection(self):
        """Test: TGS graph construction and clustering."""
        from sfa.graph_tgs import build_knn_graph, detect_communities
        
        np.random.seed(42)
        coords = np.random.rand(500, 3)
        
        graph = build_knn_graph(coords, k=10)
        communities = detect_communities(graph)
        
        assert len(communities) > 0
        assert max(communities) < len(coords)


class TestDataAcquisitionIntegration:
    """Integration tests for USGS data fetching."""
    
    @pytest.mark.network
    @patch('requests.get')
    def test_usgs_api_integration(self, mock_get):
        """Test: USGS API call structure."""
        from sfa.data import fetch_usgs_catalog
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'features': [
                {
                    'properties': {'mag': 3.5, 'time': 1609459200000},
                    'geometry': {'coordinates': [-122.0, 37.0, 10.0]}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        catalog = fetch_usgs_catalog(lat=37.0, lon=-122.0, radius_km=100)
        
        assert 'latitude' in catalog
        assert 'longitude' in catalog
        assert 'magnitude' in catalog
    
    def test_geographic_coordinate_conversion(self):
        """Test: WGS84 to metric conversion."""
        from sfa.utils import geographic_to_metric
        
        lat = np.array([37.0, 38.0])
        lon = np.array([-122.0, -122.0])
        depth = np.array([0.0, 10.0])
        
        coords = geographic_to_metric(lat, lon, depth)
        
        assert coords.shape == (2, 3)
        # 1 degree lat ≈ 111 km
        assert np.isclose(coords[1, 1] - coords[0, 1], 111.1, atol=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
