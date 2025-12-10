"""
Test suite for Mad Heuristic (Heurística del Loco)
Validates subversive tail analysis on synthetic and seismic data
"""

import numpy as np
import pytest
from sfa.mad_heuristic import MadHeuristic, SeismicMadAnalysis, ChaoticMadHeuristic


class TestMadHeuristic:
    """Test core Mad Heuristic functionality."""

    def test_symmetric_tail_exploration(self):
        """Test symmetric 20% tail extraction."""
        # Generate normal distribution
        np.random.seed(42)
        data = np.random.randn(1000)

        mad = MadHeuristic(tail_proportion=0.20, left_weight=0.5, right_weight=0.5)
        results = mad.explore_tails(data)

        # Should extract ~10% from each tail
        assert len(results["left_tail"]) > 0
        assert len(results["right_tail"]) > 0
        assert len(results["left_tail"]) + len(results["right_tail"]) <= 200

        # Tails should contain extremes
        assert np.max(results["left_tail"]) < np.median(data)
        assert np.min(results["right_tail"]) > np.median(data)

    def test_asymmetric_weighting(self):
        """Test asymmetric tail weighting (e.g., 70% right, 30% left)."""
        np.random.seed(42)
        data = np.random.randn(1000)

        mad = MadHeuristic(tail_proportion=0.20, left_weight=0.3, right_weight=0.7)
        results = mad.explore_tails(data)

        # Right tail should be larger
        assert len(results["right_tail"]) > len(results["left_tail"])

        # Verify proportions approximately correct
        left_ratio = len(results["left_tail"]) / len(data)
        right_ratio = len(results["right_tail"]) / len(data)

        assert left_ratio < 0.10  # ~6% (0.20 * 0.3)
        assert right_ratio > left_ratio

    def test_inverse_transformation(self):
        """Test 1/x perspective transformation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0])

        mad = MadHeuristic(
            tail_proportion=0.25,
            left_weight=0.5,
            right_weight=0.5,
            inverse_transform=True,
        )
        results = mad.explore_tails(data)

        # Check transformation exists
        assert "left_transformed" in results
        assert "right_transformed" in results

        # Inverse should reverse order of magnitude
        if len(results["right_transformed"]) > 0:
            assert results["right_transformed"][0] < results["right_transformed"][-1]

    def test_mad_score_calculation(self):
        """Test Mad Score computation."""
        # Gaussian data (boring)
        np.random.seed(42)
        gaussian = np.random.randn(1000)

        # Heavy-tailed data (interesting)
        cauchy = np.random.standard_cauchy(1000)
        cauchy = np.clip(cauchy, -100, 100)  # Clip extreme outliers

        mad = MadHeuristic(tail_proportion=0.15)

        gaussian_results = mad.explore_tails(gaussian)
        cauchy_results = mad.explore_tails(cauchy)

        # Cauchy (heavy-tailed) should have higher Mad Score
        assert cauchy_results["mad_score"] > gaussian_results["mad_score"]

        # Gaussian should have moderate score
        assert 0.5 < gaussian_results["mad_score"] < 2.5

    def test_subversive_filter(self):
        """Test subversive filtering (keep outliers, reject bulk)."""
        np.random.seed(42)
        data = np.random.randn(1000)

        mad = MadHeuristic(tail_proportion=0.10)
        outliers, indices = mad.subversive_filter(data)

        # Should extract ~10% of data
        assert len(outliers) <= 100
        assert len(indices) == len(outliers)

        # Outliers should be extreme values
        assert np.abs(outliers).mean() > np.abs(data).mean()


class TestSeismicMadAnalysis:
    """Test seismic-specific applications."""

    def test_megaquake_detection(self):
        """Test megaquake regime detection."""
        # Simulate Gutenberg-Richter distribution
        np.random.seed(42)
        b_value = 1.0
        magnitudes = -np.log10(np.random.rand(1000)) / b_value + 4.0

        mega_results = SeismicMadAnalysis.detect_megaquake_regime(
            magnitudes, tail_proportion=0.05
        )

        # Should identify top 5% as megaquakes
        assert mega_results["megaquake_count"] <= 50
        assert mega_results["megaquake_threshold"] > np.median(magnitudes)

        # All megaquakes should be above threshold
        assert np.all(
            mega_results["megaquake_magnitudes"] >= mega_results["megaquake_threshold"]
        )

    def test_depth_anomalies_symmetric(self):
        """Test symmetric depth anomaly detection."""
        # Simulate depth distribution with shallow crust + uppermantle
        np.random.seed(42)
        shallow = np.random.exponential(scale=10, size=400)  # 0-~40 km
        deep = np.random.normal(loc=100, scale=20, size=100)  # Deep events
        depths = np.concatenate([shallow, deep])

        depth_results = SeismicMadAnalysis.detect_depth_anomalies(depths, asymmetry=0.5)

        # Should detect both shallow and deep anomalies
        assert len(depth_results["shallow_anomalies"]) > 0
        assert len(depth_results["deep_anomalies"]) > 0

        # Thresholds should make sense
        assert depth_results["shallow_threshold"] < 10  # Very shallow
        assert depth_results["deep_threshold"] > np.median(depths)

    def test_fractal_extremes_detection(self):
        """Test D2 extremes (planar vs volumetric)."""
        # Simulate D2 values with extremes
        np.random.seed(42)
        normal_d2 = np.random.normal(loc=2.3, scale=0.2, size=100)
        planar_extreme = np.array([1.2, 1.5, 1.7])  # Anomalously low
        volumetric_extreme = np.array([2.9, 2.95, 3.0])  # Anomalously high

        d2_values = np.concatenate([normal_d2, planar_extreme, volumetric_extreme])

        d2_results = SeismicMadAnalysis.detect_fractal_extremes(
            d2_values, target="both"
        )

        # Should detect both extremes
        assert len(d2_results["planar_extreme"]) > 0
        assert len(d2_results["volumetric_extreme"]) > 0

        # Planar should be < median, volumetric > median
        median_d2 = np.median(d2_values)
        assert np.max(d2_results["planar_extreme"]) < median_d2
        assert np.min(d2_results["volumetric_extreme"]) > median_d2


class TestChaoticMode:
    """Test chaotic subversion mode."""

    def test_chaotic_sampling(self):
        """Test chaotic tail sampling with randomness."""
        np.random.seed(42)
        data = np.random.randn(1000)

        chaos_mad = ChaoticMadHeuristic(tail_proportion=0.15)

        chaotic_samples = chaos_mad.chaotic_sample(data, n_samples=50, randomness=0.3)

        # Should return requested number of samples
        assert len(chaotic_samples) == 50

        # Should contain tail values
        tail_results = chaos_mad.explore_tails(data)
        tail_values = np.concatenate(
            [tail_results["left_tail"], tail_results["right_tail"]]
        )

        # At least some samples should be from tails
        assert np.any(np.isin(chaotic_samples, tail_values))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Test error on insufficient data."""
        small_data = np.array([1, 2, 3])
        mad = MadHeuristic()

        with pytest.raises(ValueError):
            mad.explore_tails(small_data)

    def test_invalid_tail_proportion(self):
        """Test error on invalid tail proportion."""
        with pytest.raises(ValueError):
            MadHeuristic(tail_proportion=0.6)  # > 0.5

    def test_invalid_weights(self):
        """Test error on weights not summing to 1.0."""
        with pytest.raises(ValueError):
            MadHeuristic(left_weight=0.3, right_weight=0.5)  # Sum ≠ 1.0

    def test_constant_data(self):
        """Test handling of constant data (zero variance)."""
        constant_data = np.ones(100)
        mad = MadHeuristic()

        results = mad.explore_tails(constant_data)

        # Mad score should be 0 or NaN for constant data
        assert results["mad_score"] == 0.0 or np.isnan(results["mad_score"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
