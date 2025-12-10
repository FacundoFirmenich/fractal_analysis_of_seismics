"""
Statistical analysis module for Seismic Fractal Analysis.
Includes:
- Seismicity Analysis (Gutenberg-Richter)
- Spatial Statistics (Clark-Evans)
- Statistical Inference (Hypothesis testing)
- Bayesian Robustness (Hierarchical modeling)
"""

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.cluster import DBSCAN
from typing import Dict, Any, Tuple, Optional


class SeismicityAnalysis:
    """Magnitude-frequency and completeness analysis."""

    @staticmethod
    def estimate_completeness_magnitude(
        magnitudes: np.ndarray,
    ) -> Tuple[float, float]:
        """Maximum curvature method for Mc estimation."""
        if len(magnitudes) == 0:
            return 0.0, 0.0

        bins = np.arange(
            np.floor(np.min(magnitudes)),
            np.ceil(np.max(magnitudes)) + 0.1,
            0.1,
        )
        hist, bin_edges = np.histogram(magnitudes, bins=bins)

        mc_index = np.argmax(hist)
        mc = bin_edges[mc_index] + 0.05
        completeness_fraction = np.sum(magnitudes >= mc) / len(magnitudes)

        return mc, completeness_fraction

    def compute_b_value(
        self,
        magnitudes: np.ndarray,
        completeness_magnitude: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Aki-Utsu MLE b-value estimation."""
        if completeness_magnitude is None:
            completeness_magnitude, _ = self.estimate_completeness_magnitude(magnitudes)

        complete_magnitudes = magnitudes[magnitudes >= completeness_magnitude]
        if len(complete_magnitudes) < 50:
            return np.nan, np.nan, completeness_magnitude

        mean_magnitude = np.mean(complete_magnitudes)
        # b = log10(e) / (mean - (Mc - delta/2))
        b_value = np.log10(np.exp(1)) / (
            mean_magnitude - (completeness_magnitude - 0.05)
        )
        b_uncertainty = b_value / np.sqrt(len(complete_magnitudes))

        return b_value, b_uncertainty, completeness_magnitude


class SpatialStatisticalAnalysis:
    """Clark-Evans nearest neighbor analysis."""

    @staticmethod
    def clark_evans_3d(coordinates: np.ndarray) -> float:
        """Three-Dimensional Clark-Evans clustering index."""
        if len(coordinates) < 10:
            return np.nan

        # Subsample for performance if needed
        if len(coordinates) > 5000:
            sample_indices = np.random.choice(len(coordinates), 5000, replace=False)
            coords_sample = coordinates[sample_indices]
        else:
            coords_sample = coordinates

        tree = spatial.cKDTree(coords_sample)
        distances, _ = tree.query(coords_sample, k=2)
        mean_nearest_neighbor = np.mean(distances[:, 1])

        minima = coords_sample.min(axis=0)
        maxima = coords_sample.max(axis=0)
        volume = np.prod(maxima - minima)

        if volume <= 0:
            return np.nan

        density = len(coords_sample) / volume
        expected_distance = 0.554 / (density ** (1 / 3))  # Diggle (2003)

        return mean_nearest_neighbor / expected_distance

    @staticmethod
    def analyze_planar_clustering(
        coordinates: np.ndarray, eps: float = 0.05, min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Identify planar clusters using DBSCAN to validate multi-planar
        hypothesis.

        Args:
            coordinates: Normalized coordinates (N, 3).
            eps: The maximum distance between two samples for one to be
                 considered as in the neighborhood of the other.
            min_samples: The number of samples (or total weight) in a
                         neighborhood for a point to be considered as a core
                         point.

        Returns:
            Dictionary with cluster statistics.
        """
        if len(coordinates) < min_samples:
            return {"n_clusters": 0, "noise_ratio": 1.0}

        # Run DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(coordinates),
            "labels": labels,
        }


class BayesianRobustness:
    """Hierarchical Bayesian inference for D2 estimation."""

    @staticmethod
    def hierarchical_bayesian_d2(
        bootstrap_estimates_by_region: Dict[str, np.ndarray],
        n_samples: int = 5000,
        n_warmup: int = 1000,
    ) -> Dict[str, Any]:
        """
        Hierarchical Bayesian model with Metropolis-Hastings sampling.
        Estimates global and regional means/variances.
        """
        region_names = list(bootstrap_estimates_by_region.keys())
        # n_regions = len(region_names) # Unused

        # Initialize parameters
        mu_global = 1.5
        tau_global = 0.5
        mu_regions = np.array(
            [np.mean(bootstrap_estimates_by_region[r]) for r in region_names]
        )
        sigma_regions = np.array(
            [np.std(bootstrap_estimates_by_region[r], ddof=1) for r in region_names]
        )

        # Storage
        posterior_mu_global = []
        posterior_mu_regions = {r: [] for r in region_names}

        # Metropolis-Hastings loop (Simplified for brevity/performance in
        # Python)
        # Note: In a full production system, we might use PyMC or Stan.
        # Here we implement a basic MH sampler as per the original validated
        # code.

        for iteration in range(n_warmup + n_samples):
            # 1. Update Global Mean
            proposal_mu_global = mu_global + np.random.normal(0, 0.1)

            # Likelihood of regions given global
            ll_current = -0.5 * np.sum((mu_regions - mu_global) ** 2 / tau_global**2)
            ll_proposal = -0.5 * np.sum(
                (mu_regions - proposal_mu_global) ** 2 / tau_global**2
            )

            # Prior on global mean (Normal(1.5, 1.0))
            prior_current = -0.5 * ((mu_global - 1.5) ** 2)
            prior_proposal = -0.5 * ((proposal_mu_global - 1.5) ** 2)

            if np.log(np.random.rand()) < (
                ll_proposal + prior_proposal - ll_current - prior_current
            ):
                mu_global = proposal_mu_global

            # 2. Update Region Means
            for i, region in enumerate(region_names):
                data = bootstrap_estimates_by_region[region]
                proposal_mu = mu_regions[i] + np.random.normal(0, 0.05)

                # Likelihood of data given region mean
                ll_data_curr = -0.5 * np.sum(
                    (data - mu_regions[i]) ** 2 / sigma_regions[i] ** 2
                )
                ll_data_prop = -0.5 * np.sum(
                    (data - proposal_mu) ** 2 / sigma_regions[i] ** 2
                )

                # Prior of region mean given global
                prior_reg_curr = -0.5 * (
                    (mu_regions[i] - mu_global) ** 2 / tau_global**2
                )
                prior_reg_prop = -0.5 * ((proposal_mu - mu_global) ** 2 / tau_global**2)

                if np.log(np.random.rand()) < (
                    ll_data_prop + prior_reg_prop - ll_data_curr - prior_reg_curr
                ):
                    mu_regions[i] = proposal_mu

            # Store samples
            if iteration >= n_warmup:
                posterior_mu_global.append(mu_global)
                for i, region in enumerate(region_names):
                    posterior_mu_regions[region].append(mu_regions[i])

        # Compute summary statistics
        credible_intervals = {}
        for region in region_names:
            samples = np.array(posterior_mu_regions[region])
            lower = np.percentile(samples, 2.5)
            upper = np.percentile(samples, 97.5)
            credible_intervals[region] = (lower, upper)

        return {
            "posterior_mu_global": np.array(posterior_mu_global),
            "posterior_mu_regions": {
                r: np.array(v) for r, v in posterior_mu_regions.items()
            },
            "credible_intervals_95": credible_intervals,
        }

    @staticmethod
    def morans_i_depth(coordinates: np.ndarray, k: int = 10) -> Tuple[float, float]:
        """
        Compute Moran's I spatial autocorrelation index for depth.
        Tests whether earthquake depths are spatially clustered.

        Args:
            coordinates: (N, 3) array with [x, y, depth]
            k: Number of nearest neighbors to consider

        Returns:
            Tuple of (Moran's_I, p_value)
        """
        from scipy import stats
        from sklearn.neighbors import NearestNeighbors

        coords = np.asarray(coordinates, float)
        n = len(coords)
        if n < k + 2:
            return np.nan, 1.0

        # Subsample for performance if dataset is large
        if n > 2500:
            idx = np.random.choice(n, 2500, replace=False)
            coords = coords[idx]
            n = len(coords)

        depths = coords[:, 2]

        # Build k-NN spatial weights
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        dists, indices = nbrs.kneighbors(coords)

        # Create weighted adjacency matrix
        W = np.zeros((n, n), float)
        for i in range(n):
            neigh = indices[i, 1:]  # Exclude self
            w = 1.0 / (dists[i, 1:] + 1e-6)  # Inverse distance weights
            W[i, neigh] = w

        # Row-normalize weights
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums[:, None]

        # Compute Moran's I
        z = depths - depths.mean()
        z2 = np.sum(z**2)
        if z2 == 0:
            return np.nan, 1.0

        moran_i = n * np.sum(W * np.outer(z, z)) / (W.sum() * z2)

        # Z-score under null hypothesis of no spatial autocorrelation
        # Using exact variance from Cliff & Ord (1981), eq. 5.7
        EI = -1.0 / (n - 1)

        # Compute weight moments for exact variance
        W_sum = W.sum()
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        W_rowsum = W.sum(axis=1)
        W_colsum = W.sum(axis=0)
        S2 = np.sum((W_rowsum + W_colsum) ** 2)

        # Exact variance under randomization assumption
        # VI = [n*S1 - 2*n*W² + 6*W_sum²] / [(n-1)(n+1)W_sum²] - EI²
        numerator = n * S1 - 2 * n * np.sum(W**2) + 6 * W_sum**2
        denominator = (n - 1) * (n + 1) * W_sum**2
        VI = (numerator / denominator) - EI**2

        z_score = (moran_i - EI) / np.sqrt(max(VI, 1e-10))
        p = 2 * (1.0 - stats.norm.cdf(abs(z_score)))

        return float(moran_i), float(p)


class StatisticalInference:
    """
    Effect size estimation and hypothesis testing for regional comparisons.
    """

    @staticmethod
    def hedges_g(
        mean1: float,
        std1: float,
        n1: int,
        mean2: float,
        std2: float,
        n2: int,
    ) -> float:
        """
        Compute Hedges' g effect size (bias-corrected Cohen's d).

        Args:
            mean1, std1, n1: Mean, std, and sample size for group 1
            mean2, std2, n2: Mean, std, and sample size for group 2

        Returns:
            Hedges' g effect size
        """
        # Pooled standard deviation
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        if pooled_var <= 0:
            return 0.0
        pooled_sd = np.sqrt(pooled_var)
        if pooled_sd == 0:
            return 0.0

        # Cohen's d
        d = (mean2 - mean1) / pooled_sd

        # Bias correction factor J
        df = n1 + n2 - 2
        if df > 1:
            J = 1.0 - 3.0 / (4.0 * df - 1.0)
        else:
            J = 1.0

        # Hedges' g
        g = d * J

        # Cap extreme values
        if abs(g) > 3.0:
            g = np.copysign(3.0, g)

        return float(g)

    @staticmethod
    def interpret_effect(g: float) -> str:
        """
        Interpret Hedges' g effect size magnitude.

        Args:
            g: Hedges' g value

        Returns:
            Verbal interpretation
        """
        a = abs(g)
        if a < 0.2:
            return "negligible"
        if a < 0.5:
            return "small"
        if a < 0.8:
            return "medium"
        return "large"

    @staticmethod
    def pairwise_comparisons(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform pairwise comparisons of D2 between all regions.

        Args:
            results: Dictionary mapping region name to results dict
                     (must contain 'd2_mean', 'd2_std', 'event_count')

        Returns:
            DataFrame with pairwise comparison results
        """
        regions = list(results.keys())
        rows = []

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1_name = regions[i]
                r2_name = regions[j]

                r1 = results[r1_name]
                r2 = results[r2_name]

                # Skip if D2 not available
                if not (
                    np.isfinite(r1.get("d2_mean", np.nan))
                    and np.isfinite(r2.get("d2_mean", np.nan))
                ):
                    continue

                # Compute Hedges' g
                g = StatisticalInference.hedges_g(
                    r1["d2_mean"],
                    r1.get("d2_std", 0.1),
                    r1.get("event_count", 1000),
                    r2["d2_mean"],
                    r2.get("d2_std", 0.1),
                    r2.get("event_count", 1000),
                )

                rows.append(
                    {
                        "Comparison": f"{r2_name} vs {r1_name}",
                        "Region_1": r1_name,
                        "Region_2": r2_name,
                        "D2_Region_1": r1["d2_mean"],
                        "D2_Region_2": r2["d2_mean"],
                        "Delta_D2": r2["d2_mean"] - r1["d2_mean"],
                        "Hedges_g": g,
                        "Magnitude": StatisticalInference.interpret_effect(g),
                    }
                )

        return pd.DataFrame(rows)
