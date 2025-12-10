"""
GRAPH + THEORY OF GRAPH SIGNALS (TGS) FOR SFA v2.5
===================================================
Based on resultsyfeedbacks.txt implementation proposal.

Features:
- Seismic graph construction (k-NN spatial proximity)
- Graph Fourier Transform (GFT) via Laplacian eigendecomposition
- Leiden/Louvain community detection (fault segments)
- Graph fractal dimension (box-covering method)
- Graph signal anomaly detection (magnitude/depth signals)
- Sparse representation for large networks

Author: Facundo Firmenich
Date: 2025-12-07
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import warnings

# NetworkX for graph construction
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Install with: pip install networkx")

# igraph for faster computations
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    warnings.warn("igraph not available. Install with: pip install python-igraph")

# Leiden algorithm (preferred over Louvain)
try:
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    # Fallback to Louvain
    try:
        import community as community_louvain
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False
        warnings.warn("Neither leidenalg nor python-louvain available. Community detection disabled.")

from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


class SeismicGraphTGS:
    """
    Seismic Graph with Theory of Graph Signals (TGS) analysis.
    
    Constructs spatial proximity graph from earthquake catalog coordinates
    and applies graph signal processing techniques for:
    - Community detection (fault segment identification)
    - Fractal dimension estimation (graph box-covering)
    - Anomaly detection via graph filtering
    
    Attributes:
        coordinates (np.ndarray): Event coordinates (N, 3) [x, y, z] or (N, 2)
        magnitudes (np.ndarray): Event magnitudes (N,)
        depths (np.ndarray): Event depths (N,)
        k (int): Number of nearest neighbors for graph construction
        G_nx (nx.Graph): NetworkX graph (if available)
        G_ig (ig.Graph): igraph graph (if available, faster)
        L (np.ndarray): Graph Laplacian matrix
        eigenvals (np.ndarray): Laplacian eigenvalues
        eigenvects (np.ndarray): Laplacian eigenvectors
        communities (dict): Community assignments
    """
    
    def __init__(
        self,
        coordinates: np.ndarray,
        magnitudes: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        k: int = 10,
        metric: str = 'euclidean'
    ):
        """
        Initialize seismic graph.
        
        Parameters:
            coordinates: Event locations (N, 2) or (N, 3)
            magnitudes: Event magnitudes (N,), optional
            depths: Event depths (N,), optional
            k: Number of nearest neighbors
            metric: Distance metric ('euclidean', 'manhattan', etc.)
        """
        self.coords = coordinates
        self.magnitudes = magnitudes if magnitudes is not None else np.zeros(len(coordinates))
        self.depths = depths if depths is not None else np.zeros(len(coordinates))
        self.k = min(k, len(coordinates) - 1)  # Ensure k < N
        self.metric = metric
        
        # Graph objects
        self.G_nx = None
        self.G_ig = None
        
        # Spectral objects
        self.L = None
        self.eigenvals = None
        self.eigenvects = None
        
        # Communities
        self.communities = None
    
    def build_knn_graph(self) -> Optional[nx.Graph]:
        """
        Construct k-nearest neighbors graph.
        
        Returns:
            NetworkX graph (if available), otherwise None
        """
        if not NETWORKX_AVAILABLE:
            warnings.warn("NetworkX required for graph construction")
            return None
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k+1, metric=self.metric).fit(self.coords)
        distances, indices = nbrs.kneighbors(self.coords)
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(len(self.coords)))
        
        for i in range(len(self.coords)):
            for j_idx in range(1, len(indices[i])):  # Skip self (index 0)
                j = indices[i][j_idx]
                dist = distances[i][j_idx]
                if i < j:  # Avoid duplicate edges
                    G.add_edge(i, j, weight=dist)
        
        self.G_nx = G
        
        # Convert to igraph if available (faster for large graphs)
        if IGRAPH_AVAILABLE:
            edges = list(G.edges())
            weights = [G[u][v]['weight'] for u, v in edges]
            self.G_ig = ig.Graph(len(G), edges, edge_attrs={'weight': weights})
        
        return G
    
    def compute_laplacian(self, normalized: bool = True) -> np.ndarray:
        """
        Compute graph Laplacian matrix.
        
        Parameters:
            normalized: If True, compute normalized Laplacian L = I - D^(-1/2) A D^(-1/2)
                       If False, combinatorial Laplacian L = D - A
        
        Returns:
            Laplacian matrix (sparse or dense depending on graph size)
        """
        if self.G_ig is None and self.G_nx is None:
            self.build_knn_graph()
        
        # Use igraph if available (faster)
        if self.G_ig is not None and IGRAPH_AVAILABLE:
            self.L = np.array(self.G_ig.laplacian(normalized=normalized))
        elif self.G_nx is not None:
            if normalized:
                self.L = nx.normalized_laplacian_matrix(self.G_nx).toarray()
            else:
                self.L = nx.laplacian_matrix(self.G_nx).toarray()
        else:
            warnings.warn("No graph available for Laplacian computation")
            return None
        
        return self.L
    
    def graph_fourier_transform(self, signal: np.ndarray, n_eigs: int = 100) -> np.ndarray:
        """
        Apply Graph Fourier Transform to signal on graph.
        
        GFT decomposes signal into graph frequency components using
        Laplacian eigenvectors as Fourier basis.
        
        Parameters:
            signal: Signal on graph nodes (N,)
            n_eigs: Number of eigenvalues/vectors to compute
        
        Returns:
            GFT coefficients (frequency domain)
        """
        if self.L is None:
            self.compute_laplacian(normalized=True)
        
        # Compute eigendecomposition if not cached
        if self.eigenvals is None:
            n_eigs_actual = min(n_eigs, self.L.shape[0] - 2)
            try:
                self.eigenvals, self.eigenvects = eigsh(self.L, k=n_eigs_actual, which='SM')
                # Sort by eigenvalue
                sort_idx = np.argsort(self.eigenvals)
                self.eigenvals = self.eigenvals[sort_idx]
                self.eigenvects = self.eigenvects[:, sort_idx]
            except Exception as e:
                warnings.warn(f"Eigendecomposition failed: {e}")
                return np.array([])
        
        # GFT: project signal onto eigenvector basis
        gft_coeffs = self.eigenvects.T @ signal
        return gft_coeffs.real
    
    def detect_communities(self, method: str = 'leiden') -> Dict:
        """
        Detect communities (fault segments) in seismic graph.
        
        Parameters:
            method: 'leiden' (preferred), 'louvain', or 'spectral'
        
        Returns:
            Dictionary of community assignments {node_id: community_id}
        """
        if self.G_ig is None and self.G_nx is None:
            self.build_knn_graph()
        
        if method == 'leiden' and LEIDEN_AVAILABLE and self.G_ig is not None:
            # Leiden algorithm (state-of-art)
            partition = leidenalg.find_partition(
                self.G_ig,
                leidenalg.ModularityVertexPartition
            )
            self.communities = {i: partition.membership[i] for i in range(len(partition.membership))}
        
        elif method == 'louvain' and LOUVAIN_AVAILABLE and self.G_nx is not None:
            # Louvain algorithm (classic)
            self.communities = community_louvain.best_partition(self.G_nx)
        
        elif method == 'spectral' or (not LEIDEN_AVAILABLE and not LOUVAIN_AVAILABLE):
            # Fallback: spectral clustering via Laplacian
            if self.eigenvals is None:
                self.compute_laplacian(normalized=True)
                self.graph_fourier_transform(self.magnitudes, n_eigs=10)
            
            # Use Fiedler vector (2nd eigenvector) for binary partition
            if len(self.eigenvals) >= 2:
                fiedler = self.eigenvects[:, 1]
                self.communities = {i: int(fiedler[i] > 0) for i in range(len(fiedler))}
            else:
                self.communities = {i: 0 for i in range(len(self.coords))}
        
        else:
            warnings.warn(f"Method '{method}' not available. Install leidenalg or python-louvain.")
            self.communities = {i: 0 for i in range(len(self.coords))}
        
        return self.communities
    
    def graph_signal_anomaly(self, signal_type: str = 'magnitude', low_pass_ratio: float = 0.1) -> np.ndarray:
        """
        Detect anomalies in graph signal via low-pass filtering.
        
        Anomalies are events whose signal (magnitude/depth) deviates
        significantly from smooth graph trend.
        
        Parameters:
            signal_type: 'magnitude' or 'depth'
            low_pass_ratio: Fraction of eigenvectors for smooth reconstruction
        
        Returns:
            Residual magnitudes (high residual = anomaly)
        """
        # Select signal
        if signal_type == 'magnitude':
            signal = self.magnitudes
        elif signal_type == 'depth':
            signal = self.depths
        else:
            signal = signal_type  # Assume array passed
        
        # Apply GFT
        coeffs = self.graph_fourier_transform(signal)
        
        if len(coeffs) == 0:
            return np.zeros_like(signal)
        
        # Low-pass filter: keep only k lowest frequencies
        k_smooth = max(1, int(len(coeffs) * low_pass_ratio))
        coeffs_smooth = np.zeros_like(coeffs)
        coeffs_smooth[:k_smooth] = coeffs[:k_smooth]
        
        # Inverse GFT (reconstruct smooth signal)
        signal_recon = self.eigenvects @ coeffs_smooth
        
        # Residuals = anomalies
        residuals = np.abs(signal - signal_recon.real)
        
        return residuals
    
    def fractal_dimension_graph(self, method: str = 'box_covering') -> float:
        """
        Estimate fractal dimension of graph structure.
        
        Uses box-covering algorithm: N_boxes(l) ~ l^(-D_graph)
        
        Parameters:
            method: 'box_covering' (only method implemented)
        
        Returns:
            Graph fractal dimension D_graph
        """
        if self.G_nx is None:
            self.build_knn_graph()
        
        if not NETWORKX_AVAILABLE:
            return np.nan
        
        try:
            # Simplified box-covering for computational efficiency
            # Use graph distance (shortest path) for boxes
            
            # Sample box sizes (log-spaced radii)
            max_dist = nx.diameter(self.G_nx) if nx.is_connected(self.G_nx) else 10
            l_values = np.logspace(0, np.log10(max_dist), num=5, dtype=int)
            l_values = np.unique(l_values[l_values > 0])
            
            if len(l_values) < 2:
                return np.nan
            
            N_boxes = []
            for l in l_values:
                # Greedy box covering: iteratively select nodes and exclude l-neighborhood
                remaining = set(self.G_nx.nodes())
                boxes = 0
                
                while remaining:
                    # Pick arbitrary node
                    center = next(iter(remaining))
                    
                    # Find l-neighborhood
                    try:
                        lengths = nx.single_source_shortest_path_length(self.G_nx, center, cutoff=int(l))
                        neighborhood = set(lengths.keys())
                    except:
                        neighborhood = {center}
                    
                    # Remove neighborhood from remaining
                    remaining -= neighborhood
                    boxes += 1
                
                N_boxes.append(boxes)
            
            # Fit power law: N_boxes ~ l^(-D)
            log_l = np.log(l_values)
            log_N = np.log(N_boxes)
            
            # Linear fit
            coeffs = np.polyfit(log_l, log_N, 1)
            D_graph = -coeffs[0]  # Slope is -D
            
            # Clamp to physical range
            D_graph = np.clip(D_graph, 0.5, 3.0)
            
            return D_graph
        
        except Exception as e:
            warnings.warn(f"Graph fractal dimension computation failed: {e}")
            return np.nan
    
    def summary_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive graph statistics.
        
        Returns:
            Dictionary with:
            - n_nodes, n_edges
            - avg_degree, degree_std
            - n_communities
            - D_graph (fractal dimension)
            - spectral_gap (λ₂ - λ₁, stability indicator)
            - clustering_coefficient
        """
        stats = {}
        
        if self.G_nx is None:
            self.build_knn_graph()
        
        if self.G_nx is not None:
            stats['n_nodes'] = self.G_nx.number_of_nodes()
            stats['n_edges'] = self.G_nx.number_of_edges()
            degrees = [d for n, d in self.G_nx.degree()]
            stats['avg_degree'] = np.mean(degrees)
            stats['degree_std'] = np.std(degrees)
            
            try:
                stats['clustering_coefficient'] = nx.average_clustering(self.G_nx)
            except:
                stats['clustering_coefficient'] = np.nan
        
        # Communities
        if self.communities is None:
            self.detect_communities()
        
        if self.communities is not None:
            stats['n_communities'] = len(set(self.communities.values()))
        else:
            stats['n_communities'] = 0
        
        # Fractal dimension
        try:
            stats['D_graph'] = self.fractal_dimension_graph()
        except:
            stats['D_graph'] = np.nan
        
        # Spectral gap
        if self.eigenvals is None and self.L is not None:
            try:
                self.graph_fourier_transform(self.magnitudes, n_eigs=10)
            except:
                pass
        
        if self.eigenvals is not None and len(self.eigenvals) >= 2:
            stats['spectral_gap'] = float(self.eigenvals[1] - self.eigenvals[0])
        else:
            stats['spectral_gap'] = np.nan
        
        return stats


# High-level convenience function
def compute_seismic_graph_stats(
    coordinates: np.ndarray,
    magnitudes: Optional[np.ndarray] = None,
    k: int = 10
) -> Dict[str, Any]:
    """
    Compute all graph statistics in one call.
    
    Parameters:
        coordinates: Event coordinates (N, 2) or (N, 3)
        magnitudes: Event magnitudes (N,), optional
        k: Number of nearest neighbors
    
    Returns:
        Dictionary with all graph statistics
    """
    graph = SeismicGraphTGS(coordinates, magnitudes=magnitudes, k=k)
    graph.build_knn_graph()
    graph.detect_communities(method='leiden')
    
    stats = graph.summary_statistics()
    
    return stats
