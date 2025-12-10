# API Reference - Pan-American Fractal Analysis

**Version**: 1.0.0

---

## Core Modules

### `sfa.core` - Fractal Dimension Estimation

#### Class: `FractalDimensionEstimator`

Main class for computing correlation dimension D₂.

**Methods**:

```python
compute_dimension(
    coords: np.ndarray,
    method: str = 'gp',
    bootstrap_iterations: int = 200,
    **kwargs
) -> Tuple[float, float]
```

**Parameters**:
- `coords`: (N, 3) array of normalized coordinates [0,1]³
- `method`: `'gp'` (Grassberger-Procaccia) or `'takens'`
- `bootstrap_iterations`: Number of bootstrap samples for uncertainty

**Returns**:
- `d2`: Estimated correlation dimension
- `d2_sem`: Standard error of mean

**Example**:
```python
estimator = FractalDimensionEstimator()
d2, sem = estimator.compute_dimension(coords, method='gp', bootstrap_iterations=200)
```

---

```python
get_diagnostics() -> Dict[str, np.ndarray]
```

Returns correlation integral diagnostics.

**Returns**:
- Dictionary with keys: `'r_values'`, `'C_r'`, `'slope'`, `'r_min'`, `'r_max'`

---

### `sfa.data` - Data Acquisition

#### Class: `SeismicDataAcquisition`

Handles USGS catalog fetching and preprocessing.

**Methods**:

```python
fetch_usgs_catalog(
    bounds: Tuple[float, float, float, float],
    depth_range: Tuple[float, float],
    min_magnitude: float,
    start_date: str,
    end_date: str
) -> pd.DataFrame
```

**Parameters**:
- `bounds`: (lat_min, lat_max, lon_min, lon_max) in degrees
- `depth_range`: (depth_min, depth_max) in km
- `min_magnitude`: Minimum magnitude threshold
- `start_date`: ISO format 'YYYY-MM-DD'
- `end_date`: ISO format 'YYYY-MM-DD'

**Returns**:
- DataFrame with columns: `['latitude', 'longitude', 'depth', 'magnitude', 'time']`

---

```python
normalize_coordinates(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    depths: np.ndarray
) -> np.ndarray
```

Transforms (lat, lon, depth) → metric space → unit cube [0,1]³.

**Returns**:
- (N, 3) array of normalized coordinates

---

### `sfa.multifractal` - Rényi Spectrum

#### Class: `MultifractalAnalyzer`

Computes generalized Rényi dimensions.

**Methods**:

```python
compute_renyi_spectrum(
    coords: np.ndarray,
    q_values: List[float] = [0, 1, 2]
) -> Tuple[float, float, float]
```

**Parameters**:
- `coords`: (N, 3) normalized coordinates
- `q_values`: Rényi orders (default: [0, 1, 2])

**Returns**:
- `d0`: Capacity dimension
- `d1`: Information dimension
- `d2`: Correlation dimension

**Example**:
```python
analyzer = MultifractalAnalyzer()
d0, d1, d2 = analyzer.compute_renyi_spectrum(coords)
H = d1 - d0  # Hierarchical Index
```

---

### `sfa.graph_tgs` - Topological Graph Structure

#### Class: `SeismicGraphTGS`

k-NN graph analysis with community detection.

**Methods**:

```python
analyze(
    coords: np.ndarray,
    k: int = 10
) -> Tuple[int, float, float]
```

**Parameters**:
- `coords`: (N, 3) normalized coordinates
- `k`: Number of nearest neighbors

**Returns**:
- `n_communities`: Number of detected communities
- `d_graph`: Graph fractal dimension
- `spectral_gap`: Laplacian eigenvalue gap

**Example**:
```python
tgs = SeismicGraphTGS()
n_comm, d_graph, gap = tgs.analyze(coords, k=10)
```

---

### `sfa.analogies` - Scale Transformations

#### Function: `scale_transformation_operator`

Bayesian D₂ → D₃ transformation.

```python
scale_transformation_operator(
    d2_observed: float,
    d2_uncertainty: float
) -> Tuple[float, float]
```

**Parameters**:
- `d2_observed`: Measured correlation dimension
- `d2_uncertainty`: Standard error of D₂

**Returns**:
- `d3_estimate`: Estimated intrinsic 3D dimension
- `d3_std`: Propagated uncertainty

**Example**:
```python
d3, d3_std = scale_transformation_operator(2.25, 0.01)
```

---

## Utilities

### `sfa.optimization` - Performance

#### `compute_knn_sparse_morans_i`

Sparse k-NN Moran's I for spatial autocorrelation.

```python
compute_knn_sparse_morans_i(
    coords: np.ndarray,
    values: np.ndarray,
    k: int = 10
) -> Tuple[float, float]
```

**Returns**:
- `I`: Moran's I statistic
- `p_value`: Statistical significance

---

## Configuration

### Environment Variables

- `SFA_BACKEND`: Performance backend (`'numba'`, `'numpy'`)
- `SFA_CACHE_DIR`: Cache directory for downloaded catalogs

### Performance Tuning

```python
from sfa.accelerate import print_performance_info

print_performance_info()  # Displays active backend and speedups
```

---

## Error Handling

**Common Exceptions**:

- `ValueError`: Invalid coordinate dimensions or empty arrays
- `RuntimeError`: USGS API timeout or connection failure
- `ConvergenceError`: Bootstrap did not converge

**Handling**:
```python
try:
    d2, sem = estimator.compute_dimension(coords)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Computation failed: {e}")
```

---

## Version Compatibility

- **Python**: 3.9–3.14
- **NumPy**: ≥1.21
- **SciPy**: ≥1.7
- **Pandas**: ≥1.3

**Optional accelerators**:
- `numba` (Python 3.9–3.12): 3-20× speedup
- `cython`: 10-100× speedup (requires compilation)

---

## See Also

- [Tutorial](TUTORIAL.md): Step-by-step walkthrough
- [Methods](METHODS.md): Mathematical foundations
- [Examples](EXAMPLES.md): Case studies
