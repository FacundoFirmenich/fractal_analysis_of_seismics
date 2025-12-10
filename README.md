# Pan-American Seismic Fractal Analysis - Documentation

**Version**: 1.0.0  
**Last Updated**: 2025-12-08

---

## 📋 Overview

This framework provides a comprehensive Bayesian fractal analysis toolkit for seismicity characterization across diverse tectonic settings. It implements:

- **Grassberger-Procaccia correlation dimension (D₂)** with Ripley edge corrections
- **Rényi spectrum (D₀, D₁, D₂)** for hierarchical organization assessment
- **Topological Graph Structure (TGS)** analysis via k-NN networks
- **Bayesian D₃ transformation** for intrinsic dimension estimation
- **Adaptive parameter selection** via Thompson Sampling reinforcement learning
- **Uncertainty quantification** through bootstrap resampling (n=200)

**Key Features**:
- ✅ Automated USGS data acquisition
- ✅ Reproducible Bayesian inference
- ✅ Multi-scale organizational metrics
- ✅ Publication-ready visualizations
- ✅ 3-tier deployment (Streamlit/Colab/PyPI)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[USERNAME]/PanAmericanPaper.git
cd PanAmericanPaper

# Install dependencies
pip install -r requirements.txt

# Optional: performance accelerators
pip install numba  # 3-20× speedup
```

### Basic Usage

```python
from sfa.core import FractalDimensionEstimator
from sfa.data import SeismicDataAcquisition

# Fetch data
data_acq = SeismicDataAcquisition()
events = data_acq.fetch_usgs_catalog(
    bounds=(32, 38, -122, -115),  # San Andreas
    depth_range=(0, 30),
    min_magnitude=2.4,
    start_date="2010-01-01",
    end_date="2025-11-22"
)

# Normalize coordinates
coords = data_acq.normalize_coordinates(
    events['latitude'].values,
    events['longitude'].values,
    events['depth'].values
)

# Estimate D₂
estimator = FractalDimensionEstimator()
d2, d2_sem = estimator.compute_dimension(
    coords,
    method='gp',  # Grassberger-Procaccia
    bootstrap_iterations=200
)

print(f"D₂ = {d2:.3f} ± {d2_sem:.3f}")
```

**Output**: `D₂ = 2.072 ± 0.001`

---

## 📚 Documentation Structure

- **[API Reference](API.md)**: Complete module/class/function documentation
- **[Tutorial](TUTORIAL.md)**: Step-by-step walkthrough
- **[Methods](METHODS.md)**: Mathematical foundations
- **[Examples](EXAMPLES.md)**: Case studies (Pan-American analysis)

---

## 🎯 Use Cases

### 1. Regional Tectonic Comparison
```python
from scripts.run_pan_american_7_regions import main
results_df = main()  # Analyzes 7 Pan-American regions
```

### 2. Hierarchical Organization
```python
from sfa.multifractal import MultifractalAnalyzer
analyzer = MultifractalAnalyzer()
d0, d1, d2 = analyzer.compute_renyi_spectrum(coords)
H = d1 - d0  # Hierarchical Index
```

### 3. Topological Structure
```python
from sfa.graph_tgs import SeismicGraphTGS
tgs = SeismicGraphTGS()
communities, d_graph = tgs.analyze(coords)
```

---

## 🔬 Scientific Validation

Framework validated via:
- ✅ Synthetic 1D/2D/3D geometries (error <6%)
- ✅ Temporal stability (ΔD₂ < 0.05 over 15 years)
- ✅ Declustering sensitivity (Gardner-Knopoff method)
- ✅ Literature cross-validation (Kagan 2007, Hirata 1989)

**Key Results**:
- Pan-American D₂ range: **2.07-2.57** (multi-planar intermediate)
- Rényi hierarchy detected in 4/7 regions (Caribbean H=+0.182 MAX)
- Intrinsic volumetric structure (D₃=3.00) in 5/6 regions

---

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{panamfractal2025,
  title = {Pan-American Seismic Fractal Analysis Framework},
  author = {[Authors]},
  year = {2025},
  url = {https://github.com/[USERNAME]/PanAmericanPaper},
  version = {1.0.0}
}
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 📄 License

GNU General Public License v3.0 (GPLv3) - see [LICENSE](../LICENSE)

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/[USERNAME]/PanAmericanPaper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/[USERNAME]/PanAmericanPaper/discussions)
- **Email**: [contact email]

---

## 🌟 Acknowledgments

- USGS for maintaining the ComCat earthquake catalog
- Open-source scientific Python community (NumPy, SciPy, Pandas, Matplotlib)
- [Funding sources]
