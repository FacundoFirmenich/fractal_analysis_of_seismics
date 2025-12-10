# Seismic Fractal Analysis
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
Python framework for fractal dimension analysis of earthquake spatial distributions.
## Features
- **Grassberger-Procaccia correlation dimension** estimation with Theil-Sen robust regression
- **Rényi multifractal spectrum** analysis (D₀, D₁, D₂)
- **Topological graph structure** detection via Leiden community clustering
- **Bootstrap uncertainty quantification** with user-configurable iterations
- **Bayesian methods** for observational bias correction
- **Three-tier deployment**: web app, Colab notebook, Python package
## Installation
```bash
git clone [https://github.com/FacundoFirmenich/fractal_analysis_of_seismics.git](https://github.com/FacundoFirmenich/fractal_analysis_of_seismics.git)
cd fractal_analysis_of_seismics
pip install -r requirements.txt
Optional Performance Accelerators
bash
pip install numba  # 3-20× speedup for numerical operations
Quick Start
python
from sfa.core import FractalDimensionEstimator
from sfa.data import fetch_usgs_catalog
from sfa.utils import geographic_to_metric
# Fetch earthquake data from USGS
catalog = fetch_usgs_catalog(
    lat=37.0, 
    lon=-122.0, 
    radius_km=200,
    start_date="2010-01-01"
)
# Convert to metric coordinates
coords = geographic_to_metric(
    catalog['latitude'],
    catalog['longitude'],
    catalog['depth']
)
# Estimate fractal dimension
estimator = FractalDimensionEstimator()
d2, sem = estimator.compute_gp_dimension(
    coords,
    bootstrap_iterations=200
)
print(f"Correlation dimension: D₂ = {d2:.3f} ± {sem:.3f}")
Web Interface
Launch the interactive Streamlit dashboard:

bash
streamlit run streamlit_app.py
Google Colab
Open the complete analysis notebook:

Documentation
API Reference
 - Complete module documentation
Mathematical Appendix
 - Coordinate transformations and algorithms
Future Enhancements
 - Roadmap and planned features
Testing
Run the test suite:

bash
pytest tests/ -v
Test coverage:

Unit tests: Core module functionality
Integration tests: End-to-end workflows
Scientific validation: Synthetic geometries (1D, 2D, 3D)
Data Source
All seismic data sourced from: United States Geological Survey (USGS) ComCat Earthquake Catalog

Citation
If you use this software in your research, please cite:

bibtex
@software{seismic_fractal_analysis,
  author = {Firmenich, Facundo and Firmenich, Pau and Firmenich, León},
  title = {Seismic Fractal Analysis: Multi-Planar Hierarchy Framework},
  year = {2025},
  version = {1.0.0},
  url = {[https://github.com/FacundoFirmenich/fractal_analysis_of_seismics](https://github.com/FacundoFirmenich/fractal_analysis_of_seismics)}
}
License
GNU General Public License v3.0 (GPLv3) - see 
LICENSE

Authors
Facundo Firmenich - Lead Developer - ORCID: 0009-0002-6578-3811
Pau Firmenich
León Firmenich
Institution: Centro de Estudios del Sur (CEDESUR), Argentina / Universitat de Barcelona, Spain

Acknowledgments
USGS for maintaining the FDSN web service and providing open seismic data
Open-source scientific Python community (NumPy, SciPy, Pandas, Matplotlib, NetworkX)
Contact
For questions or collaboration: 
f.firmenich@cedesur.org

