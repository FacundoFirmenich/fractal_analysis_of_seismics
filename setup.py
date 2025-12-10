"""
Seismic Fractal Analysis Package
=================================
Multi-planar hierarchical analysis of Pan-American seismicity via Bayesian 
fractal framework. Correlation dimension estimation with comprehensive
statistical validation, Rényi spectrum, topological graph structure, and
Bayesian D₃ transformation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seismic-fractal-analysis",
    version="1.0.0",
    author="Facundo Firmenich, Pau Firmenich, León Firmenich",
    author_email="f.firmenich@cedesur.org",
    description="Python framework for fractal dimension analysis of earthquake spatial distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FacundoFirmenich/fractal_analysis_of_seismics",
    project_urls={
        "Bug Tracker": "https://github.com/FacundoFirmenich/fractal_analysis_of_seismics/issues",
        "Documentation": "https://github.com/FacundoFirmenich/fractal_analysis_of_seismics/tree/main/docs",
        "Source Code": "https://github.com/FacundoFirmenich/fractal_analysis_of_seismics",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "kaleido>=0.2.1",
        "networkx>=2.6.0",
        "igraph>=0.11.0",
        "leidenalg>=0.10.0",
        "scikit-learn>=1.0.0",
        "requests>=2.26.0",
        "joblib>=1.1.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "performance": ["numba>=0.56.0", "cython>=0.29.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0", "mypy>=0.950"],
        "docs": ["sphinx>=4.5.0", "sphinx-rtd-theme>=1.0.0", "nbsphinx>=0.8.0"],
    },
    entry_points={
        "console_scripts": [
            "sfa-analyze=sfa.cli:main",
            "sfa-visualize=sfa.vis:main",
        ],
    },
)
