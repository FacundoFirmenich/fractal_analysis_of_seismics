#!/usr/bin/env python3
"""
Pan-American Seismic Fractal Analysis - Streamlit App
=====================================================
Real-time fractal dimension computation with interactive visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Pan-American Fractal Analysis",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Pan-American Seismic Fractal Analysis")
st.markdown("**Real-time D₂ computation from USGS catalog**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Region selection
region_presets = {
    "San Andreas Fault": {
        "bounds": (32.0, 38.0, -122.0, -115.0),
        "depth": (0, 30),
        "description": "Transform fault, California"
    },
    "Cascadia Subduction": {
        "bounds": (40.0, 50.0, -130.0, -122.0),
        "depth": (0, 100),
        "description": "Warm subduction, Pacific Northwest"
    },
    "Cocos Plate": {
        "bounds": (10.0, 20.0, -105.0, -90.0),
        "depth": (0, 100),
        "description": "Fast subduction, Central America"
    },
    "Custom Region": {
        "bounds": (0, 0, 0, 0),
        "depth": (0, 100),
        "description": "Define your own region"
    }
}

region_name = st.sidebar.selectbox(
    "Select Region",
    list(region_presets.keys())
)

st.sidebar.markdown(f"*{region_presets[region_name]['description']}*")

# Custom bounds if selected
if region_name == "Custom Region":
    st.sidebar.subheader("Custom Bounds")
    lat_min = st.sidebar.number_input("Min Latitude", value=32.0)
    lat_max = st.sidebar.number_input("Max Latitude", value=38.0)
    lon_min = st.sidebar.number_input("Min Longitude", value=-122.0)
    lon_max = st.sidebar.number_input("Max Longitude", value=-115.0)
    bounds = (lat_min, lat_max, lon_min, lon_max)
    depth_min = st.sidebar.number_input("Min Depth (km)", value=0)
    depth_max = st.sidebar.number_input("Max Depth (km)", value=30)
    depth_range = (depth_min, depth_max)
else:
    bounds = region_presets[region_name]['bounds']
    depth_range = region_presets[region_name]['depth']

# Time range
st.sidebar.subheader("Time Range")
end_date = datetime.now()
start_date_default = end_date - timedelta(days=365*15)  # 15 years

start_date = st.sidebar.date_input(
    "Start Date",
    value=start_date_default
)
end_date_input = st.sidebar.date_input(
    "End Date",
    value=end_date
)

# Magnitude threshold
min_mag = st.sidebar.slider(
    "Minimum Magnitude",
    min_value=2.0,
    max_value=5.0,
    value=2.4,
    step=0.1
)

# Bootstrap iterations
n_bootstrap = st.sidebar.selectbox(
    "Bootstrap Iterations",
    [50, 100, 200],
    index=2
)

# Compute button
compute_button = st.sidebar.button("🚀 Compute D₂", type="primary")

# Main area tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Correlation Integral",
    "🔬 Rényi Spectrum",
    "🕸️ TGS Communities",
    "📋 Results Table"
])

# Placeholder for results
if 'results' not in st.session_state:
    st.session_state.results = None

# Computation logic
if compute_button:
    with st.spinner(f"Fetching earthquake data from USGS..."):
        try:
            # Import framework
            import sys
            sys.path.insert(0, '.')
            from sfa.data import SeismicDataAcquisition
            from sfa.core import FractalDimensionEstimator
            from sfa.multifractal import MultifractalAnalyzer
            from sfa.graph_tgs import SeismicGraphTGS
            
            # Fetch data
            data_acq = SeismicDataAcquisition()
            events = data_acq.fetch_usgs_catalog(
                bounds=bounds,
                depth_range=depth_range,
                min_magnitude=min_mag,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date_input.strftime("%Y-%m-%d")
            )
            
            st.success(f"✅ Retrieved {len(events)} earthquakes")
            
            if len(events) < 100:
                st.warning("⚠️ Less than 100 events - results may be unreliable")
            
            # Normalize coordinates
            coords = data_acq.normalize_coordinates(
                events['latitude'].values,
                events['longitude'].values,
                events['depth'].values
            )
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Compute D₂
            status_text.text("Computing D₂ (Grassberger-Procaccia)...")
            progress_bar.progress(25)
            
            estimator = FractalDimensionEstimator()
            d2, d2_sem = estimator.compute_dimension(
                coords,
                method='gp',
                bootstrap_iterations=n_bootstrap
            )
            
            # Compute Rényi
            status_text.text("Computing Rényi spectrum...")
            progress_bar.progress(50)
            
            mf_analyzer = MultifractalAnalyzer()
            d0, d1, _ = mf_analyzer.compute_renyi_spectrum(coords)
            H = d1 - d0
            
            # Compute TGS
            status_text.text("Analyzing topological structure...")
            progress_bar.progress(75)
            
            tgs = SeismicGraphTGS()
            n_communities, d_graph, _ = tgs.analyze(coords, k=10)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results
            st.session_state.results = {
                'events': events,
                'coords': coords,
                'd2': d2,
                'd2_sem': d2_sem,
                'd0': d0,
                'd1': d1,
                'H': H,
                'n_communities': n_communities,
                'd_graph': d_graph,
                'diagnostics': estimator.get_diagnostics()
            }
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

# Display results if available
if st.session_state.results:
    res = st.session_state.results
    
    # Tab 1: Correlation integral
    with tab1:
        st.subheader("Correlation Integral")
        
        diag = res['diagnostics']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(diag['r_values'], diag['C_r'], 'o-',
                  linewidth=2.5, markersize=7, color='#3498DB',
                  label='Correlation integral C(r)')
        ax.set_xlabel('Distance r (normalized)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Correlation integral C(r)', fontsize=14, fontweight='bold')
        ax.set_title(f"{region_name}: D₂ = {res['d2']:.3f} ± {res['d2_sem']:.3f}",
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.25, linewidth=1.2)
        ax.legend(fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        if res['d2'] < 2.0:
            st.info("📐 **Planar organization** - seismicity confined to fault planes")
        elif res['d2'] > 2.5:
            st.info("📦 **Volumetric organization** - distributed deformation")
        else:
            st.info("📊 **Multi-planar intermediate** - hierarchical organization")
    
    # Tab 2: Rényi spectrum
    with tab2:
        st.subheader("Rényi Spectrum")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            q = [0, 1, 2]
            dq = [res['d0'], res['d1'], res['d2']]
            ax.plot(q, dq, 'o-', linewidth=3, markersize=12,
                    color='#E74C3C', markeredgecolor='white', markeredgewidth=2)
            ax.set_xlabel('Rényi order q', fontsize=14, fontweight='bold')
            ax.set_ylabel('Dimension Dq', fontsize=14, fontweight='bold')
            ax.set_title('Multi-Scale Fractal Dimensions', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.25)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['D₀\\n(Capacity)', 'D₁\\n(Information)', 'D₂\\n(Correlation)'])
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.metric("D₀ (Capacity)", f"{res['d0']:.3f}")
            st.metric("D₁ (Information)", f"{res['d1']:.3f}")
            st.metric("D₂ (Correlation)", f"{res['d2']:.3f} ± {res['d2_sem']:.3f}")
            st.metric("Hierarchical Index H", f"{res['H']:.3f}",
                      delta="Multi-planar" if res['H'] > 0 else "Homogeneous")
    
    # Tab 3: TGS
    with tab3:
        st.subheader("Topological Graph Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Communities Detected", res['n_communities'])
            st.metric("Graph Dimension D_graph", f"{res['d_graph']:.3f}")
            st.metric("Δ(D₂ - D_graph)", f"{res['d2'] - res['d_graph']:.3f}")
        
        with col2:
            st.markdown("**Interpretation**:")
            if res['d_graph'] < res['d2']:
                st.success("✅ Topological < Euclidean → Hierarchical organization confirmed")
            else:
                st.info("ℹ️ Topological ≈ Euclidean → Homogeneous structure")
    
    # Tab 4: Results table
    with tab4:
        st.subheader("Complete Results")
        
        results_df = pd.DataFrame({
            'Metric': [
                'Region',
                'N Events',
                'Date Range',
                'Magnitude Range',
                'Depth Range (km)',
                'D₂ (Correlation)',
                'D₀ (Capacity)',
                'D₁ (Information)',
                'Hierarchical Index H',
                'TGS Communities',
                'Graph Dimension D_graph'
            ],
            'Value': [
                region_name,
                len(res['events']),
                f"{start_date} to {end_date_input}",
                f"{res['events']['magnitude'].min():.1f} - {res['events']['magnitude'].max():.1f}",
                f"{res['events']['depth'].min():.1f} - {res['events']['depth'].max():.1f}",
                f"{res['d2']:.3f} ± {res['d2_sem']:.3f}",
                f"{res['d0']:.3f}",
                f"{res['d1']:.3f}",
                f"{res['H']:.3f}",
                res['n_communities'],
                f"{res['d_graph']:.3f}"
            ]
        })
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            f"fractal_results_{region_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

else:
    st.info("👈 Configure parameters in sidebar and click **Compute D₂** to start analysis")

# Footer
st.markdown("---")
st.markdown("""
**Data Source**: United States Geological Survey (USGS)  
**Framework**: Pan-American Seismic Fractal Analysis  
**License**: GNU GPLv3
""")
