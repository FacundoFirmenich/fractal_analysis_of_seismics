"""
Pan-American Transect Analysis - 7 Regions (CORE++ COMPLETE)
==============================================================
Full integration of SFA Core++ capabilities:
- Pagination USGS (>20k events)
- Ripley edge corrections (accelerated)
- Parallel bootstrap optimization
- Multifractal analysis (Rényi spectrum D_q)
- Spatial statistics (Moran's I, Clark-Evans 3D)
- Bayesian robustness methods

Regions:
1. San Andreas Fault (Transform)
2. Cascadia Subduction (Interface)
3. Cocos Plate (Mesoamerica)
4. Caribbean Plate (Lesser Antilles)
5. Andes North (Colombia)
6. Andes Central (Peru-Chile)
7. Andes South (Chile-Argentina)

Output:
- pan_american_results_YYYYMMDD_HHMMSS.csv (expanded with multifractal)
- Individual figures per region
- Summary comparison figure
"""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sfa.data import SeismicDataAcquisition, PanAmericanPresets
from sfa.core import FractalDimensionEstimator, SyntheticValidator
from sfa.vis import FractalPlotter
from sfa.multifractal import MultifractalAnalyzer
from sfa.stats import BayesianRobustness, SeismicityAnalysis, SpatialStatisticalAnalysis
from sfa import accelerate
from sfa.graph_tgs import compute_seismic_graph_stats
from sfa.analogies import scale_transformation_operator


def run_synthetic_validation():
    """
    Run synthetic validation tests (1D line, 2D plane, 3D cube).
    Tests corrected to report theoretical values and only fail on critical (1D/2D).
    
    Returns:
        Tuple of (results_dataframe, all_passed)
    """
    print("\n" + "=" * 80)
    print("SYNTHETIC VALIDATION TESTS (CORRECTED)")
    print("=" * 80)
    
    estimator = FractalDimensionEstimator()
    validator = SyntheticValidator(estimator)
    
    results_df, all_passed = validator.run(verbose=True)
    
    if not all_passed:
        print("\n⚠️  WARNING: Critical synthetic tests failed (1D or 2D)!")
        print("    Review algorithm parameters before interpreting results.")
    
    return results_df, all_passed


def run_pan_american_analysis():
    """
    Execute Pan-American 7-region fractal analysis with COMPLETE core++ integration.
    
    Integrates:
    - USGS pagination (>20k events)
    - Parallel bootstrap
    - Accelerated Ripley corrections
    - Multifractal D_q spectrum
    - Spatial autocorrelation (Moran's I)
    - Clustering analysis (Clark-Evans 3D)
    - Seismicity statistics (b-value, Mc)
    """
    print("=" * 80)
    print("PAN-AMERICAN TRANSECT ANALYSIS - 7 REGIONS (CORE++ COMPLETE)")
    print("=" * 80)
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Performance backend info
    print("\n" + "=" * 80)
    print("PERFORMANCE BACKEND")
    print("=" * 80)
    print(f"Backend: {accelerate.get_backend()}")
    print()

    # Generate timestamp for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # RUN SYNTHETIC VALIDATION FIRST
    synthetic_results, synthetic_passed = run_synthetic_validation()
    if not synthetic_passed:
        print("\n⚠️  Continuing with real data analysis despite validation warnings...\n")
    
    # Initialize modules
    data_acq = SeismicDataAcquisition()
    fractal_est = FractalDimensionEstimator()
    seismicity_analysis = SeismicityAnalysis()

    # Create output directories
    os.makedirs("fractal_analysis_output", exist_ok=True)
    os.makedirs("backups", exist_ok=True)

    # Define the 7 Pan-American regions
    PAN_AMERICAN_7 = {
        "San Andreas Fault": PanAmericanPresets.SAN_ANDREAS,
        "Cascadia Subduction": PanAmericanPresets.CASCADIA,
        "Cocos Plate (Mesoamerica)": PanAmericanPresets.COCOS_PLATE,
        "Caribbean Plate (Lesser Antilles)": PanAmericanPresets.CARIBBEAN,
        "Andes North (Colombia)": PanAmericanPresets.ANDES_NORTH,
        "Andes Central (Peru-Chile)": PanAmericanPresets.ANDES_CENTRAL,
        "Andes South (Chile-Argentina)": PanAmericanPresets.ANDES_SOUTH,
    }

    results_summary = []

    for i, (region_name, bounds) in enumerate(PAN_AMERICAN_7.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"Region {i}/7: {region_name}")
        print(f"{'=' * 80}")
        print(f"  Bounds: {bounds}")

        # 1. Fetch Data (with pagination)
        # USER PREFERENCE: min_mag = 2.4 (optimal richness/quality balance)
        # Slightly more restrictive than 2.5, but preserves shallow seismicity
        min_mag = 2.4
        
        try:
            data = data_acq.retrieve_catalog(
                region_name, bounds, min_magnitude=min_mag, 
                start_year=2010, end_date="2025-11-22"
            )
        except Exception as e:
            print(f"  ❌ ERROR fetching data: {e}")
            continue

        if not data or data.get('event_count', 0) == 0:
            print(f"  ⚠️ WARNING: No data found for {region_name}")
            continue

        print(f"  ✅ Loaded {data['event_count']} events")

        coords_norm = data["coordinates_normalized"]
        coords_metric = data["coordinates_metric"]
        catalog = data["catalog"]

        # 2. Fractal Dimension (GP with Ripley corrections)
        print(f"  Computing D₂ (Grassberger-Procaccia + Ripley)...")
        try:
            d2_gp, sem_gp, diagnostics = fractal_est.compute_gp_dimension(
                coords_norm, bootstrap_iterations=200, return_diagnostics=True
            )
            print(f"  ✅ D₂ (GP): {d2_gp:.3f} ± {sem_gp:.3f}")
        except Exception as e:
            print(f"  ❌ ERROR computing GP: {e}")
            d2_gp, sem_gp = np.nan, np.nan
            diagnostics = None
        
        # 3. MULTIFRACTAL ANALYSIS (Rényi Spectrum D_q)
        print(f"  Computing Rényi spectrum D_q...")
        try:
            q_range = np.linspace(-5, 5, 21)
            analyzer = MultifractalAnalyzer()
            q_vals, D_q_vals = analyzer.compute_renyi_spectrum(coords_norm, q_values=q_range)
            
            # Extract metrics
            multifractal_width = D_q_vals.max() - D_q_vals.min()
            D_0 = D_q_vals[10] if len(D_q_vals) > 10 else np.nan  # q=0 capacity
            D_1 = D_q_vals[11] if len(D_q_vals) > 11 else np.nan  # q=1 information
            
            print(f"  ✅ Multifractal: Δα={multifractal_width:.3f}, D₀={D_0:.3f}, D₁={D_1:.3f}")
        except Exception as e:
            print(f"  ⚠️ Multifractal failed: {e}")
            multifractal_width, D_0, D_1 = np.nan, np.nan, np.nan
            D_q_spectrum = np.array([])
        
        # 4. SPATIAL AUTOCORRELATION (Moran's I)
        print(f"  Computing Moran's I (spatial autocorrelation)...")
        try:
            morans_i, morans_p = BayesianRobustness.morans_i_depth(
                coords_metric, k=10
            )
            print(f"  ✅ Moran's I: {morans_i:.3f} (p={morans_p:.4f})")
        except Exception as e:
            print(f"  ⚠️ Moran's I failed: {e}")
            morans_i, morans_p = np.nan, np.nan
        
        # 5. CLUSTERING ANALYSIS (Clark-Evans 3D)
        print(f"  Computing Clark-Evans 3D index...")
        try:
            ce_index = SpatialStatisticalAnalysis.clark_evans_3d(coords_norm)
            if ce_index < 1.0:
                ce_type = 'clustered'
            elif ce_index > 1.0:
                ce_type = 'dispersed'
            else:
                ce_type = 'random'
            print(f"  ✅ Clark-Evans: R={ce_index:.3f} ({ce_type})")
        except Exception as e:
            print(f"  ⚠️ Clark-Evans failed: {e}")
            ce_index, ce_type = np.nan, 'unknown'
        
        # 6. SEISMICITY STATISTICS (b-value, Mc)
        print(f"  Computing b-value and Mc...")
        try:
            b_value, b_std, mc = seismicity_analysis.compute_b_value(
                catalog['mag'].values
            )
            print(f"  ✅ Mc={mc:.2f}, b-value={b_value:.3f} ± {b_std:.3f}")
        except Exception as e:
            print(f"  ⚠️ Seismicity stats failed: {e}")
            mc, b_value, b_std = np.nan, np.nan, np.nan
        
        # 7. GRAPH TGS ANALYSIS (communities, D_Graph, spectral gap)
        print(f"  Computing graph TGS analysis...")
        try:
            graph_stats = compute_seismic_graph_stats(
                coords_metric,
                magnitudes=catalog['mag'].values if 'mag' in catalog else None,
                k=10
            )
            n_communities = graph_stats.get('n_communities', 0)
            D_graph = graph_stats.get('D_graph', np.nan)
            spectral_gap = graph_stats.get('spectral_gap', np.nan)
            print(f"  ✅ TGS: {n_communities} communities, D_Graph={D_graph:.3f}, gap={spectral_gap:.4f}")
        except Exception as e:
            print(f"  ⚠️ TGS failed: {e}")
            n_communities, D_graph, spectral_gap = 0, np.nan, np.nan
        
        # 8. D₃ TRANSFORMATION (Bayesian inference D₂ → D₃ underlying)
        print(f"  Computing D₃ transformation (Bayesian)...")
        try:
            if not np.isnan(d2_gp):
                D3_est, D3_uncertainty = scale_transformation_operator(
                    d2_gp,
                    noise_model='multiplicative',
                    use_bayesian=False  # Analytical approximation (faster)
                )
                D3_std = D3_uncertainty.get('std', np.nan) if isinstance(D3_uncertainty, dict) else np.nan
                print(f"  ✅ D₃ estimated: {D3_est:.3f} ± {D3_std:.3f}")
            else:
                D3_est, D3_std = np.nan, np.nan
        except Exception as e:
            print(f"  ⚠️ D₃ transformation failed: {e}")
            D3_est, D3_std = np.nan, np.nan

        # 9. STORE RESULTS (COMPLETE INTEGRATION)
        results_summary.append({
            "region": region_name,
            "n_events": data['event_count'],
            "d2_gp": d2_gp,
            "d2_sem": sem_gp,
            "multifractal_width": multifractal_width,
            "D_0_capacity": D_0,
            "D_1_information": D_1,
            "morans_i": morans_i,
            "morans_p": morans_p,
            "clark_evans_R": ce_index,
            "clark_evans_type": ce_type,
            "mc": mc,
            "b_value": b_value,
            "b_std": b_std,
            "n_communities": n_communities,
            "D_graph": D_graph,
            "spectral_gap": spectral_gap,
            "D3_estimated": D3_est,
            "D3_std": D3_std,
            "spatial_bounds": str(bounds),
        })

        # 10. VISUALIZATIONS
        if diagnostics and diagnostics["sample_curves"]:
            try:
                print(f"  Generating plots...")
                slope, (log_r, log_c, mask) = diagnostics["sample_curves"][0]
                
                fig_corr = FractalPlotter.plot_correlation_integral(
                    log_r, log_c, slope, mask, region_name
                )
                fig_corr.savefig(
                    f"fractal_analysis_output/{region_name.replace(' ', '_')}_correlation_{timestamp}.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig_corr)
                
                print(f"  ✅ Plots saved")
            except Exception as e:
                print(f"  ⚠️ Plot generation failed: {e}")

    # 9. SAVE RESULTS CSV (EXPANDED COLUMNS)
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")
    
    df_results = pd.DataFrame(results_summary)
    csv_path = f"fractal_analysis_output/pan_american_results_{timestamp}.csv"
    
    # Add metadata header
    with open(csv_path, 'w') as f:
        f.write(f"# Pan-American Transect Analysis - 7 Regions (CORE++ COMPLETE)\n")
        f.write(f"# Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Backend: {accelerate.get_backend()}\n")
        f.write(f"# Bootstrap iterations: 200\n")
        f.write(f"# Multifractal q-range: -5 to 5 (21 values)\n")
        f.write(f"#\n")
        df_results.to_csv(f, index=False)
    
    print(f"✅ Results saved: {csv_path}")
    print(f"   Columns: {len(df_results.columns)} (expanded with multifractal + spatial stats)")
    print(f"   Regions: {len(df_results)}")
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Execution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df_results


if __name__ == "__main__":
    results = run_pan_american_analysis()
    print(f"\n✅ All done! Results dataframe shape: {results.shape}")
