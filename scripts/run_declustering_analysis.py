#!/usr/bin/env python3
"""
Gardner-Knopoff Declustering Analysis
======================================
Paper Nov 2025 Table 3: Declustering Sensitivity Analysis

Critical finding: Caribbean Plate shows DRAMATIC shift:
  Raw D₂ = 1.58 → Declustered D₂ = 2.07 (+0.49)
  
Interpretation: Dense aftershock clusters (planar) masked volumetric 
background tectonic deformation.

This script applies space-time window declustering to all 7 regions
and recomputes D₂ for declustered catalogs.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfa.data import SeismicDataAcquisition, PanAmericanPresets
from sfa.core import FractalDimensionEstimator

def gardner_knopoff_windows(magnitude):
    """
    Gardner-Knopoff (1974) space-time windows for aftershock removal.
    
    Args:
        magnitude: Mainshock magnitude
        
    Returns:
        tuple: (time_window_days, space_window_km)
    """
    # Original Gardner-Knopoff parameters
    # Time window T(M) = 10^(0.032*M + 2.7385) days
    # Space window R(M) = 10^(0.1238*M + 0.983) km
    
    time_days = 10**(0.032 * magnitude + 2.7385)
    space_km = 10**(0.1238 * magnitude + 0.983)
    
    return time_days, space_km

def decluster_catalog(catalog, coords_metric):
    """
    Apply Gardner-Knopoff declustering to earthquake catalog.
    
    Args:
        catalog: DataFrame with columns ['time', 'mag']
        coords_metric: Nx3 array of (x,y,z) in km
        
    Returns:
        tuple: (declustered_indices, n_removed)
    """
    n_events = len(catalog)
    is_mainshock = np.ones(n_events, dtype=bool)  # All initially mainshocks
    
    # Sort by magnitude (descending) - process largest events first
    mag_order = np.argsort(catalog['mag'].values)[::-1]
    
    print(f"  Processing {n_events} events by magnitude...")
    
    for idx in mag_order:
        if not is_mainshock[idx]:
            continue  # Already classified as aftershock
            
        mag = catalog.iloc[idx]['mag']
        time = catalog.iloc[idx]['time']
        coords = coords_metric[idx]
        
        # Get Gardner-Knopoff windows
        time_window_days, space_window_km = gardner_knopoff_windows(mag)
        
        # Find events within space-time window
        for jdx in range(n_events):
            if jdx == idx or not is_mainshock[jdx]:
                continue
                
            # Check temporal proximity
            time_diff = abs((catalog.iloc[jdx]['time'] - time).total_seconds() / 86400)
            if time_diff > time_window_days:
                continue
                
            # Check spatial proximity
            dist_km = np.linalg.norm(coords_metric[jdx] - coords)
            if dist_km > space_window_km:
                continue
                
            # Event jdx is in the space-time window of idx
            # If jdx is weaker than idx, mark as aftershock
            if catalog.iloc[jdx]['mag'] < mag:
                is_mainshock[jdx] = False
    
    declustered_indices = np.where(is_mainshock)[0]
    n_removed = n_events - len(declustered_indices)
    
    return declustered_indices, n_removed

def run_declustering_analysis():
    """
    Run declustering analysis for all 7 Pan-American regions.
    
    Replicates Paper Nov 2025 Table 3 results.
    """
    print("=" * 80)
    print("GARDNER-KNOPOFF DECLUSTERING ANALYSIS - PAPER NOV 2025")
    print("=" * 80)
    print()
    
    data_acq = SeismicDataAcquisition()
    fractal_est = FractalDimensionEstimator()
    
    # Define regions
    PAN_AMERICAN_7 = {
        "San Andreas Fault": PanAmericanPresets.SAN_ANDREAS,
        "Cascadia Subduction": PanAmericanPresets.CASCADIA,
        "Cocos Plate (Mesoamerica)": PanAmericanPresets.COCOS_PLATE,
        "Caribbean Plate (Lesser Antilles)": PanAmericanPresets.CARIBBEAN,
        "Andes North (Colombia)": PanAmericanPresets.ANDES_NORTH,
        "Andes Central (Peru-Chile)": PanAmericanPresets.ANDES_CENTRAL,
        "Andes South (Chile-Argentina)": PanAmericanPresets.ANDES_SOUTH,
    }
    
    results = []
    
    for region_name, bounds in PAN_AMERICAN_7.items():
        print(f"\n{'=' * 80}")
        print(f"Region: {region_name}")
        print(f"{'=' * 80}")
        
        # Fetch data with min_mag = 2.4 (user preference)
        try:
            data = data_acq.retrieve_catalog(
                region_name, bounds, min_magnitude=2.4,
                start_year=2010, end_date="2025-11-22"
            )
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            continue
            
        if not data or data.get('event_count', 0) == 0:
            print(f"  ⚠️ WARNING: No data")
            continue
            
        n_raw = data['event_count']
        coords_metric = data['coordinates_metric']
        coords_norm = data['coordinates_normalized']
        catalog = data['catalog']
        
        print(f"  Raw catalog: {n_raw} events")
        
        # Compute raw D₂
        print("  Computing raw D₂...")
        result_raw = fractal_est.compute_dimension(
            coords_norm,
            method='gp',
            bootstrap_iterations=100
        )
        d2_raw, d2_raw_sem = result_raw  # tuple unpacking
        
        print(f"    Raw D₂ = {d2_raw:.3f} ± {d2_raw_sem:.3f}")
        
        # Apply declustering
        print("  Applying Gardner-Knopoff declustering...")
        decl_indices, n_removed = decluster_catalog(catalog, coords_metric)
        n_decl = len(decl_indices)
        removal_pct = 100 * n_removed / n_raw
        
        print(f"    Removed: {n_removed} events ({removal_pct:.1f}%)")
        print(f"    Declustered catalog: {n_decl} events")
        
        # Compute declustered D₂
        if n_decl < 100:
            print(f"    ⚠️ WARNING: Too few events ({n_decl}) for reliable D₂")
            d2_decl = np.nan
            d2_decl_sem = np.nan
        else:
            print("  Computing declustered D₂...")
            coords_decl = coords_norm[decl_indices]
            result_decl = fractal_est.compute_dimension(
                coords_decl,
                method='gp',
                bootstrap_iterations=100
            )
            d2_decl, d2_decl_sem = result_decl  # tuple unpacking
            
            print(f"    Declustered D₂ = {d2_decl:.3f} ± {d2_decl_sem:.3f}")
        
        # Compute ΔD₂
        if not np.isnan(d2_decl):
            delta_d2 = d2_decl - d2_raw
            delta_pct = 100 * delta_d2 / d2_raw
            print(f"    ΔD₂ = {delta_d2:+.3f} ({delta_pct:+.1f}%)")
            
            # Physical interpretation
            if delta_d2 > 0.2:
                print("    🔬 PHYSICAL INSIGHT: Large positive shift")
                print("       → Aftershock clusters (planar) masked volumetric background")
            elif delta_d2 < -0.2:
                print("    🔬 PHYSICAL INSIGHT: Large negative shift")
                print("       → Finite-size underestimation (catalog too small after declustering)")
            else:
                print("    ✅ D₂ robust to aftershock clustering")
        else:
            delta_d2 = np.nan
            delta_pct = np.nan
        
        # Store results
        results.append({
            'region': region_name,
            'n_raw': n_raw,
            'd2_raw': d2_raw,
            'd2_raw_sem': d2_raw_sem,
            'n_declustered': n_decl,
            'd2_declustered': d2_decl,
            'd2_decl_sem': d2_decl_sem,
            'n_removed': n_removed,
            'removal_pct': removal_pct,
            'delta_d2': delta_d2,
            'delta_pct': delta_pct
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fractal_analysis_output/declustering_analysis_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - DECLUSTERING SENSITIVITY")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    print(f"Results saved to: {output_file}")
    
    # Compare with Paper Nov 2025 Table 3
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER NOV 2025 TABLE 3")
    print("=" * 80)
    
    paper_table3 = {
        'San Andreas Fault': {'raw': 1.93, 'decl': 1.93, 'delta': 0.00},
        'Cascadia Subduction': {'raw': 1.83, 'decl': 1.69, 'delta': -0.14},
        'Caribbean Plate (Lesser Antilles)': {'raw': 1.58, 'decl': 2.07, 'delta': +0.49},
        'Andes Central (Peru-Chile)': {'raw': 2.24, 'decl': 1.98, 'delta': -0.26},
        'Andes South (Chile-Argentina)': {'raw': 2.01, 'decl': 1.99, 'delta': -0.02},
    }
    
    for region in df['region']:
        if region in paper_table3:
            paper = paper_table3[region]
            measured = df[df['region'] == region].iloc[0]
            
            print(f"\n{region}:")
            print(f"  Paper:    Raw {paper['raw']:.2f} → Decl {paper['decl']:.2f} (Δ{paper['delta']:+.2f})")
            print(f"  Measured: Raw {measured['d2_raw']:.2f} → Decl {measured['d2_declustered']:.2f} (Δ{measured['delta_d2']:+.2f})")
            
            raw_diff = abs(measured['d2_raw'] - paper['raw'])
            decl_diff = abs(measured['d2_declustered'] - paper['decl'])
            
            if raw_diff < 0.1 and decl_diff < 0.1:
                print("  ✅ EXCELLENT match with Paper")
            elif raw_diff < 0.2 or decl_diff < 0.2:
                print("  ⚠️  Acceptable match (within ±0.2)")
            else:
                print("  ❌ DISCREPANCY - investigate region bounds/Mc")
    
    return df

if __name__ == "__main__":
    results_df = run_declustering_analysis()
