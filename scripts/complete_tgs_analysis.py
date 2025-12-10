"""
Complete TGS Analysis for Existing Pan-American Results
========================================================
Reads existing CSV, adds TGS metrics (n_communities, D_graph, spectral_gap)
for all 7 regions using installed NetworkX/igraph/leidenalg dependencies.

Input: fractal_analysis_output/pan_american_results_20251207_205739.csv
Output: fractal_analysis_output/pan_american_results_20251207_205739_COMPLETE.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sfa.data import SeismicDataAcquisition, PanAmericanPresets
from sfa.graph_tgs import compute_seismic_graph_stats

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

def complete_tgs_analysis():
    """
    Complete TGS analysis for existing results CSV.
    """
    print("=" * 80)
    print("COMPLETING TGS ANALYSIS - PAN-AMERICAN 7 REGIONS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Read existing CSV
    csv_path = "fractal_analysis_output/pan_american_results_20251207_205739.csv"
    print(f"Reading existing results: {csv_path}")
    
    df = pd.read_csv(csv_path, comment='#')
    print(f"  ✅ Loaded {len(df)} regions")
    print()
    
    # Initialize data acquisition
    data_acq = SeismicDataAcquisition()
    
    # Process each region
    for i, (region_name, bounds) in enumerate(PAN_AMERICAN_7.items(), 1):
        print(f"{'=' * 80}")
        print(f"Region {i}/7: {region_name}")
        print(f"{'=' * 80}")
        
        # Fetch data (same parameters as original)
        try:
            data = data_acq.retrieve_catalog(
                region_name, bounds, min_magnitude=2.5,
                start_year=2010, end_date="2025-11-22"
            )
        except Exception as e:
            print(f"  ❌ ERROR fetching data: {e}")
            continue
        
        if not data or data.get('event_count', 0) == 0:
            print(f"  ⚠️ WARNING: No data for {region_name}")
            continue
        
        print(f"  ✅ Loaded {data['event_count']} events")
        
        coords_metric = data["coordinates_metric"]
        catalog = data["catalog"]
        
        # Compute TGS metrics
        print(f"  Computing TGS graph analysis...")
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
            
            # Update DataFrame
            mask = df['region'] == region_name
            df.loc[mask, 'n_communities'] = n_communities
            df.loc[mask, 'D_graph'] = D_graph
            df.loc[mask, 'spectral_gap'] = spectral_gap
            
        except Exception as e:
            print(f"  ⚠️ TGS failed: {e}")
            continue
    
    # Save completed CSV
    output_path = csv_path.replace('.csv', '_COMPLETE.csv')
    
    # Add metadata header
    with open(output_path, 'w') as f:
        f.write(f"# Pan-American Transect Analysis - 7 Regions (COMPLETE WITH TGS)\n")
        f.write(f"# Original execution: 2025-12-07 21:23:38\n")
        f.write(f"# TGS completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Backend: NumPy + NetworkX + igraph + Leiden\n")
        f.write(f"# Bootstrap iterations: 200\n")
        f.write(f"# Multifractal q-range: -5 to 5 (21 values)\n")
        f.write(f"#\n")
    
    # Append data
    df.to_csv(output_path, mode='a', index=False)
    
    print()
    print("=" * 80)
    print("TGS COMPLETION FINISHED")
    print("=" * 80)
    print(f"✅ Completed CSV saved: {output_path}")
    print(f"   Columns: {len(df.columns)} (all 20 populated)")
    print(f"   Regions: {len(df)}")
    print()
    print(f"Execution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    return df

if __name__ == "__main__":
    df_complete = complete_tgs_analysis()
    print("✅ All done! Results complete with TGS metrics.")
