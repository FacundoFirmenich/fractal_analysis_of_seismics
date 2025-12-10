"""
Script to run a full Pan-American analysis and generate all project outputs.
Iterates through all defined regions in PanAmericanPresets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sfa.data import SeismicDataAcquisition, PanAmericanPresets
from sfa.core import FractalDimensionEstimator
from sfa.vis import FractalPlotter


def run_full_analysis():
    """
    Execute the full Pan-American fractal analysis pipeline.

    This function orchestrates data retrieval, fractal dimension estimation,
    and visualization for all regions defined in PanAmericanPresets.
    """
    print("Starting Full Pan-American Analysis...")

    # Initialize modules
    data_acq = SeismicDataAcquisition()
    fractal_est = FractalDimensionEstimator()

    # Create output directories
    os.makedirs("fractal_analysis_output", exist_ok=True)

    regions = PanAmericanPresets.get_all_regions()
    results_summary = []

    for region_name, bounds in regions.items():
        print(f"\nAnalyzing Region: {region_name}")
        print(f"  Bounds: {bounds}")

        # 1. Fetch Data
        # Optimize for Caribbean (too large otherwise)
        min_mag = 3.5 if region_name == "Caribbean Plate" else 2.5
        data = data_acq.retrieve_catalog(
            region_name, bounds, min_magnitude=min_mag, start_year=2010
        )

        if not data:
            print(f"  WARNING: No data found for {region_name}")
            continue

        print(f"  Loaded {data['event_count']} events.")

        # Re-normalize coordinates to unit cube for dimension calculation
        # This preserves relative geometry while ensuring numerical stability
        coords = data["coordinates_normalized"]
        coords_norm = (coords - coords.min(axis=0)) / (
            coords.max(axis=0) - coords.min(axis=0)
        )

        # 2. Compute Fractal Dimension (GP Algorithm)
        print(f"  Computing D2 (Grassberger-Procaccia) for {len(coords)} events...")
        d2_gp, sem_gp, diagnostics = fractal_est.compute_gp_dimension(
            coords_norm, return_diagnostics=True
        )  # type: ignore
        print(f"  D2 (GP): {d2_gp:.3f} +/- {sem_gp:.3f}")

        # 3. Compute Fractal Dimension (Takens)
        print("  Computing Takens Dimension...")
        d2_takens = fractal_est.compute_takens_dimension(coords_norm)
        print(f"  D2 (Takens): {d2_takens:.3f}")

        # 4. Generate Plots
        print("  Generating Plots...")
        safe_name = region_name.replace(" ", "_")

        # Spatial Plot (Static for report)
        fig_spatial = FractalPlotter.plot_spatial_distribution(
            data["coordinates_metric"],
            data["catalog"]["mag"].values,
            region_name,
        )
        FractalPlotter.save_plot(
            fig_spatial, f"fractal_analysis_output/{safe_name}_spatial"
        )
        plt.close(fig_spatial)

        # Correlation Integral
        if diagnostics and diagnostics["sample_curves"]:
            slope, (log_r, log_c, mask) = diagnostics["sample_curves"][0]
            fig_curve = FractalPlotter.plot_correlation_integral(
                log_r, log_c, slope, mask, region_name
            )
            FractalPlotter.save_plot(
                fig_curve, f"fractal_analysis_output/{safe_name}_correlation"
            )
            plt.close(fig_curve)

        # 5. Append Results
        results_summary.append(
            {
                "Region": region_name,
                "Events": data["event_count"],
                "D2_GP": d2_gp,
                "SEM_GP": sem_gp,
                "D2_Takens": d2_takens,
            }
        )

    # 6. Save Summary Table
    if results_summary:
        df_results = pd.DataFrame(results_summary)
        print("\nAnalysis Complete. Saving Summary...")
        print(df_results)

        df_results.to_csv(
            "fractal_analysis_output/pan_american_results.csv", index=False
        )

        # Render summary table image
        fig_table = FractalPlotter.render_table(
            df_results.round(3),
            "fractal_analysis_output/pan_american_summary",
            "Pan-American Fractal Dimension Analysis",
        )
        plt.close(fig_table)

    print("\nFull Analysis Cycle Completed Successfully.")


if __name__ == "__main__":
    run_full_analysis()
