"""
Advanced Seismic Fractal Analysis Script.
Orchestrates Temporal, Declustering, Depth, and Bayesian analyses.
Generates Figures 1-5.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from bokeh.plotting import figure, output_file, save
from sfa.data import SeismicDataAcquisition, PanAmericanPresets
from sfa.core import FractalDimensionEstimator
from sfa.vis import StyleManager

# Ensure output directory exists
OUTPUT_DIR = "advanced_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_advanced_analysis():
    """
    Execute the advanced analysis suite:
    1. Bayesian Dimension Estimation
    2. Declustering Sensitivity (Gardner-Knopoff)
    3. Temporal Stability Analysis
    4. Depth Stratification (Andes)
    """
    print("Starting Advanced Analysis...")
    StyleManager.set_premium_style()

    data_acq = SeismicDataAcquisition()
    fractal_est = FractalDimensionEstimator()

    regions = PanAmericanPresets.get_all_regions()

    # Store results for summary figures
    results_bayesian = {}
    results_temporal = {}
    results_declustering = {}
    results_depth = {}

    # --- 1. MAIN LOOP: Data Fetching & Basic Analysis ---
    region_data = {}

    for name, bounds in regions.items():
        print(f"\nProcessing {name}...")
        # Optimize fetch
        min_mag = 3.5 if name == "Caribbean Plate" else 2.5
        data = data_acq.retrieve_catalog(
            name, bounds, min_magnitude=min_mag, start_year=2010
        )

        if not data:
            continue

        region_data[name] = data

        # A. Bayesian Analysis
        print("  Running Bayesian Inference...")
        bayes_res = fractal_est.compute_bayesian_dimension(
            data["coordinates_normalized"]
        )
        results_bayesian[name] = bayes_res

        # B. Declustering Analysis
        print("  Running Declustering Sensitivity...")
        df_raw = data["catalog"]
        df_declustered = fractal_est.decluster_catalog(df_raw)

        # Re-normalize declustered data
        # Note: We need to re-normalize to the SAME bounding box as raw?
        # Or just re-normalize declustered set. Usually re-normalize to self.
        coords_decl = df_declustered[["longitude", "latitude", "depth"]].values
        # Simple normalization for D2 calc (min-max)
        coords_norm_decl = (coords_decl - coords_decl.min(axis=0)) / (
            coords_decl.max(axis=0) - coords_decl.min(axis=0)
        )

        d2_raw, _ = fractal_est.compute_gp_dimension(
            data["coordinates_normalized"]
        )
        d2_decl, _ = fractal_est.compute_gp_dimension(coords_norm_decl)

        results_declustering[name] = {
            "raw": d2_raw,
            "declustered": d2_decl,
            "n_raw": len(df_raw),
            "n_decl": len(df_declustered),
        }

        # C. Temporal Stability (2010-2015, 2015-2020, 2020-2025)
        print("  Running Temporal Stability...")
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        windows = [
            (
                pd.Timestamp("2010-01-01", tz="UTC"),
                pd.Timestamp("2015-01-01", tz="UTC"),
            ),
            (
                pd.Timestamp("2015-01-01", tz="UTC"),
                pd.Timestamp("2020-01-01", tz="UTC"),
            ),
            (
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2025-01-01", tz="UTC"),
            ),
        ]

        temp_res = []
        for start, end in windows:
            mask = (df_raw["time"] >= start) & (df_raw["time"] < end)
            subset = df_raw[mask]
            if len(subset) > 500:
                coords_sub = subset[["longitude", "latitude", "depth"]].values
                coords_norm_sub = (coords_sub - coords_sub.min(axis=0)) / (
                    coords_sub.max(axis=0) - coords_sub.min(axis=0)
                )
                d2, _ = fractal_est.compute_gp_dimension(coords_norm_sub)
                temp_res.append(d2)
            else:
                temp_res.append(np.nan)
        results_temporal[name] = temp_res

    # D. Depth Stratification (Specific Regions)
    print("\nRunning Depth Stratification...")

    # Andes Central
    if "Andes (Central)" in region_data:
        data = region_data["Andes (Central)"]
        df = data["catalog"]
        layers = [(0, 100), (100, 200), (200, 300)]
        res = []
        for d_min, d_max in layers:
            subset = df[(df["depth"] >= d_min) & (df["depth"] < d_max)]
            if len(subset) > 500:
                coords = subset[["longitude", "latitude", "depth"]].values
                coords_norm = (coords - coords.min(axis=0)) / (
                    coords.max(axis=0) - coords.min(axis=0)
                )
                d2, _ = fractal_est.compute_gp_dimension(coords_norm)
                res.append(d2)
            else:
                res.append(np.nan)
        results_depth["Andes (Central)"] = res

    # Andes South
    if "Andes (South)" in region_data:
        data = region_data["Andes (South)"]
        df = data["catalog"]
        layers = [(0, 80), (80, 150)]
        res = []
        for d_min, d_max in layers:
            subset = df[(df["depth"] >= d_min) & (df["depth"] < d_max)]
            if len(subset) > 500:
                coords = subset[["longitude", "latitude", "depth"]].values
                coords_norm = (coords - coords.min(axis=0)) / (
                    coords.max(axis=0) - coords.min(axis=0)
                )
                d2, _ = fractal_est.compute_gp_dimension(coords_norm)
                res.append(d2)
            else:
                res.append(np.nan)
        results_depth["Andes (South)"] = res

    # --- 2. FIGURE GENERATION ---
    print("\nGenerating Figures...")

    # Figure 1: Study Region Map
    if HAS_CARTOPY:
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=":")
        ax1.add_feature(cfeature.LAND, alpha=0.1)
        ax1.set_global()
        # Focus on Americas
        ax1.set_extent([-130, -60, -60, 60], crs=ccrs.PlateCarree())
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        # Simple world map background if possible, or just grid
        ax1.grid(True, alpha=0.3)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(regions)))
    for i, (name, data) in enumerate(region_data.items()):
        # data["catalog"] has lat/lon
        df = data["catalog"]
        if HAS_CARTOPY:
            ax1.scatter(
                df["longitude"],
                df["latitude"],
                s=1,
                alpha=0.5,
                label=name,
                color=colors[i],
                transform=ccrs.PlateCarree(),
            )
        else:
            ax1.scatter(
                df["longitude"],
                df["latitude"],
                s=1,
                alpha=0.5,
                label=name,
                color=colors[i],
            )

        # Draw box
        bounds = regions[name]
        lat_min, lat_max, lon_min, lon_max = (
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
        )
        rect = plt.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            edgecolor=colors[i],
            linewidth=2,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else ax1.transData,
        )
        ax1.add_patch(rect)

    if not HAS_CARTOPY:
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")

    ax1.set_title("Figure 1: Seismotectonic Study Regions")
    ax1.legend(loc="lower left", markerscale=5)
    fig1.savefig(f"{OUTPUT_DIR}/Figure_1_Map.png", dpi=300)
    fig1.savefig(f"{OUTPUT_DIR}/Figure_1_Map.pdf")
    plt.close(fig1)

    # Figure 2: Correlation Integrals (6-panel)
    # Select 6 key regions
    key_regions = [
        "San Andreas Fault",
        "Cascadia Subduction",
        "Caribbean Plate",
        "Cocos Plate (Mesoamerica)",
        "Andes (South)",
        "Andes (Central)",
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()

    for i, name in enumerate(key_regions):
        if name in region_data:
            ax = axes2[i]
            data = region_data[name]
            # Re-compute curve for plotting
            d2, _, diag = fractal_est.compute_gp_dimension(
                data["coordinates_normalized"], return_diagnostics=True
            )
            if diag and diag["sample_curves"]:
                slope, (log_r, log_c, mask) = diag["sample_curves"][0]

                ax.scatter(log_r, log_c, c="gray", s=10, alpha=0.5)
                if np.any(mask):
                    ax.scatter(log_r[mask], log_c[mask], c="red", s=15)
                    # Fit
                    x_fit = log_r[mask]
                    y_fit = slope * x_fit + (
                        np.mean(log_c[mask]) - slope * np.mean(x_fit)
                    )
                    ax.plot(x_fit, y_fit, "k--", lw=2, label=f"D2={slope:.2f}")

                ax.set_title(f"{name}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if i >= 3:
                    ax.set_xlabel("log(r)")
                if i % 3 == 0:
                    ax.set_ylabel("log(C(r))")
            else:
                ax.text(0.5, 0.5, "Insufficient Data", ha="center")

    plt.tight_layout()
    fig2.savefig(f"{OUTPUT_DIR}/Figure_2_Correlation_Integrals.png", dpi=300)
    fig2.savefig(f"{OUTPUT_DIR}/Figure_2_Correlation_Integrals.pdf")
    plt.close(fig2)

    # Figure 3: 3D Spatial Distributions (4-panel)
    # San Andreas, Cascadia, Andes South, Andes Central
    fig3 = plt.figure(figsize=(15, 12))
    targets = [
        "San Andreas Fault",
        "Cascadia Subduction",
        "Andes (South)",
        "Andes (Central)",
    ]

    for i, name in enumerate(targets):
        if name in region_data:
            ax = fig3.add_subplot(2, 2, i + 1, projection="3d")
            data = region_data[name]
            coords = data["coordinates_metric"]
            mags = data["catalog"]["mag"].values

            p = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                -coords[:, 2],
                c=mags,
                cmap="plasma",
                s=np.exp(mags / 2),
                alpha=0.6,
            )

            ax.set_title(name)
            ax.set_xlabel("E-W (km)")
            ax.set_ylabel("N-S (km)")
            ax.set_zlabel("Depth (km)")
            if i == 1:
                plt.colorbar(p, ax=ax, label="Magnitude", shrink=0.5)

    plt.tight_layout()
    fig3.savefig(f"{OUTPUT_DIR}/Figure_3_3D_Structure.png", dpi=300)
    fig3.savefig(f"{OUTPUT_DIR}/Figure_3_3D_Structure.pdf")
    plt.close(fig3)

    # Figure 4: Results Summary (Bar Chart)
    # Use Bayesian means and stds
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    names = list(results_bayesian.keys())
    means = [results_bayesian[n]["mean"] for n in names]
    stds = [results_bayesian[n]["std"] for n in names]

    ax4.bar(
        names,
        means,
        yerr=stds,
        capsize=5,
        color=sns.color_palette("viridis", len(names)),
    )
    ax4.set_ylabel("Fractal Dimension (D2)")
    ax4.set_title(
        "Figure 4: Comparative Fractal Dimension (Bayesian Estimate)"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig4.savefig(f"{OUTPUT_DIR}/Figure_4_Summary.png", dpi=300)
    fig4.savefig(f"{OUTPUT_DIR}/Figure_4_Summary.pdf")
    plt.close(fig4)

    # Figure 5: Bayesian Posteriors (Violin Plots)
    fig5, ax5 = plt.subplots(figsize=(12, 6))

    # Prepare data for seaborn
    plot_data = []
    for name, res in results_bayesian.items():
        if len(res["samples"]) > 0:
            for s in res["samples"]:
                plot_data.append({"Region": name, "D2": s})

    df_plot = pd.DataFrame(plot_data)
    sns.violinplot(data=df_plot, x="Region", y="D2", ax=ax5, palette="viridis")
    ax5.set_title("Figure 5: Bayesian Posterior Distributions of D2")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig5.savefig(f"{OUTPUT_DIR}/Figure_5_Bayesian_Posteriors.png", dpi=300)
    fig5.savefig(f"{OUTPUT_DIR}/Figure_5_Bayesian_Posteriors.pdf")
    plt.close(fig5)

    # Save Numerical Results to CSV
    print("\nSaving Numerical Results...")

    # Declustering
    pd.DataFrame(results_declustering).T.to_csv(
        f"{OUTPUT_DIR}/results_declustering.csv"
    )

    # Temporal
    pd.DataFrame(
        results_temporal, index=["2010-2015", "2015-2020", "2020-2025"]
    ).T.to_csv(f"{OUTPUT_DIR}/results_temporal.csv")

    # Depth
    with open(f"{OUTPUT_DIR}/results_depth.txt", "w", encoding="utf-8") as f:
        f.write(str(results_depth))

    print(
        "\nAdvanced Analysis Complete. "
        "All figures and tables saved to 'advanced_analysis_output/'."
    )


if __name__ == "__main__":
    run_advanced_analysis()
