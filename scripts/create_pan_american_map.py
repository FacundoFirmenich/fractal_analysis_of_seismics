"""
Pan-American Map Visualization - 7 Regions
==========================================
Creates publication-ready map showing all 7 analyzed regions with:
- Geographic boundaries
- D₂ values (color-coded)
- N events (size-coded)
- Communities count (annotations)

Uses Plotly for interactive/static export (PNG/PDF).
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np

def create_pan_american_map(csv_path="fractal_analysis_output/pan_american_results_20251207_205739_COMPLETE.csv"):
    """
    Generate Pan-American map with all 7 regions.
    
    Parameters:
        csv_path: Path to completed results CSV
    
    Returns:
        Plotly figure object
    """
    # Read results
    df = pd.read_csv(csv_path, comment='#')
    
    # Parse spatial bounds
    def parse_bounds(bounds_str):
        """Extract (lat_min, lat_max, lon_min, lon_max) from bounds string"""
        bounds = eval(bounds_str)  # Safe here, controlled input
        lat_min, lat_max, lon_min, lon_max = bounds[:4]
        return lat_min, lat_max, lon_min, lon_max
    
    # Extract region centers and bounds
    regions_data = []
    for idx, row in df.iterrows():
        lat_min, lat_max, lon_min, lon_max = parse_bounds(row['spatial_bounds'])
        
        # Center point
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        
        regions_data.append({
            'name': row['region'],
            'lat_center': lat_center,
            'lon_center': lon_center,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'n_events': row['n_events'],
            'd2': row['d2_gp'],
            'communities': row['n_communities'],
            'D_graph': row['D_graph'],
            'morans_i': row['morans_i']
        })
    
    regions_df = pd.DataFrame(regions_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add rectangle boundaries for each region
    colors = {
        'San Andreas Fault': '#FF6B6B',
        'Cascadia Subduction': '#4ECDC4',
        'Cocos Plate (Mesoamerica)': '#45B7D1',
        'Caribbean Plate (Lesser Antilles)': '#96CEB4',
        'Andes North (Colombia)': '#FFEAA7',
        'Andes Central (Peru-Chile)': '#DFE6E9',
        'Andes South (Chile-Argentina)': '#74B9FF'
    }
    
    for idx, region in regions_df.iterrows():
        # Rectangle boundary
        lats_rect = [region['lat_min'], region['lat_max'], region['lat_max'], region['lat_min'], region['lat_min']]
        lons_rect = [region['lon_min'], region['lon_min'], region['lon_max'], region['lon_max'], region['lon_min']]
        
        color = colors.get(region['name'], '#333333')
        
        fig.add_trace(go.Scattergeo(
            lon=lons_rect,
            lat=lats_rect,
            mode='lines',
            line=dict(width=2, color=color),
            name=region['name'],
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Center marker (sized by N events, colored by D₂)
        d2_val = region['d2'] if not np.isnan(region['d2']) else 2.0
        
        fig.add_trace(go.Scattergeo(
            lon=[region['lon_center']],
            lat=[region['lat_center']],
            mode='markers+text',
            marker=dict(
                size=np.sqrt(region['n_events']) / 20,  # Scale by sqrt for visibility
                color=d2_val,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="D₂",
                    x=1.15
                ),
                cmin=1.5,
                cmax=2.8,
                line=dict(width=2, color='white')
            ),
            text=f"{region['communities']}",
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            name='',
            hovertemplate=(
                f"<b>{region['name']}</b><br>" +
                f"N events: {region['n_events']:,}<br>" +
                f"D₂: {d2_val:.3f}<br>" +
                f"Communities: {region['communities']}<br>" +
                f"D_Graph: {region['D_graph']:.3f}<br>" +
                f"Moran's I: {region['morans_i']:.3f}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Pan-American Seismic Fractal Analysis - 7 Regions<br><sub>Marker size ∝ √N events | Color = D₂ | Number = Communities</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Arial')
        ),
        geo=dict(
            scope='south america',
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showlakes=True,
            lakecolor='rgb(200, 220, 240)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            lonaxis=dict(range=[-140, -50]),
            lataxis=dict(range=[-45, 55]),
            center=dict(lat=10, lon=-95)
        ),
        height=900,
        width=1200,
        font=dict(family='Arial', size=12),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

if __name__ == "__main__":
    print("Generating Pan-American map...")
    
    fig = create_pan_american_map()
    
    # Save interactive HTML
    html_path = "fractal_analysis_output/pan_american_map.html"
    fig.write_html(html_path)
    print(f"✅ Interactive map saved: {html_path}")
    
    # Save static PNG (high resolution)
    png_path = "fractal_analysis_output/pan_american_map.png"
    try:
        fig.write_image(png_path, width=1200, height=900, scale=2)
        print(f"✅ Static PNG saved: {png_path}")
    except:
        print(f"⚠️ PNG export requires kaleido: pip install kaleido")
    
    # Save PDF (vector, publication quality)
    pdf_path = "fractal_analysis_output/pan_american_map.pdf"
    try:
        fig.write_image(pdf_path, width=1200, height=900)
        print(f"✅ PDF saved: {pdf_path}")
    except:
        print(f"⚠️ PDF export requires kaleido: pip install kaleido")
    
    print("\n✅ Map generation complete!")
    print(f"   Open {html_path} in browser for interactive version")
