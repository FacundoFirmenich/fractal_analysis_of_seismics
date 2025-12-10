"""
Data acquisition module for Seismic Fractal Analysis.
Handles USGS FDSN queries and Pan-American regional presets.
"""

import requests
import pandas as pd

# import numpy as np # Unused
import io
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from .utils import geographic_to_metric, normalize_coordinates


class PanAmericanPresets:
    """
    Pre-defined bounding boxes for the 'Systemic Americas' analysis.
    Coordinates are [min_lat, max_lat, min_lon, max_lon, min_depth, max_depth].
    """

    # --- North America ---
    ALASKA_ALEUTIANS_E = (50.0, 60.0, -170.0, -155.0, 0, 200)
    ALASKA_ALEUTIANS_W = (
        50.0,
        56.0,
        170.0,
        180.0,
        0,
        200,
    )  # Crosses dateline handled by API?
    ALASKA_DENALI = (60.0, 66.0, -154.0, -140.0, 0, 40)
    CASCADIA = (40.0, 50.0, -130.0, -122.0, 0, 100)
    SAN_ANDREAS = (32.0, 38.0, -122.0, -115.0, 0, 30)
    BASIN_AND_RANGE = (35.0, 42.0, -120.0, -110.0, 0, 40)
    YELLOWSTONE = (44.0, 46.0, -112.0, -109.0, 0, 20)
    NEW_MADRID = (35.0, 38.0, -92.0, -88.0, 0, 30)
    QUEEN_CHARLOTTE = (50.0, 55.0, -135.0, -130.0, 0, 40)

    # --- Mexico & Central America ---
    BAJA_CALIFORNIA = (23.0, 32.0, -116.0, -108.0, 0, 30)
    RIVERA_PLATE = (18.0, 22.0, -109.0, -104.0, 0, 50)
    COCOS_PLATE = (10.0, 20.0, -105.0, -90.0, 0, 100)
    MOTAGUA_FAULT = (14.0, 16.0, -92.0, -88.0, 0, 30)
    PANAMA_BLOCK = (7.0, 10.0, -83.0, -77.0, 0, 50)

    # --- Caribbean ---
    CARIBBEAN = (10.0, 19.0, -66.0, -58.0, 0, 300)
    PUERTO_RICO_TRENCH = (18.0, 20.0, -68.0, -64.0, 0, 100)
    HISPANIOLA = (17.0, 20.0, -75.0, -68.0, 0, 100)
    CAYMAN_TROUGH = (17.0, 20.0, -85.0, -75.0, 0, 50)

    # --- South America ---
    ANDES_NORTH = (0.0, 10.0, -80.0, -70.0, 0, 200)
    BUCARAMANGA_NEST = (6.0, 7.5, -73.5, -72.5, 140, 180)  # Deep nest
    ECUADOR_TRENCH = (-3.0, 1.0, -82.0, -78.0, 0, 100)
    ANDES_CENTRAL = (-25.0, -15.0, -75.0, -65.0, 0, 300)
    PERU_FLAT_SLAB = (-15.0, -5.0, -80.0, -70.0, 0, 150)
    ALTIPLANO_PUNA = (-24.0, -21.0, -69.0, -66.0, 0, 300)
    ANDES_SOUTH = (-40.0, -30.0, -75.0, -68.0, 0, 150)
    VALDIVIA_RUPTURE = (-47.0, -38.0, -76.0, -71.0, 0, 60)
    CHILE_RISE = (-48.0, -43.0, -78.0, -73.0, 0, 30)
    MAGALLANES_FAGNANO = (-55.0, -52.0, -75.0, -65.0, 0, 40)
    SCOTIA_ARC = (-60.0, -55.0, -30.0, -20.0, 0, 100)
    SANDWICH_PLATE = (-60.0, -55.0, -28.0, -24.0, 0, 200)

    @classmethod
    def get_all_regions(
        cls,
    ) -> Dict[str, Tuple[float, float, float, float, float, float]]:
        """Return a dictionary of all Pan-American presets."""
        return {
            "Alaska - Aleutians East": cls.ALASKA_ALEUTIANS_E,
            "Alaska - Aleutians West": cls.ALASKA_ALEUTIANS_W,
            "Alaska - Denali Fault": cls.ALASKA_DENALI,
            "Cascadia Subduction": cls.CASCADIA,
            "San Andreas Fault": cls.SAN_ANDREAS,
            "Basin and Range": cls.BASIN_AND_RANGE,
            "Yellowstone Hotspot": cls.YELLOWSTONE,
            "New Madrid Seismic Zone": cls.NEW_MADRID,
            "Queen Charlotte Fault": cls.QUEEN_CHARLOTTE,
            "Baja California": cls.BAJA_CALIFORNIA,
            "Rivera Plate": cls.RIVERA_PLATE,
            "Cocos Plate (Mesoamerica)": cls.COCOS_PLATE,
            "Motagua-Polochic Fault": cls.MOTAGUA_FAULT,
            "Panama Block": cls.PANAMA_BLOCK,
            "Caribbean Plate (Lesser Antilles)": cls.CARIBBEAN,
            "Puerto Rico Trench": cls.PUERTO_RICO_TRENCH,
            "Hispaniola": cls.HISPANIOLA,
            "Cayman Trough": cls.CAYMAN_TROUGH,
            "Andes North (Colombia)": cls.ANDES_NORTH,
            "Bucaramanga Nest": cls.BUCARAMANGA_NEST,
            "Ecuador Trench": cls.ECUADOR_TRENCH,
            "Andes Central (Peru-Chile)": cls.ANDES_CENTRAL,
            "Peru Flat Slab": cls.PERU_FLAT_SLAB,
            "Altiplano-Puna": cls.ALTIPLANO_PUNA,
            "Andes South (Chile-Argentina)": cls.ANDES_SOUTH,
            "Valdivia Rupture Zone": cls.VALDIVIA_RUPTURE,
            "Chile Rise": cls.CHILE_RISE,
            "Magallanes-Fagnano": cls.MAGALLANES_FAGNANO,
            "Scotia Arc": cls.SCOTIA_ARC,
            "South Sandwich Trench": cls.SANDWICH_PLATE,
        }


class SeismicDataAcquisition:
    """USGS seismic catalog acquisition with geodetic transformation."""

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def retrieve_catalog(
        self,
        region_name: str,
        spatial_bounds: Tuple[float, float, float, float, float, float],
        min_magnitude: float = 2.5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve earthquake catalog from USGS FDSN.

        Args:
            region_name: Name of the region (for logging).
            spatial_bounds: Tuple(
                min_lat, max_lat, min_lon, max_lon, min_depth, max_depth
            ).
            min_magnitude: Minimum magnitude cutoff.
            start_date: Start date (YYYY-MM-DD format). Takes precedence
                over start_year.
            end_date: End date (YYYY-MM-DD format). Takes precedence over
                end_year.
            start_year: Start year for query (used if start_date not provided).
            end_year: End year for query (used if end_date not provided).

        Returns:
            Dictionary containing catalog DataFrame and processed coordinates,
            or None if failed.
        """
        (min_lat, max_lat, min_lon, max_lon, min_depth, max_depth) = spatial_bounds

        # Determine start time
        if start_date is not None:
            start_time = start_date
        elif start_year is not None:
            start_time = f"{start_year}-01-01"
        else:
            start_time = "2010-01-01"  # Default

        # Determine end time
        if end_date is not None:
            end_time = end_date
        elif end_year is not None:
            end_time = f"{end_year}-12-31"
        else:
            end_time = datetime.now().strftime("%Y-%m-%d")

        params = {
            "format": "csv",
            "limit": 20000,
            "starttime": start_time,
            "endtime": end_time,
            "minlatitude": min_lat,
            "maxlatitude": max_lat,
            "minlongitude": min_lon,
            "maxlongitude": max_lon,
            "mindepth": min_depth,
            "maxdepth": max_depth,
            "minmagnitude": min_magnitude,
            "orderby": "time",
        }

        try:
            print(f"Fetching data for {region_name} from USGS...")
            
            # CRITICAL: Pagination for catalogs >20k events
            all_data = []
            offset = 1
            batch_size = 20000
            
            while True:
                params_paginated = params.copy()
                params_paginated['offset'] = offset
                
                response = requests.get(self.BASE_URL, params=params_paginated, timeout=60)
                
                if response.status_code != 200:
                    if offset == 1:
                        print(f"Error: USGS API returned status {response.status_code}")
                        return None
                    else:
                        # End of data reached
                        break
                
                # Parse CSV batch
                batch_df = pd.read_csv(io.StringIO(response.text))
                
                if len(batch_df) == 0:
                    # No more data
                    break
                
                all_data.append(batch_df)
                print(f"  Batch {offset//batch_size + 1}: {len(batch_df)} events")
                
                # Check if we got less than limit (last batch)
                if len(batch_df) < batch_size:
                    break
                
                offset += batch_size
            
            if not all_data:
                print(f"Error: No data retrieved for {region_name}")
                return None
            
            # Combine all batches
            df = pd.concat(all_data, ignore_index=True)
            print(f"  Total events retrieved: {len(df)}")

            # Basic validation
            required_cols = ["time", "latitude", "longitude", "depth", "mag"]
            if not all(col in df.columns for col in required_cols):
                print("Error: Missing required columns in USGS response")
                return None

            df = df[required_cols].dropna()
            df["time"] = pd.to_datetime(df["time"])

            if len(df) < 100:
                print(f"Warning: Insufficient data for {region_name} (N={len(df)})")
                return None

            # Coordinate transformation pipeline
            raw_coords = df[["longitude", "latitude", "depth"]].values
            metric_coords = geographic_to_metric(raw_coords)
            normalized_coords = normalize_coordinates(metric_coords)

            print(f"Successfully retrieved {len(df)} events for {region_name}")

            return {
                "catalog": df,
                "coordinates_normalized": normalized_coords,
                "coordinates_metric": metric_coords,
                "event_count": len(df),
                "time_span": (df["time"].min(), df["time"].max()),
            }

        except Exception as error:
            print(f"Data acquisition failed for {region_name}: {error}")
            return None
