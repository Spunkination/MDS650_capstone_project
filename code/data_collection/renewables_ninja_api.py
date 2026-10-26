"""
Renewables.ninja API client for solar PV data collection.

This module provides a client for interacting with the renewables.ninja API.
Requires an API key (free registration at https://www.renewables.ninja/).
"""

import os
import requests
import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API configuration
API_URL = "https://api.renewables.ninja/1/solar_pv"
API_KEY = os.getenv("RENEWABLES_NINJA_API_KEY", "")


class RenewablesNinjaClient:
    """Client for the renewables.ninja API."""

    def __init__(self, api_key: str = None):
        """
        Initialize the RenewablesNinjaClient.

        Args:
            api_key: API key for the renewables.ninja API
        """
        self.api_key = api_key or API_KEY
        self.api_url = API_URL

    def collect_solar_pv_data(
        self,
        latitude: float,
        longitude: float,
        power: float = 1.0,
        date_from: str = "2020-01-01",
        date_to: str = "2025-12-31",
        output_file: str = "data/raw/solar_pv_data.csv"
    ) -> pd.DataFrame:
        """
        Collect solar PV data from the renewables.ninja API.

        Args:
            latitude: Latitude of the location (e.g., -33.8688 for Sydney)
            longitude: Longitude of the location (e.g., 151.2093 for Sydney)
            power: Installed capacity in MW (default: 1.0 MW)
            date_from: Start date (format: YYYY-MM-DD)
            date_to: End date (format: YYYY-MM-DD)
            output_file: Output file path for the collected data

        Returns:
            DataFrame with the collected solar PV data
        """
        if not self.api_key:
            logging.error(
                "API key not found. "
                "Set RENEWABLES_NINJA_API_KEY environment variable "
                "or register at https://www.renewables.ninja/"
            )
            return None

        # API request parameters
        params = {
            "lat": latitude,
            "lon": longitude,
            "power": power,
            "date_from": date_from,
            "date_to": date_to
        }

        # API headers
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Make API request
        try:
            logging.info(f"Collecting solar PV data for ({latitude}, {longitude})...")
            response = requests.get(self.api_url, params=params, headers=headers)
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Save to file
            df.to_csv(output_file, index=False)
            logging.info(f"Data saved to {output_file}")

            return df

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

    def collect_data_for_nsw_regions(self) -> None:
        """
        Collect solar PV data for NSW regions.

        Returns:
            None
        """
        # NSW regions (latitude, longitude)
        nsw_regions = [
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
            {"name": "Newcastle", "lat": -32.9266, "lon": 151.7787},
            {"name": "Wollongong", "lat": -34.4266, "lon": 150.8907},
            {"name": "Central Coast", "lat": -33.3166, "lon": 151.2093},
            {"name": "Dubbo", "lat": -32.2466, "lon": 148.5857},
            {"name": "Orange", "lat": -33.2866, "lon": 149.0857},
            {"name": "Bathurst", "lat": -33.4166, "lon": 149.5857},
            {"name": "Tamworth", "lat": -31.0866, "lon": 150.9257}
        ]

        for region in nsw_regions:
            logging.info(f"Collecting data for {region['name']}...")
            df = self.collect_solar_pv_data(
                latitude=region["lat"],
                longitude=region["lon"],
                power=1.0,
                date_from="2020-01-01",
                date_to="2025-12-31",
                output_file=f"data/raw/solar_pv_{region['name'].lower()}.csv"
            )
            if df is not None:
                logging.info(f"Data collected for {region['name']}")


if __name__ == "__main__":
    # Collect data for NSW regions
    client = RenewablesNinjaClient()
    client.collect_data_for_nsw_regions()
