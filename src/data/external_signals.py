"""External signal integration: weather API, promotional calendar, economic indicators."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Weather API integration (stub-based for offline development)
# ---------------------------------------------------------------------------

class WeatherSignalProvider:
    """Provide weather data as external features for demand forecasting.

    Uses OpenWeatherMap API when available; falls back to synthetic data
    for offline development and testing.
    """

    def __init__(self, api_key: str = "", api_url: str = "") -> None:
        self.api_key = api_key
        self.api_url = api_url or "https://api.openweathermap.org/data/2.5"
        self._available = bool(api_key)

    def get_weather_features(
        self,
        dates: list[date],
        city: str = "Sao Paulo",
    ) -> pd.DataFrame:
        """Get weather features for the specified dates and city.

        Returns DataFrame with columns: date, temperature_avg, precipitation_mm,
        humidity_pct, is_rainy_day.

        Args:
            dates: List of dates to get weather data for.
            city: City name for weather lookup.

        Returns:
            DataFrame with weather features.
        """
        if self._available:
            return self._fetch_weather_api(dates, city)
        return self._generate_synthetic_weather(dates, city)

    def _fetch_weather_api(self, dates: list[date], city: str) -> pd.DataFrame:
        """Fetch weather data from the OpenWeatherMap API.

        Note: In production, this would make real API calls. Currently a stub
        that returns synthetic data matching the expected schema.
        """
        logger.info("weather_api_fetch", city=city, n_dates=len(dates))
        # Stub: return synthetic data even when API key is set
        return self._generate_synthetic_weather(dates, city)

    def _generate_synthetic_weather(self, dates: list[date], city: str) -> pd.DataFrame:
        """Generate synthetic weather data for development and testing."""
        import numpy as np

        rng = np.random.default_rng(42)
        n = len(dates)

        # Temperature: seasonal pattern (Southern Hemisphere)
        day_of_year = pd.Series([d.timetuple().tm_yday for d in dates])
        # Peak summer around Jan (day ~15), peak winter around Jul (day ~195)
        temp_base = 22 - 8 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        temperature = temp_base + rng.normal(0, 2, n)

        # Precipitation: higher in summer months
        precip_base = 4 + 3 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
        precipitation = np.maximum(0, precip_base + rng.exponential(2, n) - 2)

        humidity = 60 + 15 * np.sin(2 * np.pi * (day_of_year - 60) / 365) + rng.normal(0, 5, n)
        humidity = np.clip(humidity, 30, 100)

        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "temperature_avg": np.round(temperature, 1),
            "precipitation_mm": np.round(precipitation, 1),
            "humidity_pct": np.round(humidity, 1),
            "is_rainy_day": (precipitation > 2.0).astype(int),
        })

        logger.info("synthetic_weather_generated", city=city, rows=len(df))
        return df


# ---------------------------------------------------------------------------
# Promotional calendar
# ---------------------------------------------------------------------------

class PromotionalCalendar:
    """Manage promotional events that impact demand.

    Stores promotional periods with expected demand impact multipliers.
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def add_event(
        self,
        name: str,
        start_date: date,
        end_date: date,
        impact_multiplier: float = 1.5,
        category: str = "all",
    ) -> None:
        """Register a promotional event.

        Args:
            name: Promotion name.
            start_date: Start of the promotional period.
            end_date: End of the promotional period (inclusive).
            impact_multiplier: Expected demand multiplier (1.5 = +50%).
            category: Product category affected ('all' for store-wide).
        """
        self._events.append({
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "impact_multiplier": impact_multiplier,
            "category": category,
        })

    def get_promotional_features(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Add promotional features to a DataFrame.

        Adds columns: is_promotion, promotion_name, promotion_multiplier.

        Args:
            df: Input DataFrame with date column.
            date_col: Name of the date column.

        Returns:
            DataFrame with promotional feature columns.
        """
        df = df.copy()
        df["is_promotion"] = 0
        df["promotion_name"] = ""
        df["promotion_multiplier"] = 1.0

        for event in self._events:
            mask = (
                (df[date_col].dt.date >= event["start_date"])
                & (df[date_col].dt.date <= event["end_date"])
            )
            df.loc[mask, "is_promotion"] = 1
            df.loc[mask, "promotion_name"] = event["name"]
            df.loc[mask, "promotion_multiplier"] = event["impact_multiplier"]

        logger.info("promotional_features_added", n_events=len(self._events))
        return df


# ---------------------------------------------------------------------------
# Economic indicators (Selic, IPCA)
# ---------------------------------------------------------------------------

class EconomicIndicators:
    """Provide Brazilian economic indicators as external features.

    Includes Selic rate (basic interest rate) and IPCA (inflation index).
    Uses static reference data; in production would integrate with BCB API.
    """

    # Monthly reference data (can be updated or fetched from BCB API)
    _SELIC_MONTHLY: dict[str, float] = {
        "2024-01": 11.75, "2024-02": 11.25, "2024-03": 10.75,
        "2024-04": 10.50, "2024-05": 10.50, "2024-06": 10.50,
        "2024-07": 10.50, "2024-08": 10.50, "2024-09": 10.75,
        "2024-10": 11.25, "2024-11": 11.25, "2024-12": 12.25,
        "2025-01": 13.25, "2025-02": 13.25, "2025-03": 14.25,
        "2025-04": 14.25, "2025-05": 14.75, "2025-06": 14.75,
        "2025-07": 14.75, "2025-08": 14.75, "2025-09": 14.75,
        "2025-10": 14.75, "2025-11": 14.75, "2025-12": 14.75,
    }

    _IPCA_MONTHLY: dict[str, float] = {
        "2024-01": 0.42, "2024-02": 0.83, "2024-03": 0.16,
        "2024-04": 0.38, "2024-05": 0.46, "2024-06": 0.21,
        "2024-07": 0.38, "2024-08": -0.02, "2024-09": 0.44,
        "2024-10": 0.56, "2024-11": 0.39, "2024-12": 0.52,
        "2025-01": 0.16, "2025-02": 1.31, "2025-03": 0.56,
        "2025-04": 0.43, "2025-05": 0.36, "2025-06": 0.40,
        "2025-07": 0.35, "2025-08": 0.30, "2025-09": 0.35,
        "2025-10": 0.40, "2025-11": 0.35, "2025-12": 0.45,
    }

    def get_economic_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Add economic indicator features to a DataFrame.

        Adds columns: selic_rate, ipca_monthly, ipca_accumulated_12m.

        Args:
            df: Input DataFrame with date column.
            date_col: Name of the date column.

        Returns:
            DataFrame with economic indicator columns.
        """
        df = df.copy()

        # Map monthly Selic rate
        df["_month_key"] = df[date_col].dt.strftime("%Y-%m")
        df["selic_rate"] = df["_month_key"].map(self._SELIC_MONTHLY).fillna(14.75)

        # Map monthly IPCA
        df["ipca_monthly"] = df["_month_key"].map(self._IPCA_MONTHLY).fillna(0.40)

        # 12-month accumulated IPCA (approximate)
        monthly_values = pd.Series(self._IPCA_MONTHLY)
        accumulated: dict[str, float] = {}
        sorted_months = sorted(self._IPCA_MONTHLY.keys())
        for i, month in enumerate(sorted_months):
            start = max(0, i - 11)
            window = [self._IPCA_MONTHLY[sorted_months[j]] for j in range(start, i + 1)]
            accumulated[month] = round(sum(window), 2)

        df["ipca_accumulated_12m"] = df["_month_key"].map(accumulated).fillna(4.50)
        df = df.drop(columns=["_month_key"])

        logger.info("economic_features_added", columns=["selic_rate", "ipca_monthly", "ipca_accumulated_12m"])
        return df
