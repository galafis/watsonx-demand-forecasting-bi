"""Tests for GraniteTimeSeriesForecaster with mocked Watsonx client."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.granite_ts import GraniteTimeSeriesForecaster
from src.models.statistical import BaseForecaster


@pytest.fixture
def daily_series() -> pd.Series:
    """Create a synthetic daily time series."""
    rng = np.random.default_rng(42)
    n = 90
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 3, n)
    values = trend + seasonality + noise
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(values, index=index, name="quantity")


@pytest.fixture
def granite_forecaster() -> GraniteTimeSeriesForecaster:
    """Forecaster with dummy credentials."""
    return GraniteTimeSeriesForecaster(
        model_id="ibm/granite-tsfm",
        api_key="test-api-key-000",
        project_id="test-project-id-000",
        url="https://us-south.ml.cloud.ibm.com",
    )

class TestGraniteInit:
    """Tests for initialization."""

    def test_is_base_forecaster(self, granite_forecaster):
        assert isinstance(granite_forecaster, BaseForecaster)

    def test_default_name(self, granite_forecaster):
        assert granite_forecaster.name == "Granite-TS"

    def test_not_fitted(self, granite_forecaster):
        assert granite_forecaster.is_fitted is False


class TestInitClient:
    """Tests for _init_client."""

    def test_skips_if_set(self, granite_forecaster):
        granite_forecaster._client = MagicMock()
        orig = granite_forecaster._client
        granite_forecaster._init_client()
        assert granite_forecaster._client is orig


class TestGraniteFit:
    """Tests for fit."""

    def test_stores_history(self, granite_forecaster, daily_series):
        granite_forecaster.fit(daily_series)
        assert granite_forecaster._history is not None
        assert len(granite_forecaster._history) == len(daily_series)

    def test_sets_fitted(self, granite_forecaster, daily_series):
        granite_forecaster.fit(daily_series)
        assert granite_forecaster.is_fitted is True

    def test_copies_data(self, granite_forecaster, daily_series):
        granite_forecaster.fit(daily_series)
        daily_series.iloc[0] = -9999.0
        assert granite_forecaster._history.iloc[0] != -9999.0

class TestGranitePredict:
    """Tests for predict."""

    def test_raises_if_not_fitted(self, granite_forecaster):
        with pytest.raises(RuntimeError, match="fitted"):
            granite_forecaster.predict(horizon=7)

    def test_mocked_response(self, granite_forecaster, daily_series):
        granite_forecaster.fit(daily_series)
        mc = MagicMock()
        mc.generate_text.return_value = "120.5, 121.3, 122.1, 118.9, 119.7, 123.0, 124.2"
        granite_forecaster._client = mc
        result = granite_forecaster.predict(horizon=7)
        assert len(result) == 7
        assert result.name == "Granite-TS_forecast"
        np.testing.assert_array_almost_equal(result.values, [120.5, 121.3, 122.1, 118.9, 119.7, 123.0, 124.2])

    def test_fallback_no_client(self, granite_forecaster, daily_series):
        granite_forecaster.fit(daily_series)
        granite_forecaster._client = None
        with patch.object(granite_forecaster, "_init_client"):
            result = granite_forecaster.predict(horizon=14)
        assert len(result) == 14
        assert (result >= 0).all()

    def test_fallback_api_error(self, granite_forecaster, daily_series):
        granite_forecaster.fit(daily_series)
        mc = MagicMock()
        mc.generate_text.side_effect = RuntimeError("timeout")
        granite_forecaster._client = mc
        result = granite_forecaster.predict(horizon=7)
        assert len(result) == 7


class TestBuildForecastPrompt:
    """Tests for prompt construction."""

    def test_contains_horizon(self, granite_forecaster, daily_series):
        prompt = granite_forecaster._build_forecast_prompt(daily_series, horizon=14)
        assert "14" in prompt

    def test_contains_stats(self, granite_forecaster, daily_series):
        prompt = granite_forecaster._build_forecast_prompt(daily_series, horizon=7)
        assert "Mean:" in prompt

    def test_truncates_90_days(self, granite_forecaster):
        s = pd.Series(np.arange(200, dtype=float))
        prompt = granite_forecaster._build_forecast_prompt(s, horizon=7)
        assert "last 90 days" in prompt

class TestParseForecastResponse:
    """Tests for response parsing."""

    def test_parse_csv(self, granite_forecaster):
        r = granite_forecaster._parse_forecast_response("100.0, 105.5, 110.2", horizon=3)
        np.testing.assert_array_almost_equal(r, [100.0, 105.5, 110.2])

    def test_parse_empty_no_hist(self, granite_forecaster):
        r = granite_forecaster._parse_forecast_response("No data", horizon=3)
        np.testing.assert_array_almost_equal(r, [0.0, 0.0, 0.0])

    def test_parse_empty_with_hist(self, granite_forecaster, daily_series):
        granite_forecaster._history = daily_series
        r = granite_forecaster._parse_forecast_response("Nothing", horizon=5)
        assert len(r) == 5


class TestStatisticalFallback:
    """Tests for fallback."""

    def test_correct_horizon(self, granite_forecaster, daily_series):
        granite_forecaster._history = daily_series
        assert len(granite_forecaster._statistical_fallback(horizon=14)) == 14

    def test_non_negative(self, granite_forecaster, daily_series):
        granite_forecaster._history = daily_series
        assert (granite_forecaster._statistical_fallback(horizon=30) >= 0).all()

    def test_name(self, granite_forecaster, daily_series):
        granite_forecaster._history = daily_series
        assert granite_forecaster._statistical_fallback(horizon=7).name == "Granite-TS_forecast"


class TestFitPredict:
    """Tests for fit_predict."""

    def test_returns_forecast(self, granite_forecaster, daily_series):
        with patch.object(granite_forecaster, "_init_client"):
            result = granite_forecaster.fit_predict(daily_series, horizon=7)
        assert granite_forecaster.is_fitted is True
        assert len(result) == 7
        assert isinstance(result, pd.Series)
