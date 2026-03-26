"""Tests for statistical forecasting models: ARIMAForecaster, ETSForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.statistical import ARIMAForecaster, BaseForecaster, ETSForecaster

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_series() -> pd.Series:
    """Create a synthetic daily time series with trend and weekly seasonality."""
    rng = np.random.default_rng(42)
    n = 120
    t = np.arange(n)
    # Trend + weekly seasonality + noise
    trend = 100 + 0.3 * t
    seasonality = 15 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 5, n)
    values = trend + seasonality + noise
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(values, index=index, name="quantity")


@pytest.fixture
def short_series() -> pd.Series:
    """Create a short time series for edge case testing."""
    rng = np.random.default_rng(42)
    values = 50 + rng.normal(0, 3, 30)
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.Series(values, index=index, name="quantity")


# ---------------------------------------------------------------------------
# BaseForecaster
# ---------------------------------------------------------------------------


class TestBaseForecaster:
    """Tests for BaseForecaster abstract interface."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseForecaster(name="test")  # type: ignore[abstract]

    def test_arima_is_base_forecaster(self) -> None:
        model = ARIMAForecaster(order=(1, 0, 0), auto_order=False)
        assert isinstance(model, BaseForecaster)

    def test_ets_is_base_forecaster(self) -> None:
        model = ETSForecaster()
        assert isinstance(model, BaseForecaster)


# ---------------------------------------------------------------------------
# ARIMAForecaster
# ---------------------------------------------------------------------------


class TestARIMAForecaster:
    """Tests for ARIMA / SARIMAX forecaster."""

    def test_init_default(self) -> None:
        model = ARIMAForecaster()
        assert model.name == "ARIMA"
        assert model.is_fitted is False

    def test_init_custom_order(self) -> None:
        model = ARIMAForecaster(order=(2, 1, 2), auto_order=False)
        assert model.order == (2, 1, 2)
        assert model.auto_order is False

    def test_fit(self, daily_series: pd.Series) -> None:
        model = ARIMAForecaster(order=(1, 0, 1), auto_order=False, seasonal_order=(0, 0, 0, 0))
        model.fit(daily_series)
        assert model.is_fitted is True
        assert model._fitted_model is not None

    def test_predict_returns_correct_length(self, daily_series: pd.Series) -> None:
        model = ARIMAForecaster(order=(1, 0, 1), auto_order=False, seasonal_order=(0, 0, 0, 0))
        model.fit(daily_series)
        forecast = model.predict(horizon=30)
        assert len(forecast) == 30
        assert forecast.name == "ARIMA_forecast"

    def test_predict_values_reasonable(self, daily_series: pd.Series) -> None:
        model = ARIMAForecaster(order=(1, 1, 1), auto_order=False, seasonal_order=(0, 0, 0, 0))
        model.fit(daily_series)
        forecast = model.predict(horizon=10)
        # Forecasts should be in a reasonable range relative to the data
        data_mean = daily_series.mean()
        assert all(abs(v - data_mean) < 5 * daily_series.std() for v in forecast)

    def test_predict_without_fit_raises(self) -> None:
        model = ARIMAForecaster(order=(1, 0, 0), auto_order=False)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(horizon=10)

    def test_fit_predict(self, daily_series: pd.Series) -> None:
        model = ARIMAForecaster(order=(1, 0, 1), auto_order=False, seasonal_order=(0, 0, 0, 0))
        forecast = model.fit_predict(daily_series, horizon=15)
        assert len(forecast) == 15
        assert model.is_fitted is True

    def test_predict_interval(self, daily_series: pd.Series) -> None:
        model = ARIMAForecaster(order=(1, 0, 1), auto_order=False, seasonal_order=(0, 0, 0, 0))
        model.fit(daily_series)
        intervals = model.predict_interval(horizon=10, alpha=0.05)
        assert "forecast" in intervals.columns
        assert "lower" in intervals.columns
        assert "upper" in intervals.columns
        assert len(intervals) == 10
        # Lower should be less than upper
        assert (intervals["lower"] <= intervals["upper"]).all()

    def test_predict_interval_without_fit_raises(self) -> None:
        model = ARIMAForecaster(order=(1, 0, 0), auto_order=False)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_interval(horizon=10)

    def test_auto_order_determines_d(self, daily_series: pd.Series) -> None:
        model = ARIMAForecaster(auto_order=True, seasonal_order=(0, 0, 0, 0))
        model.fit(daily_series)
        assert model.is_fitted is True
        # d should be 0, 1, or 2
        assert model.order[1] in [0, 1, 2]

    def test_determine_d_stationary_series(self) -> None:
        rng = np.random.default_rng(42)
        stationary = pd.Series(rng.normal(100, 5, 100))
        model = ARIMAForecaster()
        d = model._determine_d(stationary)
        assert d == 0

    def test_determine_d_trended_series(self) -> None:
        t = np.arange(100)
        trended = pd.Series(50 + 2 * t + np.random.default_rng(42).normal(0, 1, 100))
        model = ARIMAForecaster()
        d = model._determine_d(trended)
        assert d >= 1


# ---------------------------------------------------------------------------
# ETSForecaster
# ---------------------------------------------------------------------------


class TestETSForecaster:
    """Tests for ETS / Holt-Winters forecaster."""

    def test_init_default(self) -> None:
        model = ETSForecaster()
        assert model.name == "ETS"
        assert model.is_fitted is False

    def test_init_custom_params(self) -> None:
        model = ETSForecaster(error="mul", trend="add", seasonal="mul", seasonal_periods=30)
        assert model.error == "mul"
        assert model.seasonal_periods == 30

    def test_fit(self, daily_series: pd.Series) -> None:
        model = ETSForecaster(seasonal_periods=7)
        model.fit(daily_series)
        assert model.is_fitted is True
        assert model._fitted_model is not None

    def test_predict_returns_correct_length(self, daily_series: pd.Series) -> None:
        model = ETSForecaster(seasonal_periods=7)
        model.fit(daily_series)
        forecast = model.predict(horizon=30)
        assert len(forecast) == 30
        assert forecast.name == "ETS_forecast"

    def test_predict_values_not_nan(self, daily_series: pd.Series) -> None:
        model = ETSForecaster(seasonal_periods=7)
        model.fit(daily_series)
        forecast = model.predict(horizon=10)
        assert not forecast.isna().any()

    def test_predict_without_fit_raises(self) -> None:
        model = ETSForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(horizon=10)

    def test_fit_predict(self, daily_series: pd.Series) -> None:
        model = ETSForecaster(seasonal_periods=7)
        forecast = model.fit_predict(daily_series, horizon=15)
        assert len(forecast) == 15
        assert model.is_fitted is True

    def test_multiplicative_with_non_positive_falls_back_to_additive(self) -> None:
        """ETS with multiplicative components should fall back to additive for non-positive data."""
        rng = np.random.default_rng(42)
        # Series with some zeros
        values = np.maximum(0, rng.normal(10, 15, 60))
        values[5] = 0.0
        index = pd.date_range("2024-01-01", periods=60, freq="D")
        y = pd.Series(values, index=index)

        model = ETSForecaster(error="mul", seasonal="mul", seasonal_periods=7)
        model.fit(y)
        assert model.is_fitted is True
        # Should have switched to additive
        assert model.error == "add"
        assert model.seasonal == "add"

    def test_forecast_reasonable_range(self, daily_series: pd.Series) -> None:
        model = ETSForecaster(seasonal_periods=7)
        model.fit(daily_series)
        forecast = model.predict(horizon=7)
        # Forecast should be within a reasonable range
        data_mean = daily_series.mean()
        data_std = daily_series.std()
        for v in forecast:
            assert abs(v - data_mean) < 10 * data_std
