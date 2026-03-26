"""Tests for feature engineering module: lag features, rolling features, calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_calendar_features,
    add_cyclical_encoding,
    add_lag_features,
    add_rolling_features,
    build_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing feature engineering."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": dates,
            "sku": "SKU001",
            "quantity": rng.poisson(lam=100, size=60).astype(float),
        }
    )


@pytest.fixture
def multi_sku_df() -> pd.DataFrame:
    """Create a multi-SKU DataFrame for testing grouped operations."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(42)
    frames = []
    for sku in ["SKU001", "SKU002", "SKU003"]:
        df = pd.DataFrame(
            {
                "date": dates,
                "sku": sku,
                "quantity": rng.poisson(lam=80, size=30).astype(float),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Lag Features
# ---------------------------------------------------------------------------


class TestAddLagFeatures:
    """Tests for add_lag_features function."""

    def test_default_lags_created(self, sample_df: pd.DataFrame) -> None:
        result = add_lag_features(sample_df, lags=[1, 7, 14, 28])
        for lag in [1, 7, 14, 28]:
            assert f"quantity_lag_{lag}" in result.columns

    def test_custom_lags(self, sample_df: pd.DataFrame) -> None:
        result = add_lag_features(sample_df, lags=[3, 5])
        assert "quantity_lag_3" in result.columns
        assert "quantity_lag_5" in result.columns
        assert "quantity_lag_1" not in result.columns

    def test_lag_values_correct(self, sample_df: pd.DataFrame) -> None:
        result = add_lag_features(sample_df, lags=[1])
        # The lag-1 value at index 1 should equal the quantity at index 0
        assert result["quantity_lag_1"].iloc[1] == result["quantity"].iloc[0]

    def test_lag_first_rows_nan(self, sample_df: pd.DataFrame) -> None:
        result = add_lag_features(sample_df, lags=[7])
        # First 7 rows should be NaN for lag-7
        assert result["quantity_lag_7"].iloc[:7].isna().all()
        assert result["quantity_lag_7"].iloc[7:].notna().all()

    def test_lag_per_sku_grouping(self, multi_sku_df: pd.DataFrame) -> None:
        result = add_lag_features(multi_sku_df, lags=[1])
        # The first row of each SKU group should have NaN lag
        for sku in multi_sku_df["sku"].unique():
            sku_data = result[result["sku"] == sku]
            assert pd.isna(sku_data["quantity_lag_1"].iloc[0])

    def test_original_df_not_modified(self, sample_df: pd.DataFrame) -> None:
        original_cols = list(sample_df.columns)
        add_lag_features(sample_df, lags=[1])
        assert list(sample_df.columns) == original_cols

    def test_custom_target_col(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df["sales"] = df["quantity"] * 2
        result = add_lag_features(df, target_col="sales", lags=[1])
        assert "sales_lag_1" in result.columns


# ---------------------------------------------------------------------------
# Rolling Features
# ---------------------------------------------------------------------------


class TestAddRollingFeatures:
    """Tests for add_rolling_features function."""

    def test_default_rolling_columns_created(self, sample_df: pd.DataFrame) -> None:
        result = add_rolling_features(sample_df, windows=[7], statistics=["mean", "std"])
        assert "quantity_roll_mean_7" in result.columns
        assert "quantity_roll_std_7" in result.columns

    def test_all_statistics(self, sample_df: pd.DataFrame) -> None:
        result = add_rolling_features(
            sample_df, windows=[7], statistics=["mean", "std", "min", "max"]
        )
        for stat in ["mean", "std", "min", "max"]:
            assert f"quantity_roll_{stat}_7" in result.columns

    def test_multiple_windows(self, sample_df: pd.DataFrame) -> None:
        result = add_rolling_features(sample_df, windows=[7, 14, 30], statistics=["mean"])
        for w in [7, 14, 30]:
            assert f"quantity_roll_mean_{w}" in result.columns

    def test_rolling_mean_values(self, sample_df: pd.DataFrame) -> None:
        result = add_rolling_features(sample_df, windows=[3], statistics=["mean"])
        # After enough data points, rolling mean should be the average of the window
        idx = 10
        expected = sample_df["quantity"].iloc[idx - 2 : idx + 1].mean()
        actual = result["quantity_roll_mean_3"].iloc[idx]
        assert abs(actual - expected) < 1e-6

    def test_rolling_per_sku(self, multi_sku_df: pd.DataFrame) -> None:
        result = add_rolling_features(multi_sku_df, windows=[3], statistics=["mean"])
        # Each SKU group should have its own rolling calculation
        for sku in multi_sku_df["sku"].unique():
            sku_data = result[result["sku"] == sku]
            assert "quantity_roll_mean_3" in sku_data.columns
            # First value should equal itself (min_periods=1)
            assert sku_data["quantity_roll_mean_3"].notna().all()

    def test_unknown_statistic_skipped(self, sample_df: pd.DataFrame) -> None:
        result = add_rolling_features(sample_df, windows=[7], statistics=["mean", "invalid_stat"])
        assert "quantity_roll_mean_7" in result.columns
        assert "quantity_roll_invalid_stat_7" not in result.columns


# ---------------------------------------------------------------------------
# Calendar Features
# ---------------------------------------------------------------------------


class TestAddCalendarFeatures:
    """Tests for add_calendar_features function."""

    def test_calendar_columns_created(self, sample_df: pd.DataFrame) -> None:
        result = add_calendar_features(sample_df)
        expected_cols = [
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "week_of_year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_day_of_week_range(self, sample_df: pd.DataFrame) -> None:
        result = add_calendar_features(sample_df)
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_is_weekend_correct(self) -> None:
        # 2024-01-06 is Saturday, 2024-01-07 is Sunday
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]),
            }
        )
        result = add_calendar_features(df)
        assert result["is_weekend"].tolist() == [0, 1, 1, 0]

    def test_month_start_end(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-31"]),
            }
        )
        result = add_calendar_features(df)
        assert result["is_month_start"].tolist() == [1, 0, 0]
        assert result["is_month_end"].tolist() == [0, 0, 1]

    def test_quarter_values(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]),
            }
        )
        result = add_calendar_features(df)
        assert result["quarter"].tolist() == [1, 2, 3, 4]

    def test_month_values(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-03-15", "2024-06-15", "2024-12-15"]),
            }
        )
        result = add_calendar_features(df)
        assert result["month"].tolist() == [3, 6, 12]


# ---------------------------------------------------------------------------
# Cyclical Encoding
# ---------------------------------------------------------------------------


class TestAddCyclicalEncoding:
    """Tests for add_cyclical_encoding function."""

    def test_sin_cos_columns_created(self, sample_df: pd.DataFrame) -> None:
        df = add_calendar_features(sample_df)
        result = add_cyclical_encoding(df, "day_of_week", period=7)
        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns

    def test_sin_cos_range(self, sample_df: pd.DataFrame) -> None:
        df = add_calendar_features(sample_df)
        result = add_cyclical_encoding(df, "day_of_week", period=7)
        assert result["day_of_week_sin"].min() >= -1.0
        assert result["day_of_week_sin"].max() <= 1.0
        assert result["day_of_week_cos"].min() >= -1.0
        assert result["day_of_week_cos"].max() <= 1.0

    def test_month_encoding_period(self) -> None:
        df = pd.DataFrame({"month": list(range(1, 13))})
        result = add_cyclical_encoding(df, "month", period=12)
        # Month 1 and month 12+1 should be close (cyclical)
        sin_1 = result["month_sin"].iloc[0]
        result["month_cos"].iloc[0]
        # sin(2pi*1/12) should be positive
        assert sin_1 > 0


# ---------------------------------------------------------------------------
# Build Features (integration)
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    """Tests for the build_features integration function."""

    def test_all_feature_groups_present(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df, include_cyclical=True)
        # Lag features
        assert any("lag" in c for c in result.columns)
        # Rolling features
        assert any("roll" in c for c in result.columns)
        # Calendar features
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        # Cyclical encoding
        assert "day_of_week_sin" in result.columns
        assert "month_cos" in result.columns

    def test_build_without_cyclical(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df, include_cyclical=False)
        assert "day_of_week_sin" not in result.columns
        assert "month_cos" not in result.columns
        # But calendar features should still exist
        assert "day_of_week" in result.columns

    def test_build_preserves_original_columns(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)
        assert "date" in result.columns
        assert "sku" in result.columns
        assert "quantity" in result.columns

    def test_build_multi_sku(self, multi_sku_df: pd.DataFrame) -> None:
        result = build_features(multi_sku_df)
        assert len(result) == len(multi_sku_df)
        assert result["sku"].nunique() == 3
