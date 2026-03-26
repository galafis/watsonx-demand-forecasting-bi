"""Feature engineering for time series: lag features, rolling statistics, calendar features."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "quantity",
    lags: Optional[list[int]] = None,
    group_col: str = "sku",
) -> pd.DataFrame:
    """Add lagged features to the DataFrame.

    Args:
        df: Input DataFrame sorted by date within each group.
        target_col: Column to compute lags on.
        lags: List of lag periods. Defaults to config values [1, 7, 14, 28].
        group_col: Column to group by before computing lags.

    Returns:
        DataFrame with added lag columns named '{target_col}_lag_{n}'.
    """
    if lags is None:
        lags = settings.features_config.get("lag_features", [1, 7, 14, 28])

    df = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        logger.debug("lag_feature_added", lag=lag, column=col_name)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "quantity",
    windows: Optional[list[int]] = None,
    statistics: Optional[list[str]] = None,
    group_col: str = "sku",
) -> pd.DataFrame:
    """Add rolling window statistics to the DataFrame.

    Args:
        df: Input DataFrame sorted by date within each group.
        target_col: Column to compute rolling statistics on.
        windows: List of window sizes. Defaults to config values [7, 14, 30].
        statistics: List of aggregation functions. Defaults to ['mean', 'std', 'min', 'max'].
        group_col: Column to group by.

    Returns:
        DataFrame with added rolling feature columns.
    """
    feat_cfg = settings.features_config
    if windows is None:
        windows = feat_cfg.get("rolling_windows", [7, 14, 30])
    if statistics is None:
        statistics = feat_cfg.get("rolling_statistics", ["mean", "std", "min", "max"])

    df = df.copy()
    for window in windows:
        rolling = df.groupby(group_col)[target_col].rolling(window=window, min_periods=1)
        for stat in statistics:
            col_name = f"{target_col}_roll_{stat}_{window}"
            if stat == "mean":
                values = rolling.mean()
            elif stat == "std":
                values = rolling.std()
            elif stat == "min":
                values = rolling.min()
            elif stat == "max":
                values = rolling.max()
            else:
                logger.warning("unknown_rolling_statistic", stat=stat)
                continue

            # Reset multi-level index from groupby + rolling
            df[col_name] = values.reset_index(level=0, drop=True)
            logger.debug("rolling_feature_added", window=window, stat=stat, column=col_name)

    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features from the date column.

    Features include: day_of_week, day_of_month, month, quarter, week_of_year,
    is_weekend, is_month_start, is_month_end.

    Args:
        df: Input DataFrame with a datetime date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with added calendar feature columns.
    """
    df = df.copy()
    dt = df[date_col]

    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    logger.info("calendar_features_added", n_features=8)
    return df


def add_cyclical_encoding(
    df: pd.DataFrame,
    col: str,
    period: int,
) -> pd.DataFrame:
    """Add sine/cosine cyclical encoding for a periodic feature.

    Args:
        df: Input DataFrame.
        col: Column to encode cyclically.
        period: Period length (e.g., 7 for day_of_week, 12 for month).

    Returns:
        DataFrame with '{col}_sin' and '{col}_cos' columns added.
    """
    df = df.copy()
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


def build_features(
    df: pd.DataFrame,
    target_col: str = "quantity",
    group_col: str = "sku",
    include_cyclical: bool = True,
) -> pd.DataFrame:
    """Build the complete feature set for time series modeling.

    Applies lag features, rolling statistics, calendar features, and optional
    cyclical encoding in sequence.

    Args:
        df: Input DataFrame with date, sku, and quantity columns.
        target_col: Target column name.
        group_col: Group column for per-SKU feature computation.
        include_cyclical: Whether to add cyclical encoding for calendar features.

    Returns:
        Feature-enriched DataFrame.
    """
    logger.info("building_features", rows=len(df), skus=df[group_col].nunique())

    df = add_lag_features(df, target_col=target_col, group_col=group_col)
    df = add_rolling_features(df, target_col=target_col, group_col=group_col)
    df = add_calendar_features(df)

    if include_cyclical:
        df = add_cyclical_encoding(df, "day_of_week", period=7)
        df = add_cyclical_encoding(df, "month", period=12)

    logger.info("features_built", total_columns=len(df.columns))
    return df
