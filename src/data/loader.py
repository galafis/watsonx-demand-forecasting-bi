"""Load sales data from CSV and Parquet files with validation and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Expected column schema after normalization
REQUIRED_COLUMNS = {"date", "sku", "quantity"}
OPTIONAL_COLUMNS = {"revenue", "price", "store_id", "category", "channel"}


def load_sales_data(
    path: str | Path,
    date_col: str = "date",
    sku_col: str = "sku",
    quantity_col: str = "quantity",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load sales data from CSV or Parquet file.

    Normalizes column names and validates the required schema.

    Args:
        path: Path to the data file (CSV or Parquet).
        date_col: Name of the date column in the source file.
        sku_col: Name of the SKU/product identifier column.
        quantity_col: Name of the quantity/demand column.
        parse_dates: Whether to parse the date column as datetime.

    Returns:
        DataFrame with normalized columns: date, sku, quantity, and any optional columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing after rename.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("loading_sales_data", path=str(path), format=path.suffix)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or Parquet.")

    # Rename columns to standard names
    rename_map = {date_col: "date", sku_col: "sku", quantity_col: "quantity"}
    df = df.rename(columns=rename_map)

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    # Parse dates
    if parse_dates:
        df["date"] = pd.to_datetime(df["date"])

    # Sort and set index
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    # Basic quality checks
    n_skus = df["sku"].nunique()
    date_range = f"{df['date'].min()} to {df['date'].max()}"
    logger.info(
        "sales_data_loaded",
        rows=len(df),
        n_skus=n_skus,
        date_range=date_range,
        columns=list(df.columns),
    )

    return df


def load_multiple_sources(
    paths: list[str | Path],
    date_col: str = "date",
    sku_col: str = "sku",
    quantity_col: str = "quantity",
) -> pd.DataFrame:
    """Load and concatenate sales data from multiple files.

    Args:
        paths: List of file paths to load.
        date_col: Name of the date column.
        sku_col: Name of the SKU column.
        quantity_col: Name of the quantity column.

    Returns:
        Concatenated DataFrame with data from all sources.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = load_sales_data(p, date_col=date_col, sku_col=sku_col, quantity_col=quantity_col)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["sku", "date"]).reset_index(drop=True)

    logger.info("multiple_sources_loaded", total_rows=len(combined), n_files=len(paths))
    return combined


def fill_missing_dates(
    df: pd.DataFrame,
    freq: str = "D",
    fill_value: float = 0.0,
    sku_col: str = "sku",
) -> pd.DataFrame:
    """Fill missing dates in time series data with specified frequency.

    Args:
        df: Input DataFrame with date and sku columns.
        freq: Date frequency string (e.g., 'D' for daily, 'W' for weekly).
        fill_value: Value to fill for missing quantity entries.
        sku_col: Name of the SKU column.

    Returns:
        DataFrame with complete date range per SKU.
    """
    filled_frames: list[pd.DataFrame] = []

    for sku, group in df.groupby(sku_col):
        date_range = pd.date_range(
            start=group["date"].min(),
            end=group["date"].max(),
            freq=freq,
        )
        full_dates = pd.DataFrame({"date": date_range, sku_col: sku})
        merged = full_dates.merge(group, on=["date", sku_col], how="left")
        merged["quantity"] = merged["quantity"].fillna(fill_value)
        filled_frames.append(merged)

    result = pd.concat(filled_frames, ignore_index=True)
    logger.info("missing_dates_filled", freq=freq, rows=len(result))
    return result
