from __future__ import annotations

from pathlib import Path

import pandas as pd


KEYWORDS = ["protein_powder", "creatine", "pre_workout", "weight_loss", "omega_3"]


def _load_trends_csv(csv_path: Path) -> pd.DataFrame:
    """
    Internal helper: read a Google Trends CSV (5y or 12m) and normalize columns.
    Assumes the export format used in this project: two header rows,
    then weekly data.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Google Trends file not found at: {csv_path}")

    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = [
        "Week",
        "protein_powder",
        "creatine",
        "pre_workout",
        "weight_loss",
        "omega_3",
    ]

    df["Week"] = pd.to_datetime(df["Week"])
    df = df.sort_values("Week").reset_index(drop=True)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common derived time features to a trends dataframe."""
    df = df.copy()
    df["Year"] = df["Week"].dt.year
    df["Month"] = df["Week"].dt.month
    df["Quarter"] = df["Week"].dt.quarter
    df["MonthName"] = df["Week"].dt.strftime("%b")
    return df


def load_trends_5y(csv_path: str | Path = "data/multiTimeline-lastfiveyears.csv") -> pd.DataFrame:
    """
    Load and clean the 5-year Google Trends dataset.

    - Reads the CSV exported from Google Trends (with the two-header rows).
    - Normalizes column names to snake_case.
    - Parses the Week column as datetime.
    - Adds derived time columns: Year, Month, Quarter, MonthName.
    """
    csv_path = Path(csv_path)
    df = _load_trends_csv(csv_path)
    return _add_time_features(df)


def load_trends_12m(csv_path: str | Path = "data/multiTimelinelast12monts.csv") -> pd.DataFrame:
    """
    Load and clean the last-12-months Google Trends dataset.

    Structure is identical to the 5-year file, but with a shorter time span.
    Useful to zoom into the latest dynamics without recomputing everything.
    """
    csv_path = Path(csv_path)
    df = _load_trends_csv(csv_path)
    return _add_time_features(df)

