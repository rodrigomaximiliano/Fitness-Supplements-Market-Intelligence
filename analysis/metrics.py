from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


KEYWORDS: List[str] = ["protein_powder", "creatine", "pre_workout", "weight_loss", "omega_3"]

LABELS = {
    "creatine": "Creatine",
    "protein_powder": "Protein Powder",
    "pre_workout": "Pre Workout",
    "omega_3": "Omega 3",
    "weight_loss": "Weight Loss Supps",
}


def cagr(start: float, end: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if start <= 0 or years <= 0:
        raise ValueError("start and years must be > 0 to compute CAGR")
    return (end / start) ** (1 / years) - 1


def yearly_averages(df: pd.DataFrame, keywords: Iterable[str] = KEYWORDS) -> pd.DataFrame:
    """Average index per year and keyword."""
    return df.groupby("Year")[list(keywords)].mean().round(1)


def cagr_table(
    yearly: pd.DataFrame,
    start_year: int,
    end_year: int,
    keywords: Iterable[str] = KEYWORDS,
) -> pd.DataFrame:
    """Build a CAGR summary table similar to the notebook output."""
    y_start = yearly.loc[start_year]
    y_end = yearly.loc[end_year]
    n_years = end_year - start_year

    rows = []
    for k in keywords:
        start_val = float(y_start[k])
        end_val = float(y_end[k])
        total_growth = (end_val - start_val) / start_val
        rows.append(
            {
                "Keyword": LABELS.get(k, k),
                f"Avg {start_year}": round(start_val, 1),
                f"Avg {end_year}": round(end_val, 1),
                "Total Growth %": round(total_growth * 100, 1),
                "CAGR %": round(cagr(start_val, end_val, n_years) * 100, 1),
            }
        )

    return pd.DataFrame(rows).sort_values("CAGR %", ascending=False).reset_index(drop=True)


def yoy_momentum(df: pd.DataFrame, keywords: Iterable[str] = KEYWORDS) -> pd.DataFrame:
    """
    Year-over-year momentum: last 12 months vs previous 12 months.
    Returns a dataframe with previous / last 12m averages and YoY %.
    """
    keywords = list(keywords)
    cutoff = df["Week"].max()
    last12 = df[df["Week"] > cutoff - pd.DateOffset(months=12)]
    prev12 = df[(df["Week"] > cutoff - pd.DateOffset(months=24)) & (df["Week"] <= cutoff - pd.DateOffset(months=12))]

    rows = []
    for k in keywords:
        prev_mean = float(prev12[k].mean())
        last_mean = float(last12[k].mean())
        yoy_change = (last_mean - prev_mean) / prev_mean if prev_mean != 0 else np.nan
        rows.append(
            {
                "Keyword": LABELS.get(k, k),
                "Prev 12mo": round(prev_mean, 1),
                "Last 12mo": round(last_mean, 1),
                "YoY Change %": round(yoy_change * 100, 1) if not np.isnan(yoy_change) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("YoY Change %", ascending=False).reset_index(drop=True)


def seasonality_table(df: pd.DataFrame, keywords: Iterable[str] = KEYWORDS) -> pd.DataFrame:
    """Average index per month (by MonthName) and keyword."""
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return (
        df.groupby("MonthName")[list(keywords)]
        .mean()
        .reindex(month_order)
        .round(1)
    )


def volatility_table(df: pd.DataFrame, keywords: Iterable[str] = KEYWORDS) -> pd.DataFrame:
    """Volatility metrics: mean, std dev, coefficient of variation, min, max."""
    rows = []
    for k in keywords:
        series = df[k]
        mean_val = float(series.mean())
        std_val = float(series.std())
        cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan
        rows.append(
            {
                "Keyword": LABELS.get(k, k),
                "Mean": round(mean_val, 1),
                "Std Dev": round(std_val, 1),
                "CV %": round(cv, 1) if not np.isnan(cv) else np.nan,
                "Min": round(float(series.min()), 1),
                "Max": round(float(series.max()), 1),
            }
        )

    return pd.DataFrame(rows).sort_values("CV %", na_position="last").reset_index(drop=True)


def quarterly_averages(df: pd.DataFrame, keywords: Iterable[str] = KEYWORDS) -> pd.DataFrame:
    """Average index per quarter and keyword."""
    q = df.groupby("Quarter")[list(keywords)].mean().round(1)
    q.index = ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"]
    return q


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float


def linear_trend(series: pd.Series) -> RegressionResult:
    """Simple linear regression over a time-indexed series."""
    x_vals = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, series.values)
    return RegressionResult(
        slope=slope,
        intercept=intercept,
        r_value=r_value,
        p_value=p_value,
        std_err=std_err,
    )

