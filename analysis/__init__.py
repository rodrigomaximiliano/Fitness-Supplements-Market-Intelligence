"""
Analysis package for the Fitness Supplements Market project.

Currently exposes:
- Data loading utilities.
- Core metrics helpers (CAGR, YoY, seasonality, volatility, regression).
"""

from .data_loader import load_trends_5y, load_trends_12m  # noqa: F401
from .metrics import (  # noqa: F401
    KEYWORDS,
    LABELS,
    RegressionResult,
    cagr,
    cagr_table,
    yearly_averages,
    yoy_momentum,
    seasonality_table,
    volatility_table,
    quarterly_averages,
    linear_trend,
)


