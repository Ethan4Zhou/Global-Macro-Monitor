"""Valuation helpers and scoring models."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class ValuationSnapshot(BaseModel):
    """Basic valuation snapshot for a market."""

    country: str
    equity_index: float = Field(..., description="Equity index level.")
    market_cap_to_gdp: float = Field(..., description="Market capitalization divided by GDP.")
    yield_10y: float = Field(..., description="10-year sovereign bond yield.")

    @property
    def equity_valuation_signal(self) -> str:
        """Return a placeholder equity valuation label."""
        if self.market_cap_to_gdp > 1.2:
            return "Rich"
        if self.market_cap_to_gdp < 0.8:
            return "Cheap"
        return "Fair"


def _zscore(series: pd.Series) -> pd.Series:
    """Return a safe z-score series for valuation features."""
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    return (numeric - numeric.mean()) / std


def compute_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Compute a standardized cheapness score where higher means cheaper conditions."""
    components: list[pd.Series] = []

    if "buffett_indicator" in df.columns:
        components.append(-_zscore(df["buffett_indicator"]))
    if "real_yield" in df.columns:
        components.append(-_zscore(df["real_yield"]))
    if "term_spread" in df.columns:
        components.append(_zscore(df["term_spread"]))
    if "equity_risk_proxy" in df.columns:
        components.append(_zscore(df["equity_risk_proxy"]))
    if "credit_spread_proxy" in df.columns:
        components.append(-_zscore(df["credit_spread_proxy"]))

    if not components:
        return pd.Series(dtype="float64")

    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def label_valuation_regime(valuation_score: float) -> str:
    """Convert valuation score into a simple cheap/fair/expensive label."""
    if pd.isna(valuation_score):
        return "unknown"
    if valuation_score >= 0.5:
        return "cheap"
    if valuation_score <= -0.5:
        return "expensive"
    return "fair"
