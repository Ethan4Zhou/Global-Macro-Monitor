"""Eurozone-specific valuation helpers."""

from __future__ import annotations

import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    """Return a safe z-score series."""
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    return (numeric - numeric.mean()) / std


def compute_eurozone_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Compute a proxy-based Eurozone valuation score where higher means cheaper."""
    components: list[pd.Series] = []
    if "equity_pe_proxy" in df.columns:
        components.append(-_zscore(df["equity_pe_proxy"]))
    if "equity_pb_proxy" in df.columns:
        components.append(-_zscore(df["equity_pb_proxy"]))
    if "real_yield_proxy" in df.columns:
        components.append(-_zscore(df["real_yield_proxy"]))
    elif "real_yield" in df.columns:
        components.append(-_zscore(df["real_yield"]))
    if "term_spread" in df.columns:
        components.append(_zscore(df["term_spread"]))
    if not components:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def label_eurozone_valuation_regime(score: float) -> str:
    """Map Eurozone valuation score to a simple cheap/fair/expensive bucket."""
    if pd.isna(score):
        return "unknown"
    if score >= 0.5:
        return "cheap"
    if score <= -0.5:
        return "expensive"
    return "fair"
