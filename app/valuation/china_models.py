"""China-specific valuation helpers."""

from __future__ import annotations

import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    """Return a safe z-score series."""
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=numeric.index, dtype="float64")
    return (numeric - numeric.mean()) / std


def compute_china_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Compute a proxy-based China valuation score where higher means cheaper."""
    components: list[pd.Series] = []

    if "hs300_pe_proxy" in df.columns:
        components.append(-_zscore(df["hs300_pe_proxy"]))
    if "hs300_pb_proxy" in df.columns:
        components.append(-_zscore(df["hs300_pb_proxy"]))
    if "real_yield_proxy" in df.columns:
        components.append(-_zscore(df["real_yield_proxy"]))
    elif "real_yield" in df.columns:
        components.append(-_zscore(df["real_yield"]))
    if "term_spread" in df.columns:
        components.append(_zscore(df["term_spread"]))

    if not components:
        return pd.Series(float("nan"), index=df.index, dtype="float64")
    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def label_china_valuation_regime(score: float) -> str:
    """Map China valuation score to a simple rich/fair/cheap bucket."""
    if pd.isna(score):
        return "unknown"
    if score >= 0.5:
        return "cheap"
    if score <= -0.5:
        return "expensive"
    return "fair"
