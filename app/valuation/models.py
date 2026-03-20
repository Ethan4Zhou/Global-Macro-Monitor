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


def _alias_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first available numeric series from a list of candidate columns."""
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(float("nan"), index=df.index, dtype="float64")


def build_valuation_component_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a standardized valuation component frame across regions.

    Higher component values always mean cheaper / more supportive conditions.
    """
    pe_proxy = _alias_series(df, ["equity_pe_proxy", "hs300_pe_proxy"])
    shiller_pe_proxy = _alias_series(df, ["shiller_pe_proxy"])
    pb_proxy = _alias_series(df, ["equity_pb_proxy", "hs300_pb_proxy"])
    real_yield_proxy = _alias_series(df, ["real_yield_proxy", "real_yield"])
    term_spread = _alias_series(df, ["term_spread"])
    equity_risk_proxy = _alias_series(df, ["equity_risk_proxy"])
    credit_spread_proxy = _alias_series(df, ["credit_spread_proxy"])
    buffett_indicator = _alias_series(df, ["buffett_indicator"])

    return pd.DataFrame(
        {
            "equity_pe_proxy_component": -_zscore(pe_proxy),
            "shiller_pe_proxy_component": -_zscore(shiller_pe_proxy),
            "equity_pb_proxy_component": -_zscore(pb_proxy),
            "real_yield_proxy_component": -_zscore(real_yield_proxy),
            "term_spread_component": _zscore(term_spread),
            "equity_risk_proxy_component": _zscore(equity_risk_proxy),
            "credit_spread_proxy_component": -_zscore(credit_spread_proxy),
            "buffett_indicator_component": -_zscore(buffett_indicator),
        },
        index=df.index,
    )


def build_weighted_block_score(
    component_frame: pd.DataFrame,
    block_weights: dict[str, tuple[list[str], float]],
) -> pd.Series:
    """Aggregate valuation components into weighted thematic blocks.

    Each tuple contains (component_columns, block_weight). Missing blocks are
    dropped and the remaining weights are renormalized row by row.
    """
    block_values: dict[str, pd.Series] = {}
    block_weight_frame = pd.DataFrame(index=component_frame.index)
    for block_name, (columns, weight) in block_weights.items():
        available_columns = [column for column in columns if column in component_frame.columns]
        if not available_columns:
            block_values[block_name] = pd.Series(float("nan"), index=component_frame.index, dtype="float64")
            block_weight_frame[block_name] = 0.0
            continue
        block_series = component_frame[available_columns].mean(axis=1, skipna=True)
        block_values[block_name] = block_series
        block_weight_frame[block_name] = block_series.notna().astype(float) * float(weight)

    if not block_values:
        return pd.Series(float("nan"), index=component_frame.index, dtype="float64")

    block_frame = pd.DataFrame(block_values, index=component_frame.index)
    total_weight = block_weight_frame.sum(axis=1).mask(lambda series: series == 0.0, float("nan"))
    weighted_sum = (block_frame * block_weight_frame).sum(axis=1, skipna=True)
    return pd.to_numeric(weighted_sum.divide(total_weight), errors="coerce")


def compute_valuation_confidence(
    df: pd.DataFrame,
    expected_inputs: list[str] | None = None,
) -> pd.Series:
    """Compute a simple valuation confidence bucket from actual available inputs."""
    expected = expected_inputs or [
        "equity_pe_proxy",
        "shiller_pe_proxy",
        "equity_pb_proxy",
        "real_yield_proxy",
        "term_spread",
        "equity_risk_proxy",
        "credit_spread_proxy",
        "buffett_indicator",
    ]
    aliases = {
        "equity_pe_proxy": ["equity_pe_proxy", "hs300_pe_proxy"],
        "shiller_pe_proxy": ["shiller_pe_proxy"],
        "equity_pb_proxy": ["equity_pb_proxy", "hs300_pb_proxy"],
        "real_yield_proxy": ["real_yield_proxy", "real_yield"],
        "term_spread": ["term_spread"],
        "equity_risk_proxy": ["equity_risk_proxy"],
        "credit_spread_proxy": ["credit_spread_proxy"],
        "buffett_indicator": ["buffett_indicator"],
    }
    availability = pd.DataFrame(index=df.index)
    for input_name in expected:
        availability[input_name] = _alias_series(df, aliases[input_name]).notna()

    counts = availability.sum(axis=1)
    confidence = pd.Series("low", index=df.index, dtype="object")
    confidence = confidence.mask(counts >= 2, "medium")
    confidence = confidence.mask(counts >= 4, "high")
    confidence = confidence.mask(counts == 0, "unknown")
    return confidence


def summarize_valuation_inputs(
    df: pd.DataFrame,
    expected_inputs: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Summarize used and missing valuation inputs as comma-separated strings."""
    aliases = {
        "equity_pe_proxy": ["equity_pe_proxy", "hs300_pe_proxy"],
        "shiller_pe_proxy": ["shiller_pe_proxy"],
        "equity_pb_proxy": ["equity_pb_proxy", "hs300_pb_proxy"],
        "real_yield_proxy": ["real_yield_proxy", "real_yield"],
        "term_spread": ["term_spread"],
        "equity_risk_proxy": ["equity_risk_proxy"],
        "credit_spread_proxy": ["credit_spread_proxy"],
        "buffett_indicator": ["buffett_indicator"],
    }
    used_values: list[str] = []
    missing_values: list[str] = []
    for _, row in df.iterrows():
        used: list[str] = []
        missing: list[str] = []
        for input_name in expected_inputs:
            value = pd.to_numeric(
                pd.Series([row.get(column) for column in aliases[input_name]]),
                errors="coerce",
            ).dropna()
            if value.empty:
                missing.append(input_name)
            else:
                used.append(input_name)
        used_values.append(",".join(used))
        missing_values.append(",".join(missing))
    return (
        pd.Series(used_values, index=df.index, dtype="object"),
        pd.Series(missing_values, index=df.index, dtype="object"),
    )


def compute_valuation_score(df: pd.DataFrame) -> pd.Series:
    """Compute a standardized cheapness score where higher means cheaper conditions."""
    component_frame = build_valuation_component_frame(df)
    components = [
        component_frame[column]
        for column in component_frame.columns
        if component_frame[column].notna().any()
    ]

    if not components:
        return pd.Series(float("nan"), index=df.index, dtype="float64")

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
