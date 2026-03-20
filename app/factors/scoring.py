"""Factor scoring logic for macro indicators.

The scoring layer intentionally avoids full-sample normalization so that
historical regime panels are not contaminated by future observations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

HISTORICAL_ZSCORE_MIN_PERIODS = 3


def _empty_like(df: pd.DataFrame) -> pd.Series:
    """Create an all-NaN series aligned to a frame."""
    return pd.Series(float("nan"), index=df.index, dtype="float64")


def _get_numeric_series(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first available numeric series from a list of candidates."""
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return _empty_like(df)


def _zscore(series: pd.Series) -> pd.Series:
    """Return a history-aware z-score without future leakage.

    Each row is standardized using only observations available up to that row.
    Early observations with insufficient history stay missing, while flat
    histories with enough observations are treated as neutral.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    expanding_mean = numeric.expanding(min_periods=1).mean()
    expanding_std = numeric.expanding(min_periods=1).std(ddof=0)
    observation_count = numeric.expanding(min_periods=1).count()

    zscore = (numeric - expanding_mean) / expanding_std
    flat_history = expanding_std.isna() | (expanding_std == 0)
    insufficient_history = observation_count < HISTORICAL_ZSCORE_MIN_PERIODS
    neutral_mask = numeric.notna() & (~insufficient_history) & flat_history

    zscore = zscore.mask(neutral_mask, 0.0)
    return zscore.where(numeric.notna() & (~insufficient_history))


def _mean_score(parts: list[pd.Series]) -> pd.Series:
    """Combine standardized components into one score."""
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _staleness_weight(days_stale: pd.Series) -> pd.Series:
    """Convert enrichment staleness into a simple scoring weight."""
    numeric = pd.to_numeric(days_stale, errors="coerce")
    weights = pd.Series(1.0, index=numeric.index, dtype="float64")
    weights = weights.where(numeric <= 90, 0.5)
    weights = weights.where(numeric <= 180, 0.0)
    weights = weights.where(numeric.notna(), 0.0)
    return weights


def _weighted_optional_component(series: pd.Series, days_stale: pd.Series, invert: bool = False) -> pd.Series | None:
    """Return a freshness-aware optional scoring component."""
    if not series.notna().any():
        return None
    weights = _staleness_weight(days_stale)
    if (weights > 0).sum() == 0:
        return None
    component = _zscore(series) * weights
    return -component if invert else component


def _infer_country(df: pd.DataFrame, country: str | None = None) -> str:
    """Infer the country from the feature frame when not supplied."""
    if country:
        return country
    if "country" in df.columns and df["country"].dropna().any():
        return str(df["country"].dropna().iloc[0])
    return "unknown"


def compute_growth_score(df: pd.DataFrame, country: str | None = None) -> pd.Series:
    """Compute a country-aware growth score."""
    inferred_country = _infer_country(df, country)
    components: list[pd.Series] = []
    pmi_3m_avg = _get_numeric_series(df, "pmi_3m_avg")
    pmi_diff_3m = _get_numeric_series(df, "pmi_diff_3m")
    if pmi_3m_avg.notna().any():
        components.append(_zscore(pmi_3m_avg))
    if pmi_diff_3m.notna().any():
        components.append(_zscore(pmi_diff_3m))
    if inferred_country == "china":
        industrial_production_yoy = _get_numeric_series(df, "industrial_production_yoy")
        unrate_3m_avg = _get_numeric_series(df, "unrate_3m_avg")
        unrate_diff_3m = _get_numeric_series(df, "unrate_diff_3m")
        industrial_days_stale = _get_numeric_series(df, "industrial_production_days_stale")
        unrate_days_stale = _get_numeric_series(df, "unrate_days_stale")
        industrial_component = _weighted_optional_component(industrial_production_yoy, industrial_days_stale)
        if industrial_component is not None:
            components.append(industrial_component)
        unrate_level_component = _weighted_optional_component(unrate_3m_avg, unrate_days_stale, invert=True)
        if unrate_level_component is not None:
            components.append(unrate_level_component)
        unrate_diff_component = _weighted_optional_component(unrate_diff_3m, unrate_days_stale, invert=True)
        if unrate_diff_component is not None:
            components.append(unrate_diff_component)
    elif inferred_country == "eurozone":
        industrial_production_yoy = _get_numeric_series(df, "industrial_production_yoy")
        unrate_3m_avg = _get_numeric_series(df, "unrate_3m_avg")
        unrate_diff_3m = _get_numeric_series(df, "unrate_diff_3m")
        sentiment_3m_avg = _get_numeric_series(df, "sentiment_3m_avg")
        industrial_days_stale = _get_numeric_series(df, "industrial_production_days_stale")
        unrate_days_stale = _get_numeric_series(df, "unrate_days_stale")
        sentiment_days_stale = _get_numeric_series(df, "sentiment_days_stale")
        industrial_component = _weighted_optional_component(industrial_production_yoy, industrial_days_stale)
        if industrial_component is not None:
            components.append(industrial_component)
        unrate_level_component = _weighted_optional_component(unrate_3m_avg, unrate_days_stale, invert=True)
        if unrate_level_component is not None:
            components.append(unrate_level_component)
        unrate_diff_component = _weighted_optional_component(unrate_diff_3m, unrate_days_stale, invert=True)
        if unrate_diff_component is not None:
            components.append(unrate_diff_component)
        sentiment_component = _weighted_optional_component(sentiment_3m_avg, sentiment_days_stale)
        if sentiment_component is not None:
            components.append(sentiment_component)
    else:
        unrate_3m_avg = _get_numeric_series(df, "unrate_3m_avg")
        unrate_diff_3m = _get_numeric_series(df, "unrate_diff_3m")
        if unrate_3m_avg.notna().any():
            components.append(-_zscore(unrate_3m_avg))
        if unrate_diff_3m.notna().any():
            components.append(-_zscore(unrate_diff_3m))
    if not components:
        return _empty_like(df)
    return _mean_score(components)


def compute_inflation_score(df: pd.DataFrame, country: str | None = None) -> pd.Series:
    """Compute a country-aware inflation score."""
    inferred_country = _infer_country(df, country)
    components: list[pd.Series] = []
    cpi_yoy = _get_numeric_series(df, "cpi_yoy")
    core_cpi_yoy = _get_numeric_series(df, "core_cpi_yoy")
    cpi_mom = _get_numeric_series(df, "cpi_mom_pct_change")
    core_cpi_mom = _get_numeric_series(df, "core_cpi_mom_pct_change")
    if cpi_yoy.notna().any():
        components.append(_zscore(cpi_yoy))
    core_days_stale = _get_numeric_series(df, "core_cpi_days_stale")
    core_component = _weighted_optional_component(core_cpi_yoy, core_days_stale)
    if core_component is not None:
        components.append(core_component)
    if not components and cpi_mom.notna().any():
        components.append(_zscore(cpi_mom))
    if not components and core_cpi_mom.notna().any():
        components.append(_zscore(core_cpi_mom))
    if not components:
        return _empty_like(df)
    return _mean_score(components)


def compute_liquidity_score(df: pd.DataFrame, country: str | None = None) -> pd.Series:
    """Compute a country-aware liquidity score."""
    inferred_country = _infer_country(df, country)
    components: list[pd.Series] = []
    policy_rate_level = _get_numeric_series(df, "policy_rate_level", "fedfunds_level")
    policy_rate_diff_3m = _get_numeric_series(df, "policy_rate_diff_3m", "fedfunds_diff_3m")
    yield_10y_level = _get_numeric_series(df, "yield_10y_level", "gs10_level")
    yield_10y_diff_3m = _get_numeric_series(df, "yield_10y_diff_3m", "gs10_diff_3m")
    m2_yoy = _get_numeric_series(df, "m2_yoy")
    m3_yoy = _get_numeric_series(df, "m3_yoy")
    m2_days_stale = _get_numeric_series(df, "m2_days_stale")
    m3_days_stale = _get_numeric_series(df, "m3_days_stale")

    if policy_rate_level.notna().any():
        components.append(-_zscore(policy_rate_level))
    if policy_rate_diff_3m.notna().any():
        components.append(-_zscore(policy_rate_diff_3m))
    if inferred_country == "china" and yield_10y_level.notna().any():
        components.append(-_zscore(yield_10y_level))
    if yield_10y_diff_3m.notna().any():
        components.append(-_zscore(yield_10y_diff_3m))
    m2_component = _weighted_optional_component(m2_yoy, m2_days_stale)
    if m2_component is not None:
        components.append(m2_component)
    m3_component = _weighted_optional_component(m3_yoy, m3_days_stale)
    if m3_component is not None:
        components.append(m3_component)

    if not components:
        return _empty_like(df)
    return _mean_score(components)


def compute_factor_scores(
    data: pd.DataFrame,
    weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Compute simple weighted scores by country using the latest indicator values."""
    required_columns = {"country", "indicator", "value"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    latest = (
        data.sort_values("date")
        .groupby(["country", "indicator"], as_index=False)
        .tail(1)
    )

    score_rows: list[dict[str, Any]] = []
    for country, country_frame in latest.groupby("country"):
        values = dict(zip(country_frame["indicator"], country_frame["value"], strict=False))
        factor_scores: dict[str, float] = {"country": country}
        total_score = 0.0
        for factor_name, indicator_weights in weights.items():
            factor_score = sum(
                values.get(indicator, 0.0) * weight
                for indicator, weight in indicator_weights.items()
            )
            factor_scores[factor_name] = factor_score
            total_score += factor_score
        factor_scores["composite"] = total_score
        score_rows.append(factor_scores)

    return pd.DataFrame(score_rows)
