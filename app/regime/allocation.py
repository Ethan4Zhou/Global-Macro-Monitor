"""Asset preference mapping from macro and valuation states."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.valuation.models import compute_valuation_score, label_valuation_regime


def _tag_preference(score: float) -> str:
    """Map a numeric preference score to a readable tag."""
    if score >= 2:
        return "bullish"
    if score <= -1:
        return "cautious"
    return "neutral"


def _equities_score(regime: str, liquidity_regime: str, valuation_regime: str) -> float:
    """Compute the equities preference score."""
    score = {
        "goldilocks": 2.0,
        "reflation": 1.0,
        "slowdown": -1.0,
        "stagflation": -2.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {"easy": 1.0, "neutral": 0.0, "tight": -1.0, "unknown": 0.0}.get(
        liquidity_regime,
        0.0,
    )
    score += {"cheap": 1.0, "fair": 0.0, "expensive": -1.0, "unknown": 0.0}.get(
        valuation_regime,
        0.0,
    )
    return score


def _duration_score(regime: str, liquidity_regime: str, valuation_regime: str) -> float:
    """Compute the duration preference score."""
    score = {
        "slowdown": 2.0,
        "goldilocks": 0.0,
        "reflation": -1.0,
        "stagflation": -2.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {"tight": 0.5, "neutral": 0.0, "easy": -0.5, "unknown": 0.0}.get(
        liquidity_regime,
        0.0,
    )
    score += {"expensive": 0.5, "fair": 0.0, "cheap": -1.0, "unknown": 0.0}.get(
        valuation_regime,
        0.0,
    )
    return score


def _gold_score(regime: str, liquidity_regime: str, valuation_regime: str) -> float:
    """Compute the gold preference score.

    Gold is treated as a more globally priced hedge, so a plain slowdown
    without inflation stress should not automatically create a bullish gold
    signal for one region.
    """
    score = {
        "stagflation": 2.0,
        "slowdown": -0.5,
        "reflation": 0.0,
        "goldilocks": -1.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {"easy": 0.5, "neutral": 0.0, "tight": -0.5, "unknown": 0.0}.get(
        liquidity_regime,
        0.0,
    )
    score += {"expensive": 1.0, "fair": 0.0, "cheap": -0.5, "unknown": 0.0}.get(
        valuation_regime,
        0.0,
    )
    return score


def _dollar_score(regime: str, liquidity_regime: str, valuation_regime: str) -> float:
    """Compute the dollar preference score."""
    score = {
        "slowdown": 1.0,
        "stagflation": 1.0,
        "reflation": 0.0,
        "goldilocks": -1.0,
        "unknown": 0.0,
    }.get(regime, 0.0)
    score += {"tight": 2.0, "neutral": 0.0, "easy": -1.0, "unknown": 0.0}.get(
        liquidity_regime,
        0.0,
    )
    score += {"expensive": 1.0, "fair": 0.0, "cheap": -1.0, "unknown": 0.0}.get(
        valuation_regime,
        0.0,
    )
    return score


def map_asset_preferences(
    regime_df: pd.DataFrame,
    valuation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map macro regime and valuation data into asset preference outputs."""
    valuation = valuation_df.copy()
    if valuation.empty:
        merge_keys = ["date", "country"] if "country" in regime_df.columns else ["date"]
        valuation = regime_df.loc[:, merge_keys].drop_duplicates().copy()
        valuation["valuation_score"] = float("nan")
        valuation["valuation_regime"] = "unknown"
    if "valuation_score" not in valuation.columns:
        valuation["valuation_score"] = compute_valuation_score(valuation)
    if "valuation_regime" not in valuation.columns:
        valuation["valuation_regime"] = valuation["valuation_score"].apply(label_valuation_regime)
    valuation["valuation_regime"] = valuation["valuation_regime"].fillna("unknown")

    merge_keys = ["date", "country"] if "country" in valuation.columns else ["date"]
    merged = regime_df.merge(
        valuation.loc[:, merge_keys + ["valuation_score", "valuation_regime"]],
        on=merge_keys,
        how="left",
    )

    merged["equities_score"] = merged.apply(
        lambda row: _equities_score(
            str(row.get("regime", "unknown")),
            str(row.get("liquidity_regime", "unknown")),
            str(row.get("valuation_regime", "unknown")),
        ),
        axis=1,
    )
    merged["duration_score"] = merged.apply(
        lambda row: _duration_score(
            str(row.get("regime", "unknown")),
            str(row.get("liquidity_regime", "unknown")),
            str(row.get("valuation_regime", "unknown")),
        ),
        axis=1,
    )
    merged["gold_score"] = merged.apply(
        lambda row: _gold_score(
            str(row.get("regime", "unknown")),
            str(row.get("liquidity_regime", "unknown")),
            str(row.get("valuation_regime", "unknown")),
        ),
        axis=1,
    )
    merged["dollar_score"] = merged.apply(
        lambda row: _dollar_score(
            str(row.get("regime", "unknown")),
            str(row.get("liquidity_regime", "unknown")),
            str(row.get("valuation_regime", "unknown")),
        ),
        axis=1,
    )

    merged["equities"] = merged["equities_score"].apply(_tag_preference)
    merged["duration"] = merged["duration_score"].apply(_tag_preference)
    merged["gold"] = merged["gold_score"].apply(_tag_preference)
    merged["dollar"] = merged["dollar_score"].apply(_tag_preference)
    merged["allocation_confidence"] = merged["valuation_regime"].apply(
        lambda value: "medium" if str(value) == "unknown" else "high"
    )
    merged["allocation_note"] = merged["valuation_regime"].apply(
        lambda value: (
            "Valuation is missing, so local allocation is still produced with reduced confidence."
            if str(value) == "unknown"
            else "Macro and valuation inputs are both available."
        )
    )

    output_columns = [
        "date",
        "country",
        "regime",
        "liquidity_regime",
        "valuation_score",
        "valuation_regime",
        "equities_score",
        "equities",
        "duration_score",
        "duration",
        "gold_score",
        "gold",
        "dollar_score",
        "dollar",
        "allocation_confidence",
        "allocation_note",
    ]
    available = [column for column in output_columns if column in merged.columns]
    sort_columns = [column for column in ["country", "date"] if column in merged.columns]
    return merged.loc[:, available].sort_values(sort_columns).reset_index(drop=True)


def save_country_asset_preferences(
    frame: pd.DataFrame,
    country: str,
    output_path: str | None = None,
) -> Path:
    """Save the asset preference table to CSV."""
    destination = Path(output_path or f"data/processed/{country}_asset_preferences.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def save_us_asset_preferences(
    frame: pd.DataFrame,
    output_path: str = "data/processed/us_asset_preferences.csv",
) -> Path:
    """Backward-compatible wrapper for saving US asset preferences."""
    return save_country_asset_preferences(frame, country="us", output_path=output_path)
