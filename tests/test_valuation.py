"""Tests for valuation scoring and asset mapping."""

from __future__ import annotations

import pandas as pd

from app.regime.allocation import map_asset_preferences
from app.valuation.models import compute_valuation_score, label_valuation_regime


def test_compute_valuation_score_prefers_cheaper_conditions() -> None:
    """Valuation score should be higher for cheaper market conditions."""
    frame = pd.DataFrame(
        {
            "buffett_indicator": [2.0, 1.0],
            "real_yield": [2.5, 1.0],
            "term_spread": [-0.5, 1.0],
            "equity_risk_proxy": [0.03, 0.08],
            "credit_spread_proxy": [2.0, 1.0],
        }
    )

    scores = compute_valuation_score(frame)

    assert scores.iloc[1] > scores.iloc[0]
    assert label_valuation_regime(scores.iloc[1]) in {"cheap", "fair"}


def test_map_asset_preferences_handles_handcrafted_scenarios() -> None:
    """Asset mapping should react sensibly to regime and valuation combinations."""
    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "growth_score": [1.0, -1.0],
            "inflation_score": [-0.5, 1.0],
            "liquidity_score": [0.8, -1.2],
            "regime": ["goldilocks", "stagflation"],
            "liquidity_regime": ["easy", "tight"],
        }
    )
    valuation_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "valuation_score": [1.0, -1.0],
            "valuation_regime": ["cheap", "expensive"],
        }
    )

    result = map_asset_preferences(regime_df, valuation_df)

    assert result.loc[0, "equities"] == "bullish"
    assert result.loc[0, "dollar"] == "cautious"
    assert result.loc[1, "gold"] == "bullish"
    assert result.loc[1, "duration"] == "cautious"


def test_map_asset_preferences_still_outputs_local_assets_when_valuation_missing() -> None:
    """Missing valuation should reduce confidence, not suppress local allocation output."""
    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "country": ["china"],
            "growth_score": [0.4],
            "inflation_score": [-0.2],
            "liquidity_score": [0.1],
            "regime": ["goldilocks"],
            "liquidity_regime": ["neutral"],
        }
    )

    result = map_asset_preferences(regime_df, pd.DataFrame())

    assert len(result) == 1
    assert result.loc[0, "equities"] in {"bullish", "neutral", "cautious"}
    assert result.loc[0, "allocation_confidence"] == "medium"
    assert result.loc[0, "valuation_regime"] == "unknown"


def test_map_asset_preferences_does_not_auto_overweight_gold_in_plain_slowdown() -> None:
    """A slowdown without inflation stress should not automatically make gold bullish."""
    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "country": ["china"],
            "growth_score": [-0.4],
            "inflation_score": [-0.3],
            "liquidity_score": [0.0],
            "regime": ["slowdown"],
            "liquidity_regime": ["neutral"],
        }
    )
    valuation_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "country": ["china"],
            "valuation_score": [1.0],
            "valuation_regime": ["cheap"],
        }
    )

    result = map_asset_preferences(regime_df, valuation_df)

    assert result.loc[0, "gold_score"] == -1.0
    assert result.loc[0, "gold"] == "cautious"
