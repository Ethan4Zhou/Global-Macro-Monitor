"""Tests for the global allocation mapping layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.global_allocation import build_global_allocation_map


def _write_country_files(
    processed: Path,
    country: str,
    regime_date: str,
    regime: str,
    liquidity_regime: str,
    growth_score: float,
    inflation_score: float,
    liquidity_score: float,
    valuation_score: float | None,
) -> None:
    """Write minimal processed inputs for one country."""
    pd.DataFrame(
        {
            "date": [pd.Timestamp(regime_date)],
            "country": [country],
            "growth_score": [growth_score],
            "inflation_score": [inflation_score],
            "liquidity_score": [liquidity_score],
            "regime": [regime],
            "liquidity_regime": [liquidity_regime],
        }
    ).to_csv(processed / f"{country}_macro_regimes.csv", index=False)

    valuation_frame = pd.DataFrame({"date": [pd.Timestamp(regime_date)], "country": [country]})
    if valuation_score is not None:
        valuation_frame["valuation_score"] = [valuation_score]
        valuation_frame["valuation_regime"] = ["fair"]
    valuation_frame.to_csv(processed / f"{country}_valuation_features.csv", index=False)


def test_global_allocation_maps_reflation_into_commodities_preference(tmp_path: Path) -> None:
    """A reflation-style global setup should lean into commodities."""
    processed = tmp_path / "processed"
    processed.mkdir()

    _write_country_files(processed, "us", "2025-01-01", "reflation", "neutral", 1.2, 0.7, 0.0, 0.0)
    _write_country_files(processed, "china", "2025-01-01", "reflation", "easy", 0.8, 0.4, 0.6, 0.1)
    _write_country_files(processed, "eurozone", "2025-01-01", "goldilocks", "neutral", 0.3, -0.1, 0.0, -0.1)

    allocation = build_global_allocation_map(processed_dir=str(processed))

    latest_available = allocation.loc[allocation["as_of_mode"] == "latest_available"]
    commodities = latest_available.loc[latest_available["asset"] == "commodities"].iloc[0]
    assert commodities["preference"] == "bullish"
    assert "Reflation conditions support commodities." in commodities["reason"]
    assert "Latest available compares each region on its own latest valid date:" in commodities["mode_context"]


def test_global_allocation_confidence_drops_with_stale_country_data(tmp_path: Path) -> None:
    """Very stale country data should push country and global confidence lower."""
    processed = tmp_path / "processed"
    processed.mkdir()

    _write_country_files(processed, "us", "2025-06-01", "goldilocks", "neutral", 0.8, -0.2, 0.1, 0.0)
    _write_country_files(processed, "china", "2024-01-01", "goldilocks", "neutral", 0.5, -0.1, 0.0, 0.0)
    _write_country_files(processed, "eurozone", "2025-06-01", "slowdown", "tight", -0.4, -0.2, -0.7, 0.2)

    allocation = build_global_allocation_map(processed_dir=str(processed))

    latest_available = allocation.loc[allocation["as_of_mode"] == "latest_available"]
    china_equities = latest_available.loc[
        latest_available["asset"] == "china_equities"
    ].iloc[0]
    global_equities = latest_available.loc[
        latest_available["asset"] == "global_equities"
    ].iloc[0]
    assert china_equities["confidence"] == "low"
    assert global_equities["confidence"] == "low"
    assert "very stale local data" in china_equities["reason"]
    assert "very stale country data" in global_equities["reason"]


def test_global_allocation_last_common_date_reason_mentions_shared_date(tmp_path: Path) -> None:
    """Last common date reasons should reference the shared evaluation date."""
    processed = tmp_path / "processed"
    processed.mkdir()

    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "country": ["us", "us"],
            "growth_score": [0.5, 0.8],
            "inflation_score": [-0.1, -0.2],
            "liquidity_score": [0.0, 0.1],
            "regime": ["goldilocks", "goldilocks"],
            "liquidity_regime": ["neutral", "neutral"],
        }
    ).to_csv(processed / "us_macro_regimes.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "country": ["us", "us"],
            "valuation_score": [0.0, 0.0],
        }
    ).to_csv(processed / "us_valuation_features.csv", index=False)
    _write_country_files(processed, "china", "2025-01-01", "goldilocks", "neutral", 0.5, -0.1, 0.0, 0.0)
    _write_country_files(processed, "eurozone", "2025-01-01", "slowdown", "tight", -0.4, -0.2, -0.7, 0.2)

    allocation = build_global_allocation_map(processed_dir=str(processed))

    global_equities = allocation.loc[
        (allocation["as_of_mode"] == "last_common_date")
        & (allocation["asset"] == "global_equities")
    ].iloc[0]
    assert "Last common date compares all contributing regions on 2025-01-01" in global_equities["mode_context"]


def test_global_allocation_missing_valuation_downgrades_confidence(tmp_path: Path) -> None:
    """Missing valuation inputs should reduce confidence by at least one level."""
    processed = tmp_path / "processed"
    processed.mkdir()

    _write_country_files(processed, "us", "2025-01-01", "goldilocks", "neutral", 0.8, -0.3, 0.0, None)
    _write_country_files(processed, "china", "2025-01-01", "goldilocks", "easy", 0.7, -0.2, 0.5, None)
    _write_country_files(processed, "eurozone", "2025-01-01", "slowdown", "tight", -0.2, -0.1, -0.6, None)

    allocation = build_global_allocation_map(processed_dir=str(processed))

    global_equities = allocation.loc[
        (allocation["as_of_mode"] == "latest_available")
        & (allocation["asset"] == "global_equities")
    ].iloc[0]
    assert global_equities["confidence"] == "medium"
    assert "macro-only view because valuation input is missing" in global_equities["reason"]


def test_global_allocation_handles_missing_valuation_inputs_gracefully(tmp_path: Path) -> None:
    """Missing valuation data should still allow a preference with lower confidence."""
    processed = tmp_path / "processed"
    processed.mkdir()

    _write_country_files(processed, "us", "2025-01-01", "goldilocks", "neutral", 0.8, -0.3, 0.0, None)
    _write_country_files(processed, "china", "2025-01-01", "goldilocks", "easy", 0.7, -0.2, 0.5, 0.2)
    _write_country_files(processed, "eurozone", "2025-01-01", "slowdown", "tight", -0.2, -0.1, -0.6, 0.1)

    allocation = build_global_allocation_map(processed_dir=str(processed))

    us_equities = allocation.loc[
        (allocation["as_of_mode"] == "latest_available")
        & (allocation["asset"] == "us_equities")
    ].iloc[0]
    assert us_equities["preference"] in {"bullish", "neutral", "cautious"}
    assert us_equities["confidence"] == "medium"
    assert "Valuation is missing, so this is a macro-only view." in us_equities["reason"]


def test_global_duration_uses_us_local_duration_signal(tmp_path: Path) -> None:
    """Global duration should follow the US local duration lens for consistency."""
    processed = tmp_path / "processed"
    processed.mkdir()

    _write_country_files(processed, "us", "2025-01-01", "goldilocks", "neutral", 0.8, -0.3, 0.0, 0.0)
    _write_country_files(processed, "china", "2025-01-01", "slowdown", "easy", -0.4, -0.2, 0.5, 0.2)
    _write_country_files(processed, "eurozone", "2025-01-01", "slowdown", "tight", -0.5, -0.2, -0.7, 0.1)

    pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")],
            "country": ["us"],
            "regime": ["goldilocks"],
            "liquidity_regime": ["neutral"],
            "valuation_score": [0.0],
            "valuation_regime": ["fair"],
            "equities_score": [2.0],
            "equities": ["bullish"],
            "duration_score": [0.0],
            "duration": ["neutral"],
            "gold_score": [-1.0],
            "gold": ["cautious"],
            "dollar_score": [-1.0],
            "dollar": ["cautious"],
            "allocation_confidence": ["high"],
            "allocation_note": ["Macro and valuation inputs are both available."],
        }
    ).to_csv(processed / "us_asset_preferences.csv", index=False)

    allocation = build_global_allocation_map(processed_dir=str(processed))

    global_duration = allocation.loc[
        (allocation["as_of_mode"] == "latest_available") & (allocation["asset"] == "duration")
    ].iloc[0]
    assert global_duration["score"] == 0.0
    assert global_duration["preference"] == "neutral"
    assert "dollar duration remains neutral" in global_duration["reason"]
