"""Tests for global aggregation and investment clock logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.global_monitor import (
    build_country_status,
    build_global_regime_summary,
    classify_staleness,
    map_global_investment_clock,
)


def test_map_global_investment_clock_quadrants() -> None:
    """Growth and inflation scores should map into stable quadrant labels."""
    assert map_global_investment_clock(1.0, 2.0) == "overheating"
    assert map_global_investment_clock(2.0, 1.0) == "reflation"
    assert map_global_investment_clock(1.0, -0.5) == "disinflationary_growth"
    assert map_global_investment_clock(-0.5, 0.5) == "slowdown"


def test_build_global_regime_summary_aggregates_country_outputs(tmp_path: Path) -> None:
    """Global summary should combine country regime and valuation files."""
    processed = tmp_path / "processed"
    processed.mkdir()

    for country, growth, inflation, liquidity, valuation, regime in [
        ("us", 1.0, 0.5, 0.2, -0.2, "reflation"),
        ("china", 0.5, -0.1, 0.1, 0.4, "goldilocks"),
        ("eurozone", -0.2, 0.3, -0.1, -0.4, "stagflation"),
    ]:
        regime_frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01")],
                "country": [country],
                "growth_score": [growth],
                "inflation_score": [inflation],
                "liquidity_score": [liquidity],
                "regime": [regime],
                "liquidity_regime": ["neutral"],
            }
        )
        valuation_frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01")],
                "country": [country],
                "valuation_score": [valuation],
            }
        )
        regime_frame.to_csv(processed / f"{country}_macro_regimes.csv", index=False)
        valuation_frame.to_csv(processed / f"{country}_valuation_features.csv", index=False)

    result = build_global_regime_summary(processed_dir=str(processed))

    latest_available = result.loc[result["as_of_mode"] == "latest_available"].iloc[0]
    assert "global_growth_score" in result.columns
    assert "global_regime" in result.columns
    assert latest_available["us_regime"] == "reflation"
    assert latest_available["global_investment_clock"] in {
        "overheating",
        "reflation",
        "disinflationary_growth",
        "slowdown",
    }


def test_partial_coverage_marks_global_view_as_partial_and_renormalizes(tmp_path: Path) -> None:
    """Global summary should flag partial coverage and renormalize weights."""
    processed = tmp_path / "processed"
    processed.mkdir()

    regime_frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")],
            "country": ["us"],
            "growth_score": [1.0],
            "inflation_score": [-0.2],
            "liquidity_score": [0.4],
            "regime": ["goldilocks"],
            "liquidity_regime": ["neutral"],
        }
    )
    valuation_frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01")],
            "country": ["us"],
            "valuation_score": [0.3],
        }
    )
    regime_frame.to_csv(processed / "us_macro_regimes.csv", index=False)
    valuation_frame.to_csv(processed / "us_valuation_features.csv", index=False)

    result = build_global_regime_summary(processed_dir=str(processed))

    latest_available = result.loc[result["as_of_mode"] == "latest_available"].iloc[0]
    assert latest_available["coverage_ratio"] == 1 / 3
    assert latest_available["global_regime"] == "partial_view"
    assert latest_available["investment_clock"] == "partial_view"
    assert "us" in latest_available["countries_available"]
    assert "china" in latest_available["countries_missing"]
    assert latest_available["effective_weights"] == "us:1.00,china:0.00,eurozone:0.00"


def test_last_common_date_uses_shared_date_and_latest_available_uses_latest(tmp_path: Path) -> None:
    """The two global modes should produce different summary dates when countries are misaligned."""
    processed = tmp_path / "processed"
    processed.mkdir()

    us_regime = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "country": ["us", "us"],
            "growth_score": [0.1, 0.2],
            "inflation_score": [0.1, 0.2],
            "liquidity_score": [0.1, 0.2],
            "regime": ["reflation", "reflation"],
            "liquidity_regime": ["neutral", "neutral"],
        }
    )
    china_regime = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "country": ["china"],
            "growth_score": [0.3],
            "inflation_score": [-0.1],
            "liquidity_score": [0.0],
            "regime": ["goldilocks"],
            "liquidity_regime": ["neutral"],
        }
    )
    eurozone_regime = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "country": ["eurozone"],
            "growth_score": [-0.2],
            "inflation_score": [0.1],
            "liquidity_score": [-0.1],
            "regime": ["stagflation"],
            "liquidity_regime": ["tight"],
        }
    )
    for country, frame in [("us", us_regime), ("china", china_regime), ("eurozone", eurozone_regime)]:
        frame.to_csv(processed / f"{country}_macro_regimes.csv", index=False)
        pd.DataFrame({"date": frame["date"], "country": frame["country"], "valuation_score": [0.0] * len(frame)}).to_csv(
            processed / f"{country}_valuation_features.csv",
            index=False,
        )

    result = build_global_regime_summary(processed_dir=str(processed))
    latest_available = result.loc[result["as_of_mode"] == "latest_available"].iloc[0]
    last_common = result.loc[result["as_of_mode"] == "last_common_date"].iloc[0]

    assert pd.Timestamp(latest_available["summary_date"]) == pd.Timestamp("2025-02-01")
    assert pd.Timestamp(last_common["summary_date"]) == pd.Timestamp("2025-01-01")


def test_staleness_and_country_ready_vs_globally_usable(tmp_path: Path) -> None:
    """Latest-available mode should keep stale but locally ready countries usable."""
    processed = tmp_path / "processed"
    processed.mkdir()

    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "country": ["us", "us"],
            "growth_score": [0.1, 0.2],
            "inflation_score": [0.1, 0.2],
            "liquidity_score": [0.1, 0.2],
            "regime": ["reflation", "reflation"],
            "liquidity_regime": ["neutral", "neutral"],
        }
    ).to_csv(processed / "us_macro_regimes.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "country": ["china"],
            "growth_score": [0.1],
            "inflation_score": [0.1],
            "liquidity_score": [0.1],
            "regime": ["reflation"],
            "liquidity_regime": ["neutral"],
        }
    ).to_csv(processed / "china_macro_regimes.csv", index=False)
    for country in ["us", "china"]:
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-02-01" if country == "us" else "2024-01-01"]),
                "country": [country],
                "valuation_score": [0.0],
            }
        ).to_csv(processed / f"{country}_valuation_features.csv", index=False)

    status = build_country_status(processed_dir=str(processed), mode="latest_available")
    us_row = status.loc[status["country"] == "us"].iloc[0]
    china_row = status.loc[status["country"] == "china"].iloc[0]

    assert bool(us_row["country_ready"]) is True
    assert bool(us_row["globally_usable_latest"]) is True
    assert bool(china_row["country_ready"]) is True
    assert bool(china_row["globally_usable_latest"]) is True
    assert china_row["staleness_status"] in {"stale", "very_stale"}


def test_classify_staleness_buckets() -> None:
    """Days stale should map into the expected staleness labels."""
    assert classify_staleness(30) == "fresh"
    assert classify_staleness(120) == "stale"
    assert classify_staleness(240) == "very_stale"
