"""Tests for China minimum-input scoring and classification."""

from __future__ import annotations

import pandas as pd

from app.factors.features import align_monthly_panel, build_country_macro_features_frame
from app.factors.scoring import compute_growth_score, compute_liquidity_score
from app.regime.classifier import classify_country_macro_regime


def test_china_pmi_only_growth_scoring_produces_values() -> None:
    """China growth should score from PMI without requiring unemployment."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="MS"),
            "country": ["china"] * 6,
            "pmi_level": [49.0, 49.2, 49.4, 49.6, 49.8, 50.0],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.6],
            "pmi_3m_avg": [None, None, 49.2, 49.4, 49.6, 49.8],
        }
    )
    score = compute_growth_score(frame, country="china")
    assert score.notna().any()
    assert pd.notna(score.iloc[-1])


def test_china_monthly_asof_alignment_with_staggered_latest_dates() -> None:
    """Monthly feature alignment should carry forward latest available values."""
    panel = align_monthly_panel(
        {
            "pmi": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                    "pmi": [49.3, 49.0],
                }
            ),
            "policy_rate": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
                    "policy_rate": [1.54, 1.40, 1.49],
                }
            ),
            "yield_10y": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
                    "yield_10y": [1.81, 1.77, 1.82],
                }
            ),
            "cpi": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                    "cpi": [0.2, 1.3],
                }
            ),
        }
    )
    features = build_country_macro_features_frame(panel=panel, country="china")
    latest = features.iloc[-1]
    assert str(pd.Timestamp(latest["date"]).date()) == "2026-03-01"
    assert latest["pmi_level"] == 49.0
    assert latest["cpi_yoy"] == 1.3


def test_china_minimum_input_regime_classification() -> None:
    """China should classify a regime with the minimum four required inputs."""
    features = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-01", periods=6, freq="MS"),
            "country": ["china"] * 6,
            "pmi_level": [49.0, 49.2, 49.4, 49.6, 49.8, 50.0],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.6],
            "pmi_3m_avg": [None, None, 49.2, 49.4, 49.6, 49.8],
            "cpi_yoy": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            "policy_rate_level": [1.8, 1.75, 1.7, 1.65, 1.6, 1.55],
            "policy_rate_diff_3m": [None, None, None, -0.15, -0.15, -0.15],
            "yield_10y_level": [2.0, 1.98, 1.95, 1.92, 1.9, 1.88],
            "yield_10y_diff_3m": [None, None, None, -0.08, -0.08, -0.07],
        }
    )
    classified = classify_country_macro_regime(features, country="china")
    latest = classified.iloc[-1]
    assert pd.notna(latest["growth_score"])
    assert pd.notna(latest["inflation_score"])
    assert pd.notna(latest["liquidity_score"])
    assert latest["regime"] != "unknown"


def test_china_enriched_scoring_uses_industrial_and_m2_when_available() -> None:
    """China scoring should become richer when optional series are present."""
    features = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-01", periods=6, freq="MS"),
            "country": ["china"] * 6,
            "pmi_level": [49.0, 49.2, 49.4, 49.6, 49.8, 50.0],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.6],
            "pmi_3m_avg": [None, None, 49.2, 49.4, 49.6, 49.8],
            "industrial_production_yoy": [4.8, 5.0, 5.1, 5.3, 5.4, 5.6],
            "unrate_3m_avg": [5.2, 5.1, 5.1, 5.0, 4.9, 4.8],
            "unrate_diff_3m": [None, None, None, -0.2, -0.2, -0.3],
            "cpi_yoy": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            "core_cpi_yoy": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "policy_rate_level": [1.8, 1.75, 1.7, 1.65, 1.6, 1.55],
            "policy_rate_diff_3m": [None, None, None, -0.15, -0.15, -0.15],
            "yield_10y_level": [2.0, 1.98, 1.95, 1.92, 1.9, 1.88],
            "yield_10y_diff_3m": [None, None, None, -0.08, -0.08, -0.07],
            "m2_yoy": [7.0, 7.1, 7.2, 7.4, 7.6, 7.8],
        }
    )
    growth = compute_growth_score(features, country="china")
    liquidity = compute_liquidity_score(features, country="china")
    assert pd.notna(growth.iloc[-1])
    assert pd.notna(liquidity.iloc[-1])


def test_china_stale_enrichment_is_downweighted_but_not_blocking() -> None:
    """Stale optional signals should still allow valid minimum-input scoring."""
    features = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-01", periods=6, freq="MS"),
            "country": ["china"] * 6,
            "pmi_level": [49.0, 49.2, 49.4, 49.6, 49.8, 50.0],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.6],
            "pmi_3m_avg": [None, None, 49.2, 49.4, 49.6, 49.8],
            "cpi_yoy": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            "policy_rate_level": [1.8, 1.75, 1.7, 1.65, 1.6, 1.55],
            "policy_rate_diff_3m": [None, None, None, -0.15, -0.15, -0.15],
            "yield_10y_level": [2.0, 1.98, 1.95, 1.92, 1.9, 1.88],
            "yield_10y_diff_3m": [None, None, None, -0.08, -0.08, -0.07],
            "industrial_production_yoy": [4.8, 5.0, 5.1, 5.3, 5.4, 5.6],
            "industrial_production_days_stale": [120, 120, 120, 120, 120, 120],
            "m2_yoy": [7.0, 7.1, 7.2, 7.4, 7.6, 7.8],
            "m2_days_stale": [120, 120, 120, 120, 120, 120],
        }
    )

    growth = compute_growth_score(features, country="china")
    liquidity = compute_liquidity_score(features, country="china")

    assert pd.notna(growth.iloc[-1])
    assert pd.notna(liquidity.iloc[-1])


def test_china_very_stale_enrichment_is_ignored() -> None:
    """Very stale optional signals should be excluded from the effective score."""
    base = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-01", periods=6, freq="MS"),
            "country": ["china"] * 6,
            "pmi_level": [49.0, 49.2, 49.4, 49.6, 49.8, 50.0],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.6],
            "pmi_3m_avg": [None, None, 49.2, 49.4, 49.6, 49.8],
            "cpi_yoy": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            "policy_rate_level": [1.8, 1.75, 1.7, 1.65, 1.6, 1.55],
            "policy_rate_diff_3m": [None, None, None, -0.15, -0.15, -0.15],
            "yield_10y_level": [2.0, 1.98, 1.95, 1.92, 1.9, 1.88],
            "yield_10y_diff_3m": [None, None, None, -0.08, -0.08, -0.07],
        }
    )
    enriched = base.assign(
        industrial_production_yoy=[4.8, 5.0, 5.1, 5.3, 5.4, 5.6],
        industrial_production_days_stale=[250, 250, 250, 250, 250, 250],
        m2_yoy=[7.0, 7.1, 7.2, 7.4, 7.6, 7.8],
        m2_days_stale=[250, 250, 250, 250, 250, 250],
    )

    growth_base = compute_growth_score(base, country="china")
    growth_enriched = compute_growth_score(enriched, country="china")
    liquidity_base = compute_liquidity_score(base, country="china")
    liquidity_enriched = compute_liquidity_score(enriched, country="china")

    assert growth_base.equals(growth_enriched)
    assert liquidity_base.equals(liquidity_enriched)
