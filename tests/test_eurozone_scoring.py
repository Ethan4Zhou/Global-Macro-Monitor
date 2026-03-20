"""Tests for Eurozone scoring and freshness-aware enrichment."""

from __future__ import annotations

import pandas as pd

from app.factors.scoring import compute_growth_score, compute_liquidity_score


def test_eurozone_minimum_scoring_works_without_optional_enrichment() -> None:
    """Eurozone scores should work with the minimum configured inputs."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-01", periods=6, freq="MS"),
            "country": ["eurozone"] * 6,
            "pmi_level": [48.5, 48.7, 48.9, 49.1, 49.3, 49.6],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.7],
            "pmi_3m_avg": [None, None, 48.7, 48.9, 49.1, 49.3],
            "cpi_yoy": [3.0, 2.9, 2.7, 2.5, 2.3, 2.1],
            "policy_rate_level": [4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
            "policy_rate_diff_3m": [None, None, None, -0.3, -0.3, -0.3],
            "yield_10y_level": [2.7, 2.6, 2.5, 2.4, 2.3, 2.2],
            "yield_10y_diff_3m": [None, None, None, -0.3, -0.3, -0.3],
        }
    )
    assert pd.notna(compute_growth_score(frame, country="eurozone").iloc[-1])
    assert pd.notna(compute_liquidity_score(frame, country="eurozone").iloc[-1])


def test_eurozone_very_stale_enrichment_is_ignored() -> None:
    """Very stale Eurozone enrichment inputs should not affect effective scoring."""
    base = pd.DataFrame(
        {
            "date": pd.date_range("2025-10-01", periods=6, freq="MS"),
            "country": ["eurozone"] * 6,
            "pmi_level": [48.5, 48.7, 48.9, 49.1, 49.3, 49.6],
            "pmi_diff_3m": [None, None, None, 0.6, 0.6, 0.7],
            "pmi_3m_avg": [None, None, 48.7, 48.9, 49.1, 49.3],
            "cpi_yoy": [3.0, 2.9, 2.7, 2.5, 2.3, 2.1],
            "policy_rate_level": [4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
            "policy_rate_diff_3m": [None, None, None, -0.3, -0.3, -0.3],
            "yield_10y_level": [2.7, 2.6, 2.5, 2.4, 2.3, 2.2],
            "yield_10y_diff_3m": [None, None, None, -0.3, -0.3, -0.3],
        }
    )
    enriched = base.assign(
        industrial_production_yoy=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        industrial_production_days_stale=[250] * 6,
        m3_yoy=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        m3_days_stale=[250] * 6,
        sentiment_3m_avg=[99.0, 99.1, 99.2, 99.3, 99.4, 99.5],
        sentiment_days_stale=[250] * 6,
    )
    assert compute_growth_score(base, country="eurozone").equals(
        compute_growth_score(enriched, country="eurozone")
    )
    assert compute_liquidity_score(base, country="eurozone").equals(
        compute_liquidity_score(enriched, country="eurozone")
    )
