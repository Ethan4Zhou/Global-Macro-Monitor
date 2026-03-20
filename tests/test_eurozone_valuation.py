"""Tests for Eurozone valuation loading and scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.classifier import classify_country_macro_regime
from app.valuation.eurozone_models import compute_eurozone_valuation_score
from app.valuation.features import (
    build_country_valuation_features_frame,
    inspect_eurozone_valuation_inputs,
)


def test_eurozone_valuation_reads_normalized_data_before_manual(tmp_path: Path) -> None:
    """Eurozone valuation inspection should prefer normalized API data."""
    api_dir = tmp_path / "api"
    normalized = api_dir / "eurozone" / "normalized"
    normalized.mkdir(parents=True)
    manual_dir = tmp_path / "manual" / "eurozone"
    manual_dir.mkdir(parents=True)
    for series_id, value in {"cpi": 2.5, "policy_rate": 3.0, "yield_10y": 2.2}.items():
        pd.DataFrame(
            {
                "date": ["2026-02-01"],
                "value": [value],
                "series_id": [series_id],
                "country": ["eurozone"],
                "source": ["eurostat" if series_id == "cpi" else "ecb"],
                "frequency": ["monthly"],
                "release_date": ["2026-02-15"],
                "ingested_at": ["2026-03-01"],
            }
        ).to_csv(normalized / f"{series_id}.csv", index=False)
        pd.DataFrame({"date": ["2026-02-01"], "value": [99.0], "series_id": [series_id]}).to_csv(
            manual_dir / f"{series_id}.csv",
            index=False,
        )

    diagnostics = inspect_eurozone_valuation_inputs(
        api_dir=str(api_dir),
        manual_dir=str(tmp_path / "manual"),
    )
    assert diagnostics["valuation_ready"] is True
    assert diagnostics["actual_sources_found"] == ["normalized_api"]


def test_eurozone_valuation_feature_generation_from_normalized_data(tmp_path: Path) -> None:
    """Eurozone valuation features should use normalized proxy inputs."""
    api_dir = tmp_path / "api"
    normalized = api_dir / "eurozone" / "normalized"
    normalized.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01"],
            "value": [14.0, 13.5],
            "series_id": ["equity_pe_proxy", "equity_pe_proxy"],
            "country": ["eurozone", "eurozone"],
            "source": ["manual", "manual"],
            "frequency": ["monthly", "monthly"],
            "release_date": ["2026-01-15", "2026-02-15"],
            "ingested_at": ["2026-03-01", "2026-03-01"],
        }
    ).to_csv(normalized / "equity_pe_proxy.csv", index=False)
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "country": ["eurozone", "eurozone"],
            "cpi_yoy": [2.5, 2.4],
            "policy_rate_level": [3.0, 2.9],
            "yield_10y_level": [2.2, 2.1],
        }
    )
    valuation = build_country_valuation_features_frame(
        macro_features=macro,
        country="eurozone",
        api_dir=str(api_dir),
    )
    assert "equity_pe_proxy" in valuation.columns
    assert "real_yield_proxy" in valuation.columns
    assert pd.notna(compute_eurozone_valuation_score(valuation).iloc[-1])


def test_eurozone_minimum_regime_classification() -> None:
    """Eurozone should classify with the minimum configured input set."""
    features = pd.DataFrame(
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
    classified = classify_country_macro_regime(features, country="eurozone")
    assert classified["regime"].iloc[-1] != "unknown"
