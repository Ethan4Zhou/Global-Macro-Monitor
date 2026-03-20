"""Tests for China valuation proxies and scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.classifier import classify_country_macro_regime
from app.valuation.china_models import compute_china_valuation_score, label_china_valuation_regime
from app.valuation.features import build_country_valuation_features_frame, inspect_china_valuation_inputs


def test_build_china_valuation_features_frame_uses_proxy_columns(tmp_path: Path) -> None:
    """China valuation features should expose proxy columns even with partial inputs."""
    api_dir = tmp_path / "api"
    normalized = api_dir / "china" / "normalized"
    normalized.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01"],
            "value": [12.0, 12.5],
            "series_id": ["hs300_pe_proxy", "hs300_pe_proxy"],
            "country": ["china", "china"],
            "source": ["china_akshare", "china_akshare"],
            "frequency": ["monthly", "monthly"],
            "release_date": ["2026-01-15", "2026-02-15"],
            "ingested_at": ["2026-03-01", "2026-03-01"],
        }
    ).to_csv(normalized / "hs300_pe_proxy.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-01"],
            "value": [1.4, 1.5],
            "series_id": ["hs300_pb_proxy", "hs300_pb_proxy"],
            "country": ["china", "china"],
            "source": ["china_akshare", "china_akshare"],
            "frequency": ["monthly", "monthly"],
            "release_date": ["2026-01-15", "2026-02-15"],
            "ingested_at": ["2026-03-01", "2026-03-01"],
        }
    ).to_csv(normalized / "hs300_pb_proxy.csv", index=False)

    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "country": ["china", "china"],
            "cpi_yoy": [0.2, 0.4],
            "policy_rate_level": [1.5, 1.4],
            "yield_10y_level": [1.8, 1.7],
        }
    )

    valuation = build_country_valuation_features_frame(
        macro_features=macro,
        country="china",
        api_dir=str(api_dir),
    )

    assert "hs300_pe_proxy" in valuation.columns
    assert "hs300_pb_proxy" in valuation.columns
    assert "real_yield_proxy" in valuation.columns
    assert valuation["valuation_method"].iloc[-1] == "proxy_based"


def test_compute_china_valuation_score_returns_non_nan_when_proxies_exist() -> None:
    """China valuation score should work with proxy-based PE, PB, real yield, and term spread."""
    frame = pd.DataFrame(
        {
            "hs300_pe_proxy": [14.0, 12.0, 10.0],
            "hs300_pb_proxy": [1.8, 1.5, 1.2],
            "real_yield_proxy": [2.0, 1.0, 0.5],
            "term_spread": [0.1, 0.4, 0.8],
        }
    )
    score = compute_china_valuation_score(frame)
    assert pd.notna(score.iloc[-1])
    assert label_china_valuation_regime(score.iloc[-1]) in {"cheap", "fair", "expensive"}


def test_china_regime_still_works_when_valuation_is_missing() -> None:
    """China regime classification should remain independent from valuation availability."""
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

    assert classified["regime"].iloc[-1] != "unknown"


def test_china_valuation_inspection_reads_normalized_path_first(tmp_path: Path) -> None:
    """China valuation diagnostics should prefer normalized API files over manual fallback."""
    api_dir = tmp_path / "api"
    normalized = api_dir / "china" / "normalized"
    normalized.mkdir(parents=True)
    manual_dir = tmp_path / "manual" / "china"
    manual_dir.mkdir(parents=True)

    for series_id, value in {"cpi": 0.4, "policy_rate": 1.5, "yield_10y": 1.8}.items():
        pd.DataFrame(
            {
                "date": ["2026-02-01"],
                "value": [value],
                "series_id": [series_id],
                "country": ["china"],
                "source": ["china_akshare"],
                "frequency": ["monthly"],
                "release_date": ["2026-02-15"],
                "ingested_at": ["2026-03-01"],
            }
        ).to_csv(normalized / f"{series_id}.csv", index=False)
        pd.DataFrame({"date": ["2026-02-01"], "value": [99.0], "series_id": [series_id]}).to_csv(
            manual_dir / f"{series_id}.csv",
            index=False,
        )

    diagnostics = inspect_china_valuation_inputs(api_dir=str(api_dir), manual_dir=str(tmp_path / "manual"))

    assert diagnostics["valuation_ready"] is True
    assert diagnostics["loaded_data_path"].endswith("api/china/normalized")
    assert diagnostics["canonical_series_ids_found"] == ["cpi", "policy_rate", "yield_10y"]
    assert diagnostics["actual_sources_found"] == ["normalized_api"]
    assert "real_yield_proxy" in diagnostics["proxy_inputs_used"]


def test_china_valuation_inspection_requires_canonical_ids(tmp_path: Path) -> None:
    """Display-style filenames should not satisfy canonical valuation requirements."""
    api_dir = tmp_path / "api"
    normalized = api_dir / "china" / "normalized"
    normalized.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01"],
            "value": [0.4],
            "series_id": ["Cpi"],
            "country": ["china"],
            "source": ["china_akshare"],
            "frequency": ["monthly"],
            "release_date": ["2026-02-15"],
            "ingested_at": ["2026-03-01"],
        }
    ).to_csv(normalized / "Cpi.csv", index=False)

    diagnostics = inspect_china_valuation_inputs(api_dir=str(api_dir), manual_dir=str(tmp_path / "manual"))

    assert diagnostics["valuation_ready"] is False
    assert diagnostics["canonical_series_ids_found"] == []


def test_china_valuation_inspection_false_when_no_actual_proxy_inputs(tmp_path: Path) -> None:
    """Valuation readiness should be false when required canonical inputs are absent."""
    diagnostics = inspect_china_valuation_inputs(
        api_dir=str(tmp_path / "api"),
        manual_dir=str(tmp_path / "manual"),
    )

    assert diagnostics["valuation_ready"] is False
    assert diagnostics["proxy_inputs_used"] == []
    assert diagnostics["proxy_inputs_missing"] == ["cpi", "policy_rate", "yield_10y", "hs300_pe_proxy", "hs300_pb_proxy"]


def test_validate_china_data_series_status_uses_actual_loaded_sources(tmp_path: Path) -> None:
    """China diagnostics table should reflect actual normalized data, not config defaults."""
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01"],
            "value": [0.4],
            "series_id": ["cpi"],
            "country": ["china"],
            "source": ["china_akshare"],
            "frequency": ["monthly"],
            "release_date": ["2026-02-15"],
            "ingested_at": ["2026-03-01"],
        }
    ).to_csv(normalized_dir / "cpi.csv", index=False)

    from app.data.china_ingestion import validate_china_data

    result = validate_china_data(base_dir=str(normalized_dir))
    cpi_row = result["series_status"].loc[result["series_status"]["series_id"] == "cpi"].iloc[0]
    unrate_row = result["series_status"].loc[result["series_status"]["series_id"] == "unrate"].iloc[0]

    assert cpi_row["source_used"] == "china_akshare"
    assert int(cpi_row["row_count"]) == 1
    assert unrate_row["source_used"] == "No loaded data"
    assert int(unrate_row["row_count"]) == 0
    assert unrate_row["status"] == "missing"
