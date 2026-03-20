"""Tests for valuation scoring and asset mapping."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from app.regime.allocation import map_asset_preferences
from app.valuation.features import build_country_valuation_features_frame, inspect_us_valuation_inputs
from app.valuation.models import (
    compute_valuation_confidence,
    compute_valuation_score,
    label_valuation_regime,
    summarize_valuation_inputs,
)


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


def test_compute_valuation_confidence_reflects_input_coverage() -> None:
    """Valuation confidence should rise with broader input coverage."""
    frame = pd.DataFrame(
        {
            "real_yield_proxy": [1.0, 1.0, 1.0],
            "term_spread": [0.2, 0.2, 0.2],
            "equity_pe_proxy": [None, 15.0, 15.0],
            "equity_pb_proxy": [None, None, 1.5],
        }
    )

    confidence = compute_valuation_confidence(
        frame,
        expected_inputs=["equity_pe_proxy", "equity_pb_proxy", "real_yield_proxy", "term_spread"],
    )

    assert confidence.tolist() == ["medium", "medium", "high"]


def test_summarize_valuation_inputs_uses_canonical_names() -> None:
    """Valuation input summaries should use unified canonical names."""
    frame = pd.DataFrame(
        {
            "hs300_pe_proxy": [12.0],
            "real_yield_proxy": [0.5],
            "term_spread": [0.3],
        }
    )

    used, missing = summarize_valuation_inputs(
        frame,
        expected_inputs=["equity_pe_proxy", "equity_pb_proxy", "real_yield_proxy", "term_spread"],
    )

    assert used.iloc[0] == "equity_pe_proxy,real_yield_proxy,term_spread"
    assert missing.iloc[0] == "equity_pb_proxy"


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


def test_map_asset_preferences_uses_valuation_confidence() -> None:
    """Allocation confidence should inherit partial valuation coverage."""
    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "country": ["eurozone"],
            "growth_score": [0.4],
            "inflation_score": [-0.2],
            "liquidity_score": [0.1],
            "regime": ["goldilocks"],
            "liquidity_regime": ["neutral"],
        }
    )
    valuation_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "country": ["eurozone"],
            "valuation_score": [0.2],
            "valuation_regime": ["fair"],
            "valuation_confidence": ["medium"],
        }
    )

    result = map_asset_preferences(regime_df, valuation_df)

    assert result.loc[0, "allocation_confidence"] == "medium"


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


def test_us_valuation_frame_prefers_api_first_inputs_and_asof_alignment(tmp_path: Path) -> None:
    """US valuation should use normalized API inputs with monthly as-of alignment."""
    api_dir = tmp_path / "api"
    normalized_dir = api_dir / "us" / "normalized"
    normalized_dir.mkdir(parents=True)
    manual_dir = tmp_path / "manual"
    (manual_dir / "us").mkdir(parents=True)

    pd.DataFrame(
        {
            "date": ["2025-12-01", "2026-02-01"],
            "value": [180.0, 185.0],
            "series_id": ["buffett_indicator", "buffett_indicator"],
            "source": ["fred", "fred"],
        }
    ).to_csv(normalized_dir / "buffett_indicator.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2025-12-01", "2026-02-01"],
            "value": [24.0, 26.0],
            "series_id": ["equity_pe_proxy", "equity_pe_proxy"],
            "source": ["multpl", "multpl"],
        }
    ).to_csv(normalized_dir / "equity_pe_proxy.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2025-10-01", "2026-01-01"],
            "value": [4.2, 4.5],
            "series_id": ["equity_pb_proxy", "equity_pb_proxy"],
            "source": ["multpl", "multpl"],
        }
    ).to_csv(normalized_dir / "equity_pb_proxy.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2025-12-01", "2026-02-01"],
            "value": [4.0, 3.8],
            "series_id": ["earnings_yield_proxy", "earnings_yield_proxy"],
            "source": ["multpl", "multpl"],
        }
    ).to_csv(normalized_dir / "earnings_yield_proxy.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2025-12-01", "2026-02-01"],
            "value": [1.8, 1.9],
            "series_id": ["credit_spread_proxy", "credit_spread_proxy"],
            "source": ["fred", "fred"],
        }
    ).to_csv(normalized_dir / "credit_spread_proxy.csv", index=False)

    pd.DataFrame({"date": ["2026-02-01"], "value": [999.0]}).to_csv(
        manual_dir / "us" / "buffett_indicator.csv",
        index=False,
    )

    macro_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
            "cpi_yoy": [2.0, 2.1, 2.2],
            "policy_rate_level": [4.5, 4.5, 4.5],
            "yield_10y_level": [4.6, 4.5, 4.4],
        }
    )

    valuation = build_country_valuation_features_frame(
        macro_features=macro_features,
        country="us",
        manual_dir=str(manual_dir),
        api_dir=str(api_dir),
    )

    march_row = valuation.loc[valuation["date"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert march_row["buffett_indicator"] == 185.0
    assert march_row["equity_pb_proxy"] == 4.5
    assert march_row["credit_spread_proxy"] == 1.9
    assert pd.notna(march_row["equity_risk_proxy"])
    assert "buffett_indicator" in march_row["valuation_inputs_used"]
    assert "equity_pb_proxy" in march_row["valuation_inputs_used"]
    assert march_row["valuation_confidence"] == "high"


def test_inspect_us_valuation_inputs_reflects_actual_loaded_files(tmp_path: Path) -> None:
    """US valuation diagnostics should use actual normalized data, not config defaults."""
    normalized_dir = tmp_path / "api" / "us" / "normalized"
    normalized_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01"],
            "value": [26.0],
            "series_id": ["equity_pe_proxy"],
            "source": ["multpl"],
        }
    ).to_csv(normalized_dir / "equity_pe_proxy.csv", index=False)

    diagnostics = inspect_us_valuation_inputs(api_dir=str(tmp_path / "api"))

    assert diagnostics["loaded_data_path"].endswith("api/us/normalized")
    assert diagnostics["canonical_series_ids_found"] == ["equity_pe_proxy"]
    assert diagnostics["actual_sources_found"] == ["multpl"]
    assert diagnostics["valuation_ready"] is True
