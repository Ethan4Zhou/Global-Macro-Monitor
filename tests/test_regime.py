"""Tests for score and regime classification helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.regime import classifier as classifier_module
from app.factors.scoring import (
    compute_growth_score,
    compute_inflation_score,
    compute_liquidity_score,
)
from app.regime.classifier import classify_us_macro_regime


def test_score_functions_return_finite_values_with_sufficient_history() -> None:
    """Score functions should return finite values once the feature panel is populated."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "unrate_3m_avg": [4.5, 4.4, 4.3, 4.2, 4.1, 4.0],
            "unrate_diff_3m": [np.nan, np.nan, np.nan, -0.3, -0.3, -0.3],
            "cpi_yoy": [3.8, 3.6, 3.4, 3.2, 3.0, 2.8],
            "core_cpi_yoy": [4.2, 4.0, 3.8, 3.6, 3.4, 3.2],
            "fedfunds_level": [5.5, 5.4, 5.3, 5.1, 4.9, 4.7],
            "fedfunds_diff_3m": [np.nan, np.nan, np.nan, -0.4, -0.5, -0.6],
            "gs10_diff_3m": [np.nan, np.nan, np.nan, -0.2, -0.3, -0.4],
            "m2_yoy": [0.5, 0.8, 1.2, 1.6, 2.0, 2.4],
        }
    )

    growth = compute_growth_score(frame)
    inflation = compute_inflation_score(frame)
    liquidity = compute_liquidity_score(frame)

    assert np.isfinite(growth.iloc[-1])
    assert np.isfinite(inflation.iloc[-1])
    assert np.isfinite(liquidity.iloc[-1])


def test_classify_us_macro_regime_labels_handcrafted_examples() -> None:
    """Classifier should map score signs to the expected regime labels."""
    features = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "unrate_3m_avg": [5.5, 4.8, 4.0, 3.5],
            "unrate_diff_3m": [0.4, 0.2, -0.2, -0.4],
            "cpi_yoy": [5.0, 4.5, 2.0, 1.5],
            "core_cpi_yoy": [5.2, 4.7, 2.1, 1.4],
            "fedfunds_level": [5.5, 5.0, 4.0, 3.0],
            "fedfunds_diff_3m": [0.5, 0.2, -0.2, -0.6],
            "gs10_diff_3m": [0.3, 0.1, -0.1, -0.4],
            "m2_yoy": [-1.0, 0.0, 2.0, 4.0],
        }
    )

    result = classify_us_macro_regime(features)

    assert result.loc[0, "regime"] == "unknown"
    assert result.loc[1, "regime"] == "unknown"
    assert result.loc[2, "regime"] == "goldilocks"
    assert result.loc[3, "regime"] == "goldilocks"
    assert result.loc[0, "regime_confidence"] == "unknown"
    assert result.loc[2, "regime_confidence"] in {"medium", "high"}
    assert set(result["liquidity_regime"]) <= {"easy", "neutral", "tight", "unknown"}
    assert {"regime_raw", "regime_confidence", "regime_note"}.issubset(result.columns)


def test_historical_standardization_does_not_look_ahead() -> None:
    """Later outliers should not change previously computed scores."""
    base = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "cpi_yoy": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    extended = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-06-01")],
                    "cpi_yoy": [10.0],
                }
            ),
        ],
        ignore_index=True,
    )

    base_scores = compute_inflation_score(base)
    extended_scores = compute_inflation_score(extended)

    pd.testing.assert_series_equal(
        base_scores.reset_index(drop=True),
        extended_scores.iloc[: len(base_scores)].reset_index(drop=True),
        check_names=False,
    )


def test_regime_classifier_holds_boundary_flip_until_signal_strengthens(monkeypatch) -> None:
    """Near-zero score flips should be smoothed instead of changing regime immediately."""
    features = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "country": ["us"] * 4,
        }
    )

    monkeypatch.setattr(
        classifier_module,
        "compute_growth_score",
        lambda frame, country=None: pd.Series([0.8, 0.4, -0.05, -0.8], index=frame.index, dtype="float64"),
    )
    monkeypatch.setattr(
        classifier_module,
        "compute_inflation_score",
        lambda frame, country=None: pd.Series([0.9, 0.8, 0.7, 0.8], index=frame.index, dtype="float64"),
    )
    monkeypatch.setattr(
        classifier_module,
        "compute_liquidity_score",
        lambda frame, country=None: pd.Series([0.0, 0.0, 0.0, 0.0], index=frame.index, dtype="float64"),
    )

    result = classifier_module.classify_country_macro_regime(features, country="us")

    assert result["regime_raw"].tolist() == ["reflation", "reflation", "stagflation", "stagflation"]
    assert result["regime"].tolist() == ["reflation", "reflation", "reflation", "stagflation"]
    assert result.loc[2, "regime_confidence"] == "low"
    assert "held back" in result.loc[2, "regime_note"]
