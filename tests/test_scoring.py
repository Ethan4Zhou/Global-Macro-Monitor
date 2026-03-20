"""Tests for factor scoring and regime classification."""

from __future__ import annotations

import pandas as pd

from app.factors.scoring import compute_factor_scores
from app.regime.classifier import classify_macro_regime


def test_compute_factor_scores_returns_composite() -> None:
    """Factor scoring should produce country-level composite scores."""
    frame = pd.DataFrame(
        [
            {"date": "2025-01-01", "country": "US", "indicator": "pmi", "value": 1.0},
            {"date": "2025-01-01", "country": "US", "indicator": "cpi", "value": 0.5},
        ]
    )
    weights = {"growth": {"pmi": 1.0}, "inflation": {"cpi": 1.0}}

    result = compute_factor_scores(frame, weights)

    assert list(result["country"]) == ["US"]
    assert float(result.loc[0, "composite"]) == 1.5


def test_classify_macro_regime_assigns_label() -> None:
    """Regime classifier should assign a readable regime label."""
    scores = pd.DataFrame([{"country": "US", "growth": 0.2, "inflation": -0.1}])

    result = classify_macro_regime(scores)

    assert result.loc[0, "regime"] == "Goldilocks"
