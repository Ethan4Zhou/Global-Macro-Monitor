"""Tests for the descriptive regime evaluation layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.change_detection import append_global_allocation_history, append_global_summary_history
from app.regime.evaluation import (
    build_regime_evaluation_outputs,
    compute_confidence_bucket_summary,
    compute_forward_return_summary,
    compute_regime_transition_matrix,
)


def _write_histories(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write small summary/allocation histories for evaluation tests."""
    summary_path = tmp_path / "global_summary_history.csv"
    allocation_path = tmp_path / "global_allocation_history.csv"
    for summary_row, allocation_row, timestamp in [
        (
            {
                "as_of_mode": "latest_available",
                "summary_date": pd.Timestamp("2025-01-01"),
                "global_regime": "slowdown",
                "investment_clock": "slowdown",
            },
            {
                "date": pd.Timestamp("2025-01-01"),
                "summary_date": pd.Timestamp("2025-01-01"),
                "as_of_mode": "latest_available",
                "asset": "global_equities",
                "preference": "neutral",
                "confidence": "high",
            },
            "2025-01-15 10:00:00",
        ),
        (
            {
                "as_of_mode": "latest_available",
                "summary_date": pd.Timestamp("2025-02-01"),
                "global_regime": "goldilocks",
                "investment_clock": "disinflationary_growth",
            },
            {
                "date": pd.Timestamp("2025-02-01"),
                "summary_date": pd.Timestamp("2025-02-01"),
                "as_of_mode": "latest_available",
                "asset": "global_equities",
                "preference": "bullish",
                "confidence": "medium",
            },
            "2025-02-15 10:00:00",
        ),
        (
            {
                "as_of_mode": "latest_available",
                "summary_date": pd.Timestamp("2025-03-01"),
                "global_regime": "goldilocks",
                "investment_clock": "disinflationary_growth",
            },
            {
                "date": pd.Timestamp("2025-03-01"),
                "summary_date": pd.Timestamp("2025-03-01"),
                "as_of_mode": "latest_available",
                "asset": "global_equities",
                "preference": "bullish",
                "confidence": "low",
            },
            "2025-03-15 10:00:00",
        ),
    ]:
        append_global_summary_history(
            pd.DataFrame([summary_row]),
            pd.Timestamp(timestamp),
            history_path=str(summary_path),
        )
        append_global_allocation_history(
            pd.DataFrame([allocation_row]),
            pd.Timestamp(timestamp),
            history_path=str(allocation_path),
        )
    return pd.read_csv(summary_path), pd.read_csv(allocation_path)


def _proxy_series() -> dict[str, pd.DataFrame]:
    """Build a small monthly proxy series."""
    return {
        "global_equities": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01"]
                ),
                "value": [100, 103, 102, 104, 106, 108, 110],
            }
        )
    }


def test_transition_matrix_generation(tmp_path: Path) -> None:
    """Transition matrix should count regime switches correctly."""
    summary_history, _ = _write_histories(tmp_path)
    summary_history["summary_date"] = pd.to_datetime(summary_history["summary_date"])
    summary_history["run_timestamp"] = pd.to_datetime(summary_history["run_timestamp"])

    matrix = compute_regime_transition_matrix(summary_history)
    matched = matrix.loc[
        (matrix["from_regime"] == "slowdown") & (matrix["to_regime"] == "goldilocks")
    ].iloc[0]
    assert matched["count"] == 1


def test_forward_return_calculation(tmp_path: Path) -> None:
    """Forward return summary should produce average/median/hit ratio rows."""
    summary_history, _ = _write_histories(tmp_path)
    summary_history["summary_date"] = pd.to_datetime(summary_history["summary_date"])
    summary_history["run_timestamp"] = pd.to_datetime(summary_history["run_timestamp"])

    summary = compute_forward_return_summary(summary_history, _proxy_series())
    matched = summary.loc[
        (summary["state_type"] == "global_regime")
        & (summary["state"] == "goldilocks")
        & (summary["window_months"] == 1)
    ]
    assert not matched.empty
    assert "average_forward_return" in matched.columns
    assert "hit_ratio" in matched.columns


def test_partial_proxy_availability(tmp_path: Path) -> None:
    """Evaluation should work even when only one proxy asset exists."""
    _write_histories(tmp_path)
    returns_dir = tmp_path / "returns"
    returns_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]),
            "value": [100, 102, 101, 103],
        }
    ).to_csv(returns_dir / "global_equities.csv", index=False)

    outputs = build_regime_evaluation_outputs(
        processed_dir=str(tmp_path),
        manual_returns_dir=str(returns_dir),
    )
    assert "regime_forward_return_summary.csv" in outputs
    assert outputs["regime_forward_return_summary.csv"]["asset"].nunique() == 1


def test_confidence_bucket_summary(tmp_path: Path) -> None:
    """Confidence bucket summary should aggregate forward returns by confidence level."""
    _, allocation_history = _write_histories(tmp_path)
    allocation_history["summary_date"] = pd.to_datetime(allocation_history["summary_date"])
    allocation_history["run_timestamp"] = pd.to_datetime(allocation_history["run_timestamp"])

    summary = compute_confidence_bucket_summary(allocation_history, _proxy_series())
    matched = summary.loc[
        (summary["asset"] == "global_equities") & (summary["confidence"] == "medium")
    ]
    assert not matched.empty
    assert "average_forward_return" in matched.columns
