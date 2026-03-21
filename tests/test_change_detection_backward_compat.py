"""Backward-compatibility tests for change detection history loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.change_detection import build_mode_comparison


def test_missing_run_timestamp_in_history_returns_no_comparison(tmp_path: Path) -> None:
    """Older history files without run_timestamp should gracefully degrade."""
    pd.DataFrame(
        {
            "as_of_mode": ["latest_available"],
            "summary_date": [pd.Timestamp("2026-03-01")],
            "global_regime": ["goldilocks"],
        }
    ).to_csv(tmp_path / "global_summary_history.csv", index=False)
    pd.DataFrame(
        {
            "as_of_mode": ["latest_available"],
            "asset": ["global_equities"],
            "preference": ["neutral"],
        }
    ).to_csv(tmp_path / "global_allocation_history.csv", index=False)

    comparison = build_mode_comparison("latest_available", history_dir=str(tmp_path))

    assert comparison["comparison_available"] is False
    assert pd.isna(comparison["prior_snapshot_timestamp"])
    assert comparison["regime_change_count"] == 0
    assert comparison["preference_change_count"] == 0
    assert comparison["confidence_change_count"] == 0
    assert comparison["regime_changes"] == []
    assert comparison["preference_changes"] == []
    assert comparison["confidence_changes"] == []
