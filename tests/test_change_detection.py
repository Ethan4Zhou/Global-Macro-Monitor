"""Tests for comparison-object based change detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.change_detection import (
    append_global_allocation_history,
    append_global_summary_history,
    build_mode_comparison,
)


def _write_histories(
    tmp_path: Path,
    *,
    mode: str = "latest_available",
    summary_rows: list[dict[str, object]],
    allocation_rows: list[pd.DataFrame],
    timestamps: list[str],
) -> None:
    """Write aligned summary and allocation histories for one mode."""
    summary_path = tmp_path / "global_summary_history.csv"
    allocation_path = tmp_path / "global_allocation_history.csv"
    for summary_row, allocation_frame, timestamp in zip(summary_rows, allocation_rows, timestamps):
        append_global_summary_history(
            pd.DataFrame([{**summary_row, "as_of_mode": mode}]),
            pd.Timestamp(timestamp),
            history_path=str(summary_path),
        )
        allocation = allocation_frame.copy()
        allocation["as_of_mode"] = mode
        append_global_allocation_history(
            allocation,
            pd.Timestamp(timestamp),
            history_path=str(allocation_path),
        )


def test_no_prior_snapshot_means_no_changes(tmp_path: Path) -> None:
    """Without a prior comparable snapshot, all change sections should stay empty."""
    _write_histories(
        tmp_path,
        summary_rows=[
            {
                "summary_date": pd.Timestamp("2025-02-01"),
                "global_regime": "goldilocks",
                "investment_clock": "disinflationary_growth",
                "us_regime": "goldilocks",
                "china_regime": "goldilocks",
                "eurozone_regime": "slowdown",
            }
        ],
        allocation_rows=[
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-02-01"]),
                    "summary_date": pd.to_datetime(["2025-02-01"]),
                    "asset": ["global_equities"],
                    "preference": ["bullish"],
                    "confidence": ["medium"],
                }
            )
        ],
        timestamps=["2025-02-15 10:00:00"],
    )

    comparison = build_mode_comparison("latest_available", processed_dir=str(tmp_path))
    assert comparison["comparison_available"] is False
    assert pd.isna(comparison["prior_snapshot_timestamp"])
    assert comparison["regime_change_count"] == 0
    assert comparison["preference_change_count"] == 0
    assert comparison["confidence_change_count"] == 0
    assert comparison["comparison_reason"] == "No prior snapshot is available yet for this mode."


def test_prior_snapshot_exists_allows_changes(tmp_path: Path) -> None:
    """With a prior comparable snapshot, changes should be available."""
    _write_histories(
        tmp_path,
        summary_rows=[
            {
                "summary_date": pd.Timestamp("2025-01-01"),
                "global_regime": "slowdown",
                "investment_clock": "slowdown",
                "us_regime": "slowdown",
                "china_regime": "goldilocks",
                "eurozone_regime": "slowdown",
            },
            {
                "summary_date": pd.Timestamp("2025-02-01"),
                "global_regime": "goldilocks",
                "investment_clock": "disinflationary_growth",
                "us_regime": "goldilocks",
                "china_regime": "goldilocks",
                "eurozone_regime": "slowdown",
            },
        ],
        allocation_rows=[
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01"]),
                    "summary_date": pd.to_datetime(["2025-01-01"]),
                    "asset": ["global_equities"],
                    "preference": ["neutral"],
                    "confidence": ["high"],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-02-01"]),
                    "summary_date": pd.to_datetime(["2025-02-01"]),
                    "asset": ["global_equities"],
                    "preference": ["bullish"],
                    "confidence": ["medium"],
                }
            ),
        ],
        timestamps=["2025-01-15 10:00:00", "2025-02-15 10:00:00"],
    )

    comparison = build_mode_comparison("latest_available", processed_dir=str(tmp_path))
    assert comparison["comparison_available"] is True
    assert pd.notna(comparison["prior_snapshot_timestamp"])
    assert comparison["regime_change_count"] > 0
    assert comparison["preference_change_count"] > 0
    assert comparison["confidence_change_count"] > 0


def test_mode_mismatch_means_no_comparison(tmp_path: Path) -> None:
    """Snapshots from another mode must not be used for the selected mode."""
    _write_histories(
        tmp_path,
        mode="last_common_date",
        summary_rows=[
            {
                "summary_date": pd.Timestamp("2025-02-01"),
                "global_regime": "goldilocks",
                "investment_clock": "disinflationary_growth",
                "us_regime": "goldilocks",
                "china_regime": "goldilocks",
                "eurozone_regime": "slowdown",
            }
        ],
        allocation_rows=[
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-02-01"]),
                    "summary_date": pd.to_datetime(["2025-02-01"]),
                    "asset": ["global_equities"],
                    "preference": ["bullish"],
                    "confidence": ["medium"],
                }
            )
        ],
        timestamps=["2025-02-15 10:00:00"],
    )

    comparison = build_mode_comparison("latest_available", processed_dir=str(tmp_path))
    assert comparison["comparison_available"] is False
    assert comparison["regime_changes"] == []


def test_schema_mismatch_means_no_comparison(tmp_path: Path) -> None:
    """Schema mismatches should block comparison and explain why."""
    summary_path = tmp_path / "global_summary_history.csv"
    allocation_path = tmp_path / "global_allocation_history.csv"
    pd.DataFrame(
        {
            "as_of_mode": ["latest_available", "latest_available"],
            "summary_date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "global_regime": ["slowdown", "goldilocks"],
            "investment_clock": ["slowdown", "disinflationary_growth"],
            "us_regime": ["slowdown", "goldilocks"],
            "run_timestamp": pd.to_datetime(["2025-01-15 10:00:00", "2025-02-15 10:00:00"]),
            "selected_mode": ["latest_available", "latest_available"],
            "schema_version": ["older", "newer"],
        }
    ).to_csv(summary_path, index=False)
    pd.DataFrame(
        {
            "as_of_mode": ["latest_available", "latest_available"],
            "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "summary_date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "asset": ["global_equities", "global_equities"],
            "preference": ["neutral", "bullish"],
            "confidence": ["high", "medium"],
            "run_timestamp": pd.to_datetime(["2025-01-15 10:00:00", "2025-02-15 10:00:00"]),
            "selected_mode": ["latest_available", "latest_available"],
            "schema_version": ["older", "newer"],
        }
    ).to_csv(allocation_path, index=False)

    comparison = build_mode_comparison("latest_available", processed_dir=str(tmp_path))
    assert comparison["comparison_available"] is False
    assert comparison["comparison_reason"] == "Change history for this mode starts from the latest schema version."


def test_history_dir_overrides_processed_dir(tmp_path: Path) -> None:
    """Comparison should be able to read history from a dedicated runtime directory."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    _write_histories(
        runtime_dir,
        summary_rows=[
            {
                "summary_date": pd.Timestamp("2025-01-01"),
                "global_regime": "slowdown",
                "investment_clock": "slowdown",
                "us_regime": "slowdown",
                "china_regime": "goldilocks",
                "eurozone_regime": "slowdown",
            },
            {
                "summary_date": pd.Timestamp("2025-02-01"),
                "global_regime": "goldilocks",
                "investment_clock": "disinflationary_growth",
                "us_regime": "goldilocks",
                "china_regime": "goldilocks",
                "eurozone_regime": "slowdown",
            },
        ],
        allocation_rows=[
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01"]),
                    "summary_date": pd.to_datetime(["2025-01-01"]),
                    "asset": ["global_equities"],
                    "preference": ["neutral"],
                    "confidence": ["high"],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-02-01"]),
                    "summary_date": pd.to_datetime(["2025-02-01"]),
                    "asset": ["global_equities"],
                    "preference": ["bullish"],
                    "confidence": ["medium"],
                }
            ),
        ],
        timestamps=["2025-01-15 10:00:00", "2025-02-15 10:00:00"],
    )

    comparison = build_mode_comparison(
        "latest_available",
        processed_dir=str(tmp_path / "processed"),
        history_dir=str(runtime_dir),
    )
    assert comparison["comparison_available"] is True
    assert comparison["preference_change_count"] > 0
