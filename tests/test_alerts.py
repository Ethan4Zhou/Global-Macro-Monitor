"""Tests for high-value monitor alerts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime import alerts


def test_build_monitor_alerts_creates_expected_rows(tmp_path, monkeypatch) -> None:
    """Alert builder should surface only high-value monitor states."""
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "as_of_mode": "latest_available",
                "summary_date": "2026-03-01",
                "global_regime": "partial_view",
                "coverage_ratio": 0.5,
                "coverage_warning": "Global summary is based on incomplete country coverage.",
                "us_latest_date": "2026-03-01",
                "china_latest_date": "2026-03-01",
                "eurozone_latest_date": "2026-03-01",
            }
        ]
    ).to_csv(processed / "global_macro_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "region": "china",
                "snapshot_date": "2026-03-01",
                "consensus_deviation_score": 0.8,
                "deviation_summary": "Model is more growth-negative than consensus.",
            }
        ]
    ).to_csv(processed / "consensus_deviation.csv", index=False)

    monkeypatch.setattr(
        alerts,
        "build_mode_comparison",
        lambda *args, **kwargs: {
            "comparison_available": True,
            "regime_changes": [
                {
                    "entity_name": "global_regime",
                    "old_value": "slowdown",
                    "new_value": "goldilocks",
                    "summary_date": pd.Timestamp("2026-03-01"),
                    "reason": "regime transition",
                }
            ],
            "confidence_changes": [
                {
                    "entity_name": "global_equities",
                    "old_value": "high",
                    "new_value": "medium",
                    "direction": "downgrade",
                    "summary_date": pd.Timestamp("2026-03-01"),
                    "reason": "missing valuation inputs",
                }
            ],
        },
    )
    monkeypatch.setattr(
        alerts,
        "build_country_status",
        lambda mode: pd.DataFrame(
            [
                {
                    "country": "china",
                    "country_ready": True,
                    "globally_usable_latest": False,
                    "staleness_status": "very_stale",
                    "days_stale": 220,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        alerts,
        "build_global_nowcast_overlay",
        lambda summary_date, country_regime_dates: {
            "overlay_direction": "defensive",
            "overlay_score": -0.7,
            "freshest_market_date": pd.Timestamp("2026-03-02"),
            "overlay_drivers": ["china:policy_rate:tightening:rates"],
        },
    )
    monkeypatch.setattr(alerts, "ALERTS_PATH", processed / "monitor_alerts.csv")

    output = alerts.build_monitor_alerts(processed_dir=str(processed))

    assert not output.empty
    assert {"partial_coverage", "global_regime", "confidence_downgrade", "very_stale_country", "country_not_usable", "consensus_gap", "nowcast_shift"}.issubset(set(output["alert_type"]))
    assert (processed / "monitor_alerts.csv").exists()


def test_build_monitor_alerts_writes_empty_when_summary_missing(tmp_path, monkeypatch) -> None:
    """Alert builder should still emit a valid empty CSV when upstream summary is missing."""
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(alerts, "ALERTS_PATH", processed / "monitor_alerts.csv")

    output = alerts.build_monitor_alerts(processed_dir=str(processed))

    assert output.empty
    assert (processed / "monitor_alerts.csv").exists()
