"""Tests for the top-level refresh orchestration."""

from __future__ import annotations

import main as cli_main


def test_refresh_monitor_continues_when_some_fetches_fail(monkeypatch) -> None:
    """A failed source fetch should not stop the rest of the refresh pipeline."""
    calls: list[str] = []

    def fake_fetch_us() -> None:
        calls.append("fetch_us")

    def fake_fetch_country_api_data(country: str) -> None:
        calls.append(f"fetch_{country}")
        if country == "china":
            raise RuntimeError("china feed unavailable")

    def fake_run_global_monitor() -> None:
        calls.append("run_global_monitor")

    def fake_fetch_consensus_sources(region: str) -> None:
        calls.append(f"consensus_{region}")

    def fake_build_consensus_snapshots() -> None:
        calls.append("build_consensus_snapshots")

    def fake_build_consensus_deviation() -> None:
        calls.append("build_consensus_deviation")

    def fake_evaluate_regimes() -> None:
        calls.append("evaluate_regimes")

    def fake_evaluate_confidence() -> None:
        calls.append("evaluate_confidence")

    def fake_build_alerts() -> None:
        calls.append("build_alerts")

    monkeypatch.setattr(cli_main, "run_fetch_us", fake_fetch_us)
    monkeypatch.setattr(cli_main, "run_fetch_country_api_data", fake_fetch_country_api_data)
    monkeypatch.setattr(cli_main, "run_global_monitor", fake_run_global_monitor)
    monkeypatch.setattr(cli_main, "run_fetch_consensus_sources", fake_fetch_consensus_sources)
    monkeypatch.setattr(cli_main, "run_build_consensus_snapshots", fake_build_consensus_snapshots)
    monkeypatch.setattr(cli_main, "run_build_consensus_deviation", fake_build_consensus_deviation)
    monkeypatch.setattr(cli_main, "run_evaluate_regimes", fake_evaluate_regimes)
    monkeypatch.setattr(cli_main, "run_evaluate_confidence", fake_evaluate_confidence)
    monkeypatch.setattr(cli_main, "run_build_alerts", fake_build_alerts)

    cli_main.run_refresh_monitor()

    assert "fetch_us" in calls
    assert "fetch_china" in calls
    assert "run_global_monitor" in calls
    assert "consensus_us" in calls
    assert "consensus_china" in calls
    assert calls[-1] == "build_alerts"
