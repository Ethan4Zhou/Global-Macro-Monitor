"""Tests for lightweight nowcast overlay helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime import nowcast


def _write_series(path: Path, dates: list[str], source: str) -> None:
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": range(len(dates)),
            "series_id": [path.stem] * len(dates),
            "country": ["china"] * len(dates),
            "source": [source] * len(dates),
            "frequency": ["monthly"] * len(dates),
            "release_date": [None] * len(dates),
            "ingested_at": ["2026-03-20"] * len(dates),
        }
    )
    frame.to_csv(path, index=False)


def test_collect_country_input_status_prefers_normalized_api(tmp_path, monkeypatch) -> None:
    normalized = tmp_path / "data/raw/api/china/normalized"
    manual = tmp_path / "data/raw/manual/china"
    normalized.mkdir(parents=True)
    manual.mkdir(parents=True)
    _write_series(normalized / "policy_rate.csv", ["2026-03-01"], "china_akshare")
    _write_series(manual / "policy_rate.csv", ["2025-01-01"], "manual_fallback")

    monkeypatch.setattr(
        nowcast,
        "COUNTRY_INPUT_PRIORITY",
        {"china": [normalized, manual], "us": [], "eurozone": []},
    )

    status = nowcast.collect_country_input_status("china")
    assert status["latest_date"].max() == pd.Timestamp("2026-03-01")
    assert status.iloc[0]["source_used"] == "china_akshare"


def test_build_country_nowcast_overlay_flags_newer_market_inputs(tmp_path, monkeypatch) -> None:
    normalized = tmp_path / "data/raw/api/eurozone/normalized"
    normalized.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "date": ["2026-03-20"],
            "value": [2.5],
            "series_id": ["policy_rate"],
            "country": ["eurozone"],
            "source": ["ecb"],
            "frequency": ["daily"],
            "release_date": [None],
            "ingested_at": ["2026-03-20"],
        }
    )
    frame.to_csv(normalized / "policy_rate.csv", index=False)

    monkeypatch.setattr(
        nowcast,
        "COUNTRY_INPUT_PRIORITY",
        {"eurozone": [normalized], "us": [], "china": []},
    )
    monkeypatch.setattr(nowcast, "COUNTRY_PROCESSED_FILES", {"eurozone": []})

    overlay = nowcast.build_country_nowcast_overlay("eurozone", pd.Timestamp("2026-03-01"))
    assert overlay["has_newer_market_input"] is True
    assert overlay["freshest_market_date"] == pd.Timestamp("2026-03-20")
    assert overlay["freshest_market_series"] == ["policy_rate"]
