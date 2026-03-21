"""Tests for lightweight nowcast overlay helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime import nowcast


def _write_series(path: Path, dates: list[str], source: str, *, country: str = "china") -> None:
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": range(len(dates)),
            "series_id": [path.stem] * len(dates),
            "country": [country] * len(dates),
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
    assert overlay["overlay_direction"] == "neutral"
    assert overlay["overlay_confidence"] == "low"


def test_build_country_nowcast_overlay_scores_easing_signal(tmp_path, monkeypatch) -> None:
    normalized = tmp_path / "data/raw/api/china/normalized"
    normalized.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-02-01", "2026-03-01"],
            "value": [2.0, 1.8],
            "series_id": ["policy_rate", "policy_rate"],
            "country": ["china", "china"],
            "source": ["china_akshare", "china_akshare"],
            "frequency": ["monthly", "monthly"],
            "release_date": [None, None],
            "ingested_at": ["2026-03-20", "2026-03-20"],
        }
    ).to_csv(normalized / "policy_rate.csv", index=False)

    monkeypatch.setattr(
        nowcast,
        "COUNTRY_INPUT_PRIORITY",
        {"china": [normalized], "us": [], "eurozone": []},
    )
    monkeypatch.setattr(nowcast, "COUNTRY_PROCESSED_FILES", {"china": []})

    overlay = nowcast.build_country_nowcast_overlay("china", pd.Timestamp("2026-02-01"))
    assert overlay["overlay_score"] > 0
    assert overlay["overlay_direction"] == "risk_on"
    assert overlay["overlay_confidence"] == "medium"
    assert overlay["signal_drivers"][0]["driver"] == "easing"


def test_collect_country_input_status_reads_shared_global_market_inputs(tmp_path, monkeypatch) -> None:
    """Shared global market series should be available to each region."""
    normalized = tmp_path / "data/raw/api/china/normalized"
    global_markets = tmp_path / "data/raw/api/global_markets/normalized"
    normalized.mkdir(parents=True)
    global_markets.mkdir(parents=True)
    _write_series(normalized / "policy_rate.csv", ["2026-03-01"], "china_akshare", country="china")
    _write_series(global_markets / "dxy_proxy.csv", ["2026-03-20"], "fred", country="global")

    monkeypatch.setattr(
        nowcast,
        "COUNTRY_INPUT_PRIORITY",
        {"china": [normalized, global_markets], "us": [], "eurozone": []},
    )

    status = nowcast.collect_country_input_status("china")
    assert set(status["series_id"]) == {"policy_rate", "dxy_proxy"}
    assert status.loc[status["series_id"] == "dxy_proxy", "source_used"].iloc[0] == "fred"


def test_build_country_nowcast_overlay_scores_global_market_risk_signal(tmp_path, monkeypatch) -> None:
    """Daily shared market inputs should feed the risk overlay."""
    normalized = tmp_path / "data/raw/api/china/normalized"
    global_markets = tmp_path / "data/raw/api/global_markets/normalized"
    normalized.mkdir(parents=True)
    global_markets.mkdir(parents=True)
    _write_series(normalized / "policy_rate.csv", ["2026-03-01"], "china_akshare", country="china")
    pd.DataFrame(
        {
            "date": ["2026-03-19", "2026-03-20"],
            "value": [102.0, 101.0],
            "series_id": ["dxy_proxy", "dxy_proxy"],
            "country": ["global", "global"],
            "source": ["fred", "fred"],
            "frequency": ["daily", "daily"],
            "release_date": [None, None],
            "ingested_at": ["2026-03-20", "2026-03-20"],
        }
    ).to_csv(global_markets / "dxy_proxy.csv", index=False)

    monkeypatch.setattr(
        nowcast,
        "COUNTRY_INPUT_PRIORITY",
        {"china": [normalized, global_markets], "us": [], "eurozone": []},
    )
    monkeypatch.setattr(nowcast, "COUNTRY_PROCESSED_FILES", {"china": []})

    overlay = nowcast.build_country_nowcast_overlay("china", pd.Timestamp("2026-03-01"))
    assert overlay["has_newer_market_input"] is True
    assert overlay["dimension_scores"]["risk"] > 0
    assert overlay["signal_drivers"][0]["driver"] == "weaker_dollar"


def test_build_global_nowcast_overlay_aggregates_country_signals(tmp_path, monkeypatch) -> None:
    us_dir = tmp_path / "data/raw/fred"
    china_dir = tmp_path / "data/raw/api/china/normalized"
    eurozone_dir = tmp_path / "data/raw/api/eurozone/normalized"
    us_dir.mkdir(parents=True)
    china_dir.mkdir(parents=True)
    eurozone_dir.mkdir(parents=True)

    pd.DataFrame(
        {"date": ["2026-02-01", "2026-03-01"], "value": [4.5, 4.3]}
    ).assign(series_id="policy_rate", country="us", source="fred", frequency="monthly", release_date=None, ingested_at="2026-03-20").to_csv(us_dir / "FEDFUNDS.csv", index=False)
    pd.DataFrame(
        {"date": ["2026-02-01", "2026-03-01"], "value": [2.0, 1.8]}
    ).assign(series_id="policy_rate", country="china", source="china_akshare", frequency="monthly", release_date=None, ingested_at="2026-03-20").to_csv(china_dir / "policy_rate.csv", index=False)
    pd.DataFrame(
        {"date": ["2026-02-01", "2026-03-01"], "value": [2.5, 2.7]}
    ).assign(series_id="policy_rate", country="eurozone", source="ecb", frequency="monthly", release_date=None, ingested_at="2026-03-20").to_csv(eurozone_dir / "policy_rate.csv", index=False)

    monkeypatch.setattr(
        nowcast,
        "COUNTRY_INPUT_PRIORITY",
        {"us": [us_dir], "china": [china_dir], "eurozone": [eurozone_dir]},
    )
    monkeypatch.setattr(nowcast, "COUNTRY_PROCESSED_FILES", {"us": [], "china": [], "eurozone": []})

    overlay = nowcast.build_global_nowcast_overlay(
        pd.Timestamp("2026-02-01"),
        {
            "us": pd.Timestamp("2026-02-01"),
            "china": pd.Timestamp("2026-02-01"),
            "eurozone": pd.Timestamp("2026-02-01"),
        },
    )
    assert overlay["overlay_direction"] in {"neutral", "risk_on", "defensive"}
    assert overlay["overlay_confidence"] in {"medium", "high"}
    assert overlay["overlay_drivers"]
