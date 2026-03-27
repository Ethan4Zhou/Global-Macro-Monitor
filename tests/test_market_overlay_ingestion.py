"""Tests for shared daily market-overlay ingestion helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data import market_overlay_ingestion


def test_normalize_market_overlay_frame_applies_shared_schema() -> None:
    """Normalized overlay frames should always match the expected schema."""
    frame = pd.DataFrame(
        {
            "date": ["2026-03-19", "2026-03-20"],
            "value": ["101.5", "100.2"],
        }
    )
    normalized = market_overlay_ingestion.normalize_market_overlay_frame(
        frame,
        series_id="dxy_proxy",
        country="global",
    )
    assert list(normalized.columns) == market_overlay_ingestion.NORMALIZED_COLUMNS
    assert normalized["series_id"].iloc[-1] == "dxy_proxy"
    assert normalized["country"].iloc[-1] == "global"
    assert normalized["source"].iloc[-1] == "fred"


def test_fetch_market_overlay_bundle_normalizes_all_series(monkeypatch) -> None:
    """The bundle fetcher should normalize every configured shared series."""

    def fake_fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
        return pd.DataFrame({"date": ["2026-03-20"], "value": [1.0]})

    monkeypatch.setattr(market_overlay_ingestion, "fetch_fred_series", fake_fetch_fred_series)
    bundle = market_overlay_ingestion.fetch_market_overlay_bundle(api_key="demo")
    assert set(bundle) == set(market_overlay_ingestion.MARKET_OVERLAY_FRED_SERIES)
    assert all(not frame.empty for frame in bundle.values())
    assert bundle["dxy_proxy"]["country"].iloc[-1] == "global"
    assert bundle["sp500_proxy"]["country"].iloc[-1] == "us"


def test_save_market_overlay_series_writes_csv(tmp_path) -> None:
    """Saving one overlay series should create a CSV in the target folder."""
    frame = pd.DataFrame(
        {
            "date": ["2026-03-20"],
            "value": [1.0],
            "series_id": ["vix_proxy"],
            "country": ["global"],
            "source": ["fred"],
            "frequency": ["daily"],
            "release_date": [None],
            "ingested_at": ["2026-03-20T00:00:00+00:00"],
        }
    )
    output = market_overlay_ingestion.save_market_overlay_series(
        frame,
        series_id="vix_proxy",
        output_dir=str(tmp_path / "normalized"),
    )
    assert output == Path(tmp_path / "normalized" / "vix_proxy.csv")
    assert output.exists()


def test_fetch_market_overlay_bundle_uses_public_fallback_for_regional_equities(monkeypatch) -> None:
    """Public-site fallbacks should populate overlays when FRED fails."""

    def _raise_fetch(series_id: str, api_key: str) -> pd.DataFrame:
        raise RuntimeError("fred unavailable")

    def _fake_public_fetch(source_series_id: str, country: str, frequency: str) -> pd.DataFrame:
        mapping = {
            "gold_proxy_public": ("gold_proxy", "global", "macrotrends", 3020.0),
            "oil_proxy_public": ("oil_proxy", "global", "macrotrends", 68.4),
            "copper_proxy_public": ("copper_proxy", "global", "macrotrends", 4.92),
            "sp500_proxy_public": ("sp500_proxy", "us", "stooq", 5699.4),
            "china_equity_proxy_public": ("china_equity_proxy", "china", "tradingeconomics", 3350.0),
            "eurozone_equity_proxy_public": ("eurostoxx50_proxy", "eurozone", "tradingeconomics", 5520.0),
        }
        if source_series_id not in mapping:
            return pd.DataFrame(columns=market_overlay_ingestion.NORMALIZED_COLUMNS)
        series_name, series_country, series_source, value = mapping[source_series_id]
        return pd.DataFrame(
            {
                "date": ["2026-03-20"],
                "value": [value],
                "series_id": [series_name],
                "country": [series_country],
                "source": [series_source],
                "frequency": ["daily"],
                "release_date": ["2026-03-20"],
                "ingested_at": ["2026-03-22T00:00:00+00:00"],
            }
        )

    monkeypatch.setattr(market_overlay_ingestion, "fetch_fred_series", _raise_fetch)
    monkeypatch.setattr(market_overlay_ingestion, "fetch_public_site_series", _fake_public_fetch)

    bundle = market_overlay_ingestion.fetch_market_overlay_bundle(api_key="demo")
    assert bundle["gold_proxy"]["source"].iloc[-1] == "macrotrends"
    assert bundle["sp500_proxy"]["source"].iloc[-1] == "stooq"
    assert bundle["china_equity_proxy"]["source"].iloc[-1] == "tradingeconomics"
    assert bundle["eurostoxx50_proxy"]["source"].iloc[-1] == "tradingeconomics"
