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
