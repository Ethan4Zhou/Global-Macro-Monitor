"""Tests for external source adapters and config-driven API fetching."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.data.fetchers import fetch_country_api_bundle
from app.data.sources.ecb_client import fetch_ecb_series
from app.data.sources.eurostat_client import fetch_eurostat_series
from app.data.sources.oecd_client import fetch_oecd_series
from app.data.sources.tushare_client import fetch_tushare_series


@patch("app.data.sources.ecb_client.requests.get")
def test_ecb_adapter_normalizes_payload(mock_get: Mock) -> None:
    """ECB client should normalize a mocked SDMX payload."""
    mock_response = Mock()
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "structure": {
            "dimensions": {
                "observation": [
                    {"values": [{"id": "2024-01"}, {"id": "2024-02"}]}
                ]
            }
        },
        "dataSets": [{"series": {"0:0:0:0": {"observations": {"0": [4.0], "1": [4.0]}}}}],
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_ecb_series("FM/B.TEST", country="eurozone", frequency="monthly", api_base="https://example.com")
    assert list(result.columns) == ["date", "value", "series_id", "country", "source", "frequency", "release_date", "ingested_at"]
    assert result["country"].unique().tolist() == ["eurozone"]
    assert result["source"].unique().tolist() == ["ecb"]


@patch("app.data.sources.eurostat_client.requests.get")
def test_eurostat_adapter_parses_jsonstat(mock_get: Mock) -> None:
    """Eurostat client should parse mocked JSON-stat observations."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "dimension": {
            "time": {
                "category": {
                    "index": {"2024M01": 0, "2024M02": 1}
                }
            }
        },
        "value": {"0": 2.6, "1": 2.4},
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_eurostat_series("dataset?geo=EA20", country="eurozone", frequency="monthly", api_base="https://example.com")
    assert result["value"].tolist() == [2.6, 2.4]
    assert result["source"].iloc[0] == "eurostat"


@patch("app.data.sources.oecd_client.requests.get")
def test_oecd_adapter_parses_csv(mock_get: Mock) -> None:
    """OECD client should parse mocked CSV payloads."""
    mock_response = Mock()
    mock_response.text = "TIME_PERIOD,OBS_VALUE\n2024-01,49.8\n2024-02,50.1\n"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_oecd_series("OECD.TEST", country="eurozone", frequency="monthly", api_base="https://example.com")
    assert result["value"].tolist() == [49.8, 50.1]
    assert result["source"].iloc[0] == "oecd"


@patch("app.data.sources.tushare_client.requests.post")
def test_tushare_adapter_parses_payload(mock_post: Mock) -> None:
    """Tushare client should normalize mocked API payloads."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "data": {
            "fields": ["month", "nt_yoy"],
            "items": [["202401", 0.3], ["202402", 0.7]],
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = fetch_tushare_series("cn_cpi", country="china", frequency="monthly", token="demo-token")
    assert result["country"].iloc[0] == "china"
    assert result["source"].iloc[0] == "tushare"
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


@patch("app.data.sources.tushare_client.requests.post")
def test_tushare_adapter_supports_hs300_dailybasic_fields(mock_post: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tushare client should support HS300 PE/PB proxies via index_dailybasic."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "code": 0,
        "data": {
            "fields": ["trade_date", "pe_ttm"],
            "items": [["20250331", 12.38], ["20250401", 12.4]],
        },
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    monkeypatch.setenv("TUSHARE_API_URL", "http://example.test")

    result = fetch_tushare_series("hs300_pe_proxy", country="china", frequency="monthly", token="demo-token")

    assert result["series_id"].iloc[0] == "hs300_pe_proxy"
    assert result["value"].iloc[0] == 12.38
    assert mock_post.call_args.kwargs["json"]["api_name"] == "index_dailybasic"


def test_config_driven_country_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Country API fetching should route indicators through configured adapters."""
    called: list[str] = []

    def _fake_fetcher(*args, **kwargs):  # type: ignore[no-untyped-def]
        called.append(kwargs.get("source_series_id") or kwargs.get("series_id"))
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "value": [1.0],
                "series_id": [kwargs.get("source_series_id") or kwargs.get("series_id")],
                "country": [kwargs["country"]],
                "source": ["mock"],
                "frequency": [kwargs["frequency"]],
                "release_date": pd.to_datetime(["2024-01-01"]),
                "ingested_at": pd.to_datetime(["2024-01-02"]),
            }
        )

    monkeypatch.setattr("app.data.fetchers.API_SOURCE_FETCHERS", {
        "eurostat": _fake_fetcher,
        "ecb": _fake_fetcher,
        "oecd": _fake_fetcher,
        "tushare": _fake_fetcher,
    })
    bundle = fetch_country_api_bundle("eurozone")
    assert "cpi" in bundle
    assert "policy_rate" in bundle
    assert called


def test_config_driven_fetch_allows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured fallback sources should prevent a hard failure on adapter errors."""
    def _raise_fetcher(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr("app.data.fetchers.API_SOURCE_FETCHERS", {
        "eurostat": _raise_fetcher,
        "ecb": _raise_fetcher,
        "oecd": _raise_fetcher,
        "tushare": _raise_fetcher,
    })
    bundle = fetch_country_api_bundle("eurozone")
    assert bundle == {}
