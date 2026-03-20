"""Tests for FRED data fetching helpers."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd

from app.data.fetchers import fetch_fred_series


@patch("app.data.fetchers.requests.get")
def test_fetch_fred_series_cleans_observations(mock_get: Mock) -> None:
    """FRED observations should be parsed into a clean typed DataFrame."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "observations": [
            {"date": "2024-02-01", "value": "."},
            {"date": "2024-01-01", "value": "3.1"},
            {"date": "2024-03-01", "value": "3.4"},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_fred_series(series_id="CPIAUCSL", api_key="demo-key")

    assert list(result.columns) == ["date", "value", "series_id"]
    assert list(result["series_id"].unique()) == ["CPIAUCSL"]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]
    assert result["value"].tolist()[0] == 3.1
    assert pd.isna(result["value"].tolist()[1])
