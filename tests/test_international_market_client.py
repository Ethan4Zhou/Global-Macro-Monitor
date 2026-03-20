"""Tests for international equity valuation adapters."""

from __future__ import annotations

import pandas as pd

from app.data.sources.international_market_client import fetch_international_market_series


class _DummyResponse:
    """Simple mock response wrapper."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        """Pretend the request succeeded."""


def test_fetch_international_market_series_normalizes_eurozone_pe(monkeypatch) -> None:
    """Eurozone PE proxy should parse and expand into the shared normalized schema."""
    html = """
    <table>
      <thead>
        <tr><th>Date</th><th>Price</th><th>PE Ratio</th><th>CAPE Ratio</th></tr>
      </thead>
      <tbody>
        <tr><td>Mar 17, 2026</td><td>620.0</td><td>27.1</td><td>19.9</td></tr>
        <tr><td>Sep 20, 2025</td><td>590.0</td><td>24.3</td><td>18.2</td></tr>
      </tbody>
    </table>
    """

    monkeypatch.setattr(
        "app.data.sources.international_market_client.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    frame = fetch_international_market_series(
        source_series_id="eurozone_equity_pe_proxy",
        country="eurozone",
        frequency="monthly",
    )

    assert list(frame["series_id"].unique()) == ["equity_pe_proxy"]
    assert list(frame["source"].unique()) == ["siblis"]
    assert frame["date"].max() == pd.Timestamp("2026-03-01")
    assert pd.to_numeric(frame["value"], errors="coerce").notna().all()


def test_fetch_international_market_series_maps_china_cape(monkeypatch) -> None:
    """China CAPE proxy should map into the canonical shiller_pe_proxy id."""
    html = """
    <table>
      <thead>
        <tr><th>Date</th><th>Price</th><th>PE Ratio</th><th>CAPE Ratio</th></tr>
      </thead>
      <tbody>
        <tr><td>Mar 19, 2026</td><td>3550.0</td><td>15.4</td><td>14.1</td></tr>
        <tr><td>Sep 10, 2025</td><td>3320.0</td><td>14.8</td><td>13.7</td></tr>
      </tbody>
    </table>
    """

    monkeypatch.setattr(
        "app.data.sources.international_market_client.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    frame = fetch_international_market_series(
        source_series_id="china_shiller_pe_proxy",
        country="china",
        frequency="monthly",
    )

    assert list(frame["series_id"].unique()) == ["shiller_pe_proxy"]
    assert frame["country"].iloc[-1] == "china"
    assert frame["date"].max() == pd.Timestamp("2026-03-01")
