"""Tests for free public-site adapters."""

from __future__ import annotations

import pandas as pd

from app.data.sources.public_site_client import fetch_public_site_series


class _DummyResponse:
    """Simple mock response wrapper."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        """Pretend the request succeeded."""


def test_fetch_public_site_series_parses_tradingeconomics_monthly_latest(monkeypatch) -> None:
    """TradingEconomics public pages should normalize latest monthly macro rows."""
    html = """
    <html><body>
      China's core inflation rate rose to 0.6 percent in February 2026 from 0.4 percent in January 2026.
    </body></html>
    """
    monkeypatch.setattr(
        "app.data.sources.public_site_client.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    frame = fetch_public_site_series(
        source_series_id="china_core_cpi_public",
        country="china",
        frequency="monthly",
    )

    assert list(frame["series_id"].unique()) == ["core_cpi"]
    assert list(frame["source"].unique()) == ["tradingeconomics"]
    assert frame["date"].iloc[0] == pd.Timestamp("2026-02-01")
    assert frame["value"].iloc[0] == 0.6


def test_fetch_public_site_series_parses_ycharts_metric(monkeypatch) -> None:
    """YCharts metric pages should normalize ETF proxy valuation rows."""
    html = """
    <html><body>
      <div>Weighted Average Price to Book Ratio</div>
      <div>2.34</div>
    </body></html>
    """
    monkeypatch.setattr(
        "app.data.sources.public_site_client.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    frame = fetch_public_site_series(
        source_series_id="eurozone_equity_pb_proxy_public",
        country="eurozone",
        frequency="monthly",
    )

    assert frame["series_id"].iloc[0] == "equity_pb_proxy"
    assert frame["source"].iloc[0] == "ycharts"
    assert frame["date"].iloc[0].day == 1
    assert frame["value"].iloc[0] == 2.34


def test_fetch_public_site_series_parses_macrotrends_history(monkeypatch) -> None:
    """Macrotrends pages should normalize daily commodity history rows."""
    html = """
    <script>
      var originalData = [
        { date: "2026-03-19", close: "3012.5" },
        { date: "2026-03-20", close: "3020.1" }
      ];
    </script>
    """
    monkeypatch.setattr(
        "app.data.sources.public_site_client.requests.get",
        lambda *args, **kwargs: _DummyResponse(html),
    )

    frame = fetch_public_site_series(
        source_series_id="gold_proxy_public",
        country="global",
        frequency="daily",
    )

    assert frame["series_id"].iloc[-1] == "gold_proxy"
    assert frame["source"].iloc[-1] == "macrotrends"
    assert frame["date"].iloc[-1] == pd.Timestamp("2026-03-20")
    assert frame["value"].iloc[-1] == 3020.1


def test_fetch_public_site_series_parses_stooq_csv(monkeypatch) -> None:
    """Stooq CSV downloads should normalize close prices into daily rows."""
    csv_text = "Date,Open,High,Low,Close,Volume\n2026-03-19,0,0,0,5670.2,0\n2026-03-20,0,0,0,5699.4,0\n"
    monkeypatch.setattr(
        "app.data.sources.public_site_client.requests.get",
        lambda *args, **kwargs: _DummyResponse(csv_text),
    )

    frame = fetch_public_site_series(
        source_series_id="sp500_proxy_public",
        country="us",
        frequency="daily",
    )

    assert frame["series_id"].iloc[-1] == "sp500_proxy"
    assert frame["source"].iloc[-1] == "stooq"
    assert frame["date"].iloc[-1] == pd.Timestamp("2026-03-20")
    assert frame["value"].iloc[-1] == 5699.4


def test_fetch_public_site_series_parses_pboc_m2_report(monkeypatch) -> None:
    """PBOC financial report pages should normalize latest M2 growth."""
    responses = {
        "https://www.pbc.gov.cn/en/3688247/3688978/3709137/index.html": """
        <html><body>
          <a href="/en/3688247/3688978/3709137/6000001/index.html">Financial Statistics Report (February 2026)</a>
        </body></html>
        """,
        "https://www.pbc.gov.cn/en/3688247/3688978/3709137/6000001/index.html": """
        <html><body>
          Broad money supply (M2) rising by 7.2 percent year on year.
        </body></html>
        """,
    }

    monkeypatch.setattr(
        "app.data.sources.public_site_client.requests.get",
        lambda url, **kwargs: _DummyResponse(responses[url]),
    )

    frame = fetch_public_site_series(
        source_series_id="china_m2_pbc_public",
        country="china",
        frequency="monthly",
    )

    assert frame["series_id"].iloc[0] == "m2"
    assert frame["source"].iloc[0] == "pboc"
    assert frame["date"].iloc[0] == pd.Timestamp("2026-02-01")
    assert frame["value"].iloc[0] == 7.2
