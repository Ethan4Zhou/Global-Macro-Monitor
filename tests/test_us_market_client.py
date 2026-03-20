"""Tests for the US public market valuation adapter."""

from __future__ import annotations

from textwrap import dedent

from app.data.sources.us_market_client import fetch_us_market_series


class _MockResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_fetch_us_market_series_normalizes_monthly_multpl_table(monkeypatch) -> None:
    """Monthly Multpl tables should normalize to month-start valuation rows."""
    html = dedent(
        """
        <table>
          <thead><tr><th>Date</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Mar 19, 2026</td><td>28.39</td></tr>
            <tr><td>Feb 1, 2026</td><td>27.10</td></tr>
          </tbody>
        </table>
        """
    )

    monkeypatch.setattr(
        "app.data.sources.us_market_client.requests.get",
        lambda *args, **kwargs: _MockResponse(html),
    )

    frame = fetch_us_market_series("equity_pe_proxy", country="us", frequency="monthly")

    assert list(frame["series_id"].unique()) == ["equity_pe_proxy"]
    assert frame["date"].iloc[-1].strftime("%Y-%m-%d") == "2026-03-01"
    assert frame["value"].iloc[-1] == 28.39
    assert frame["source"].iloc[-1] == "multpl"


def test_fetch_us_market_series_forward_fills_quarterly_pb(monkeypatch) -> None:
    """Quarterly Multpl tables should expand into monthly as-of rows."""
    html = dedent(
        """
        <table>
          <thead><tr><th>Date</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Mar 19, 2026</td><td>5.10</td></tr>
            <tr><td>Dec 31, 2025</td><td>4.80</td></tr>
          </tbody>
        </table>
        """
    )

    monkeypatch.setattr(
        "app.data.sources.us_market_client.requests.get",
        lambda *args, **kwargs: _MockResponse(html),
    )

    frame = fetch_us_market_series("equity_pb_proxy", country="us", frequency="monthly")

    assert frame["date"].iloc[0].day == 1
    assert "2026-01-01" in frame["date"].dt.strftime("%Y-%m-%d").tolist()
    assert frame.loc[frame["date"].dt.strftime("%Y-%m-%d") == "2026-01-01", "value"].iloc[0] == 4.8
    assert frame.loc[frame["date"].dt.strftime("%Y-%m-%d") == "2026-03-01", "value"].iloc[0] == 5.1
