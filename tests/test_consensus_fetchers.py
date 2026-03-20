"""Tests for automatic consensus-source fetching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.consensus.fetchers import (
    ConsensusSource,
    _pboc_press_records,
    _rss_records,
    fetch_and_ingest_consensus_sources,
)


class _FakeResponse:
    """Small fake requests response for source-fetching tests."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self) -> None:
        return None


def test_rss_source_normalizes_into_note_records(monkeypatch) -> None:
    """RSS items should normalize into region note records."""
    xml_payload = """
    <rss><channel>
      <item>
        <title>Fed sees resilient growth and easing inflation</title>
        <link>https://www.federalreserve.gov/example</link>
        <description><![CDATA[Growth remains resilient and inflation is easing.]]></description>
        <pubDate>Thu, 19 Mar 2026 12:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """

    monkeypatch.setattr(
        "app.consensus.fetchers.requests.get",
        lambda url, timeout=20: _FakeResponse(xml_payload),
    )

    records = _rss_records(
        ConsensusSource(
            key="fed_test",
            region="us",
            source_name="Federal Reserve",
            source_type="official",
            fetch_type="rss",
            url="https://www.federalreserve.gov/feeds/test.xml",
        )
    )

    assert len(records) == 1
    assert records[0]["source_name"] == "Federal Reserve"
    assert records[0]["date"] == "2026-03-19"
    assert "Source URL" in records[0]["body"]


def test_rdf_rss_source_normalizes_with_namespaces(monkeypatch) -> None:
    """Namespace-based RSS feeds such as BIS should still parse into notes."""
    xml_payload = """<?xml version="1.0" encoding="utf-8"?>
    <rdf:RDF xmlns="http://purl.org/rss/1.0/"
             xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel rdf:about="https://www.bis.org/doclist/cbspeeches.rss">
        <title>Central bankers' speeches</title>
      </channel>
      <item rdf:about="https://www.bis.org/review/example.htm">
        <title>Speech on the euro area outlook</title>
        <link>https://www.bis.org/review/example.htm</link>
        <description>Speech by ECB official on euro area growth and inflation.</description>
        <dc:date>2026-03-18T09:49:00Z</dc:date>
      </item>
    </rdf:RDF>
    """

    monkeypatch.setattr(
        "app.consensus.fetchers.requests.get",
        lambda url, timeout=20: _FakeResponse(xml_payload),
    )

    records = _rss_records(
        ConsensusSource(
            key="bis_test",
            region="eurozone",
            source_name="BIS",
            source_type="institution",
            fetch_type="rss",
            url="https://www.bis.org/doclist/cbspeeches.rss",
            include_keywords=("euro area",),
        )
    )

    assert len(records) == 1
    assert records[0]["source_name"] == "BIS"
    assert records[0]["date"] == "2026-03-18"


def test_pboc_press_index_parses_recent_articles(monkeypatch) -> None:
    """PBOC press-release index pages should produce note records."""
    responses = {
        "https://www.pbc.gov.cn/en/3688110/3688172/index.html": """
            <html><body><a href="/en/3688110/3688172/5552468/index.html">2026</a></body></html>
        """,
        "https://www.pbc.gov.cn/en/3688110/3688172/5552468/index.html": """
            <html><body>
            2026-03-18 <a href="/en/3688110/3688172/5552468/6000000/index.html">PBOC meeting signals supportive policy</a>
            </body></html>
        """,
        "https://www.pbc.gov.cn/en/3688110/3688172/5552468/6000000/index.html": """
            <html><body><h2>PBOC meeting signals supportive policy</h2>
            <p>Domestic demand remains weak, inflation is easing, and liquidity will remain ample.</p>
            Contact Us
            </body></html>
        """,
    }

    monkeypatch.setattr(
        "app.consensus.fetchers.requests.get",
        lambda url, timeout=20: _FakeResponse(responses[url]),
    )

    records = _pboc_press_records(
        ConsensusSource(
            key="pboc_press",
            region="china",
            source_name="PBOC",
            source_type="official",
            fetch_type="pboc_press",
            url="https://www.pbc.gov.cn/en/3688110/3688172/index.html",
        )
    )

    assert len(records) == 1
    assert records[0]["source_name"] == "PBOC"
    assert records[0]["date"] == "2026-03-18"
    assert "supportive policy" in records[0]["title"].lower()


def test_fetch_and_ingest_sources_writes_processed_notes(tmp_path: Path, monkeypatch) -> None:
    """Automatic source fetching should write raw payloads and processed normalized notes."""
    xml_payload = """
    <rss><channel>
      <item>
        <title>ECB sees disinflation continuing</title>
        <link>https://www.ecb.europa.eu/example</link>
        <description><![CDATA[Disinflation continues while growth stabilises.]]></description>
        <pubDate>Fri, 20 Mar 2026 08:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """
    monkeypatch.setattr(
        "app.consensus.fetchers.requests.get",
        lambda url, timeout=20: _FakeResponse(xml_payload),
    )
    monkeypatch.setattr(
        "app.consensus.fetchers.CONSENSUS_SOURCE_REGISTRY",
        {
            "eurozone": [
                ConsensusSource(
                    key="ecb_test",
                    region="eurozone",
                    source_name="ECB",
                    source_type="official",
                    fetch_type="rss",
                    url="https://www.ecb.europa.eu/rss/test.html",
                )
            ]
        },
    )

    notes, summary = fetch_and_ingest_consensus_sources(
        region="eurozone",
        raw_dir=str(tmp_path / "raw"),
        output_path=tmp_path / "processed" / "consensus_notes.csv",
    )

    assert len(summary) == 1
    assert summary.loc[0, "note_count"] == 1
    assert not notes.empty
    assert set(notes["region"]) == {"eurozone"}
    assert (tmp_path / "raw" / "eurozone" / "auto" / "ecb_test.json").exists()
    saved = pd.read_csv(tmp_path / "processed" / "consensus_notes.csv")
    assert saved.loc[0, "source_name"] == "ECB"
