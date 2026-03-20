"""Automatic consensus-note fetching from public official and institutional sources."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from app.consensus.sources import CONSENSUS_NOTE_COLUMNS, ingest_consensus_notes

DEFAULT_TIMEOUT_SECONDS = 20
AUTO_CONSENSUS_SUBDIR = "auto"
PBOC_PRESS_INDEX_URL = "https://www.pbc.gov.cn/en/3688110/3688172/index.html"


@dataclass(frozen=True)
class ConsensusSource:
    """Configuration for one automatic consensus source."""

    key: str
    region: str
    source_name: str
    source_type: str
    fetch_type: str
    url: str
    max_items: int = 8
    include_keywords: tuple[str, ...] = ()


CONSENSUS_SOURCE_REGISTRY: dict[str, list[ConsensusSource]] = {
    "us": [
        ConsensusSource(
            key="fed_monetary_press",
            region="us",
            source_name="Federal Reserve",
            source_type="official",
            fetch_type="rss",
            url="https://www.federalreserve.gov/feeds/press_monetary.xml",
            max_items=10,
        ),
        ConsensusSource(
            key="fed_speeches",
            region="us",
            source_name="Federal Reserve",
            source_type="official",
            fetch_type="rss",
            url="https://www.federalreserve.gov/feeds/speeches.xml",
            max_items=12,
        ),
        ConsensusSource(
            key="bis_us",
            region="us",
            source_name="BIS",
            source_type="institution",
            fetch_type="rss",
            url="https://www.bis.org/doclist/cbspeeches.rss",
            max_items=12,
            include_keywords=("federal reserve", "fomc", "jerome powell", "u.s. economy", "united states"),
        ),
    ],
    "eurozone": [
        ConsensusSource(
            key="ecb_press",
            region="eurozone",
            source_name="ECB",
            source_type="official",
            fetch_type="rss",
            url="https://www.ecb.europa.eu/rss/press.html",
            max_items=12,
        ),
        ConsensusSource(
            key="ecb_statpress",
            region="eurozone",
            source_name="ECB",
            source_type="official",
            fetch_type="rss",
            url="https://www.ecb.europa.eu/rss/statpress.html",
            max_items=8,
        ),
        ConsensusSource(
            key="bis_eurozone",
            region="eurozone",
            source_name="BIS",
            source_type="institution",
            fetch_type="rss",
            url="https://www.bis.org/doclist/cbspeeches.rss",
            max_items=12,
            include_keywords=(
                "ecb",
                "euro area",
                "eurozone",
                "lagarde",
                "eurosystem",
                "bank of france",
                "deutsche bundesbank",
                "de nederlandsche bank",
                "banca d'italia",
            ),
        ),
    ],
    "china": [
        ConsensusSource(
            key="pboc_press",
            region="china",
            source_name="PBOC",
            source_type="official",
            fetch_type="pboc_press",
            url=PBOC_PRESS_INDEX_URL,
            max_items=12,
        ),
        ConsensusSource(
            key="bis_china",
            region="china",
            source_name="BIS",
            source_type="institution",
            fetch_type="rss",
            url="https://www.bis.org/doclist/cbspeeches.rss",
            max_items=12,
            include_keywords=("china", "people's bank of china", "pboc", "renminbi"),
        ),
    ],
}


def _fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Fetch one URL and return decoded text."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or response.apparent_encoding or "utf-8"
    return response.text


def _strip_html(raw_html: str) -> str:
    """Convert simple HTML fragments to normalized plain text."""
    text = re.sub(r"<script.*?>.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_rss_items(xml_text: str) -> list[dict[str, Any]]:
    """Parse a basic RSS payload into item dictionaries."""
    normalized_xml = xml_text.lstrip("\ufeff").lstrip("ï»¿").strip()
    root = ET.fromstring(normalized_xml)
    namespaces = {
        "rss": "http://purl.org/rss/1.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    items: list[dict[str, Any]] = []
    xml_items = root.findall(".//item")
    if not xml_items:
        xml_items = root.findall("rss:item", namespaces)
    for item in xml_items:
        title = item.findtext("title") or item.findtext("rss:title", default="", namespaces=namespaces)
        description = (
            item.findtext("description")
            or item.findtext("rss:description", default="", namespaces=namespaces)
            or item.findtext("content:encoded")
            or ""
        )
        published = item.findtext("pubDate") or item.findtext("dc:date", default="", namespaces=namespaces)
        link = item.findtext("link") or item.findtext("rss:link", default="", namespaces=namespaces)
        items.append(
            {
                "title": str(title).strip(),
                "body": _strip_html(str(description)),
                "date": pd.to_datetime(published, errors="coerce"),
                "url": str(link).strip(),
            }
        )
    return items


def _filter_items(items: list[dict[str, Any]], include_keywords: tuple[str, ...], max_items: int) -> list[dict[str, Any]]:
    """Keep the most recent relevant items for one source."""
    filtered: list[dict[str, Any]] = []
    normalized_keywords = tuple(keyword.lower() for keyword in include_keywords)
    for item in items:
        title = str(item.get("title", ""))
        body = str(item.get("body", ""))
        text = f"{title} {body}".lower()
        if normalized_keywords and not any(keyword in text for keyword in normalized_keywords):
            continue
        if pd.isna(item.get("date")) or not title:
            continue
        filtered.append(item)
    filtered.sort(key=lambda item: pd.Timestamp(item["date"]), reverse=True)
    return filtered[:max_items]


def _rss_records(source: ConsensusSource) -> list[dict[str, Any]]:
    """Fetch and normalize one RSS feed into note-like records."""
    items = _parse_rss_items(_fetch_text(source.url))
    filtered = _filter_items(items, include_keywords=source.include_keywords, max_items=source.max_items)
    records: list[dict[str, Any]] = []
    for item in filtered:
        body = str(item.get("body", "")).strip()
        if item.get("url"):
            body = f"{body}\n\nSource URL: {item['url']}".strip()
        records.append(
            {
                "source_name": source.source_name,
                "source_type": source.source_type,
                "date": pd.Timestamp(item["date"]).date().isoformat(),
                "title": str(item.get("title", "")).strip(),
                "body": body,
            }
        )
    return records


def _extract_year_links(index_html: str, base_url: str) -> list[tuple[int, str]]:
    """Extract candidate PBOC year pages from the press-release index."""
    matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>\s*(20\d{2})\s*</a>', index_html, flags=re.IGNORECASE)
    links: list[tuple[int, str]] = []
    for href, year_text in matches:
        year = int(year_text)
        if year < 2020:
            continue
        links.append((year, urljoin(base_url, href)))
    unique = {(year, link) for year, link in links}
    return sorted(unique, key=lambda item: item[0], reverse=True)


def _extract_pboc_article_rows(year_html: str, base_url: str) -> list[tuple[pd.Timestamp, str, str]]:
    """Extract dated article rows from a PBOC yearly press-release page."""
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows: list[tuple[pd.Timestamp, str, str]] = []
    for date_text, href, title_html in pattern.findall(year_html):
        title = _strip_html(title_html)
        date_value = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(date_value) or not title:
            continue
        rows.append((pd.Timestamp(date_value), title, urljoin(base_url, href)))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def _extract_pboc_article_body(article_html: str) -> str:
    """Extract the main article body from a PBOC press-release page."""
    cleaned = article_html
    article_match = re.search(
        r"(<h2[^>]*>.*?)(?:Contact Us|The English translation may only be used as a reference)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet = article_match.group(1) if article_match else cleaned
    text = _strip_html(snippet)
    text = re.sub(r"Date of last update.*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _pboc_press_records(source: ConsensusSource) -> list[dict[str, Any]]:
    """Fetch recent PBOC English press releases and convert them into notes."""
    index_html = _fetch_text(source.url)
    year_links = _extract_year_links(index_html=index_html, base_url=source.url)
    if not year_links:
        return []

    records: list[dict[str, Any]] = []
    for _, year_url in year_links[:2]:
        year_html = _fetch_text(year_url)
        for article_date, article_title, article_url in _extract_pboc_article_rows(year_html, base_url=year_url):
            if len(records) >= source.max_items:
                break
            article_body = _extract_pboc_article_body(_fetch_text(article_url))
            records.append(
                {
                    "source_name": source.source_name,
                    "source_type": source.source_type,
                    "date": article_date.date().isoformat(),
                    "title": article_title,
                    "body": f"{article_body}\n\nSource URL: {article_url}".strip(),
                }
            )
        if len(records) >= source.max_items:
            break
    return records


def fetch_consensus_source_records(region: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Fetch all automatic consensus sources for one region."""
    normalized_region = str(region).strip().lower()
    sources = CONSENSUS_SOURCE_REGISTRY.get(normalized_region, [])
    payloads: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for source in sources:
        try:
            if source.fetch_type == "rss":
                payloads[source.key] = _rss_records(source)
            elif source.fetch_type == "pboc_press":
                payloads[source.key] = _pboc_press_records(source)
            else:
                payloads[source.key] = []
        except Exception as exc:  # pragma: no cover - network/runtime protection
            payloads[source.key] = []
            errors[source.key] = str(exc)
    return payloads, errors


def write_consensus_source_payloads(
    region: str,
    payloads: dict[str, list[dict[str, Any]]],
    raw_dir: str = "data/raw/consensus",
) -> list[Path]:
    """Write fetched note payloads into the raw consensus folder."""
    target_dir = Path(raw_dir) / region / AUTO_CONSENSUS_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for source_key, records in payloads.items():
        destination = target_dir / f"{source_key}.json"
        destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        written_paths.append(destination)
    return written_paths


def _summary_rows(region: str, payloads: dict[str, list[dict[str, Any]]], errors: dict[str, str]) -> pd.DataFrame:
    """Build a small fetch summary table for CLI diagnostics."""
    rows: list[dict[str, Any]] = []
    for source_key, records in payloads.items():
        dates = [pd.to_datetime(record.get("date"), errors="coerce") for record in records]
        valid_dates = [date for date in dates if pd.notna(date)]
        rows.append(
            {
                "region": region,
                "source_key": source_key,
                "note_count": len(records),
                "latest_date": max(valid_dates) if valid_dates else pd.NaT,
                "error": errors.get(source_key, ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_and_ingest_consensus_sources(
    region: str,
    raw_dir: str = "data/raw/consensus",
    output_path: str | Path = "data/processed/consensus_notes.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch automatic consensus sources for one region and ingest them into normalized storage."""
    payloads, errors = fetch_consensus_source_records(region=region)
    write_consensus_source_payloads(region=region, payloads=payloads, raw_dir=raw_dir)
    auto_dir = Path(raw_dir) / region / AUTO_CONSENSUS_SUBDIR
    notes = ingest_consensus_notes(region=region, path=str(auto_dir), output_path=output_path)
    summary = _summary_rows(region=region, payloads=payloads, errors=errors)
    return notes.loc[notes["region"] == region].reset_index(drop=True), summary


def consensus_notes_available(frame: pd.DataFrame) -> bool:
    """Return whether a fetched/ingested notes frame contains usable rows."""
    return not frame.empty and set(CONSENSUS_NOTE_COLUMNS).issubset(frame.columns)
