"""Consensus note ingestion utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

CONSENSUS_NOTE_COLUMNS = [
    "region",
    "source_name",
    "source_type",
    "date",
    "title",
    "body",
    "note_id",
    "ingestion_timestamp",
]
CONSENSUS_NOTES_OUTPUT_PATH = Path("data/processed/consensus_notes.csv")
SUPPORTED_REGIONS = {"us", "eurozone", "china"}
SOURCE_TYPE_KEYWORDS = {
    "official": ["federal reserve", "ecb", "european central bank", "pboc", "people's bank", "ministry", "bureau"],
    "institution": ["goldman", "jpmorgan", "morgan stanley", "ubs", "nomura", "barclays", "citi", "bank of america"],
    "media": ["reuters", "bloomberg", "financial times", "wall street journal", "wsj", "cnbc"],
}


def _ingestion_timestamp() -> pd.Timestamp:
    """Return a timezone-aware ingestion timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _empty_notes_frame() -> pd.DataFrame:
    """Return an empty normalized notes frame."""
    return pd.DataFrame(columns=CONSENSUS_NOTE_COLUMNS)


def _infer_source_type(source_name: str, body: str = "") -> str:
    """Infer a coarse source type from source metadata."""
    text = f"{source_name} {body}".lower()
    for source_type, keywords in SOURCE_TYPE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return source_type
    return "other"


def _note_id(region: str, source_name: str, date: object, title: str, body: str) -> str:
    """Build a deterministic note id."""
    raw = f"{region}|{source_name}|{date}|{title}|{body}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse simple key-value metadata from the top of a text note."""
    metadata: dict[str, str] = {}
    lines = text.splitlines()
    body_start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            body_start = index + 1
            break
        if ":" not in stripped:
            body_start = index
            break
        key, value = stripped.split(":", maxsplit=1)
        metadata[key.strip().lower()] = value.strip()
        body_start = index + 1
    body = "\n".join(lines[body_start:]).strip()
    if not body:
        body = text.strip()
    return metadata, body


def _normalize_note_record(region: str, record: dict[str, object], fallback_name: str) -> dict[str, object]:
    """Normalize one raw record into the shared note schema."""
    source_name = str(record.get("source_name") or fallback_name).strip() or fallback_name
    body = str(record.get("body") or "").strip()
    title = str(record.get("title") or fallback_name).strip() or fallback_name
    raw_source_type = str(record.get("source_type") or "").strip().lower()
    source_type = raw_source_type or _infer_source_type(source_name=source_name, body=body)
    date_value = pd.to_datetime(record.get("date"), errors="coerce")
    if pd.isna(date_value):
        raise ValueError(f"Consensus note is missing a valid date: {fallback_name}")
    note_id = str(record.get("note_id") or _note_id(region, source_name, date_value.date(), title, body))
    return {
        "region": region,
        "source_name": source_name,
        "source_type": source_type,
        "date": date_value,
        "title": title,
        "body": body,
        "note_id": note_id,
        "ingestion_timestamp": _ingestion_timestamp(),
    }


def _load_text_note(region: str, path: Path) -> list[dict[str, object]]:
    """Load one markdown or text note file."""
    text = path.read_text(encoding="utf-8").strip()
    metadata, body = _parse_front_matter(text)
    fallback_name = path.stem.replace("_", " ").strip() or path.name
    record = {
        "source_name": metadata.get("source_name") or metadata.get("source") or fallback_name,
        "source_type": metadata.get("source_type"),
        "date": metadata.get("date"),
        "title": metadata.get("title") or fallback_name,
        "body": body,
    }
    return [_normalize_note_record(region=region, record=record, fallback_name=fallback_name)]


def _load_json_note(region: str, path: Path) -> list[dict[str, object]]:
    """Load one JSON note file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    fallback_name = path.stem.replace("_", " ").strip() or path.name
    records = payload if isinstance(payload, list) else [payload]
    return [
        _normalize_note_record(region=region, record=dict(record), fallback_name=fallback_name)
        for record in records
    ]


def load_consensus_notes_from_path(region: str, path: str) -> pd.DataFrame:
    """Load raw consensus notes from one file or folder path."""
    normalized_region = str(region).strip().lower()
    if normalized_region not in SUPPORTED_REGIONS:
        raise ValueError(f"Unsupported region: {region}")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Consensus note path does not exist: {target}")

    files = (
        sorted(item for item in target.rglob("*") if item.is_file())
        if target.is_dir()
        else [target]
    )
    records: list[dict[str, object]] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in {".md", ".txt", ".json"}:
            continue
        if suffix == ".json":
            records.extend(_load_json_note(normalized_region, file_path))
        else:
            records.extend(_load_text_note(normalized_region, file_path))

    if not records:
        return _empty_notes_frame()
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ingestion_timestamp"] = pd.to_datetime(frame["ingestion_timestamp"], errors="coerce")
    frame = frame.dropna(subset=["date", "body"])
    frame = frame.drop_duplicates(subset=["note_id"], keep="last")
    return frame.loc[:, CONSENSUS_NOTE_COLUMNS].sort_values(["region", "date", "note_id"]).reset_index(drop=True)


def save_consensus_notes(frame: pd.DataFrame, output_path: str | Path = CONSENSUS_NOTES_OUTPUT_PATH) -> Path:
    """Save normalized consensus notes to processed storage."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def ingest_consensus_notes(
    region: str,
    path: str,
    output_path: str | Path = CONSENSUS_NOTES_OUTPUT_PATH,
) -> pd.DataFrame:
    """Ingest consensus notes into normalized processed storage."""
    incoming = load_consensus_notes_from_path(region=region, path=path)
    destination = Path(output_path)
    if destination.exists():
        existing = pd.read_csv(destination)
        if not existing.empty:
            for column in ["date", "ingestion_timestamp"]:
                if column in existing.columns:
                    existing[column] = pd.to_datetime(existing[column], errors="coerce")
            incoming = pd.concat([existing, incoming], ignore_index=True)
    if incoming.empty:
        save_consensus_notes(_empty_notes_frame(), output_path=destination)
        return _empty_notes_frame()
    incoming = incoming.drop_duplicates(subset=["note_id"], keep="last")
    incoming = incoming.loc[:, CONSENSUS_NOTE_COLUMNS].sort_values(["region", "date", "note_id"]).reset_index(drop=True)
    save_consensus_notes(incoming, output_path=destination)
    return incoming

