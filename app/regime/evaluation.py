"""Descriptive regime evaluation helpers for the global macro monitor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.regime.change_detection import ALLOCATION_HISTORY_PATH, HISTORY_DIR, SUMMARY_HISTORY_PATH

FORWARD_WINDOWS = [1, 3, 6]
REGIME_FREQUENCY_COLUMNS = ["selected_mode", "state_type", "state", "count", "share"]
TRANSITION_MATRIX_COLUMNS = [
    "selected_mode",
    "from_regime",
    "to_regime",
    "count",
    "transition_probability",
]
FORWARD_RETURN_COLUMNS = [
    "selected_mode",
    "state_type",
    "state",
    "asset",
    "window_months",
    "count",
    "average_forward_return",
    "median_forward_return",
    "hit_ratio",
]
CONFIDENCE_BUCKET_COLUMNS = [
    "selected_mode",
    "asset",
    "confidence",
    "window_months",
    "count",
    "average_forward_return",
    "median_forward_return",
    "hit_ratio",
]


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with normalized time columns when it exists."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in ["date", "summary_date", "run_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _latest_by_summary_date(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Deduplicate history by keeping the latest run for each summary date."""
    required = ["selected_mode", "summary_date", "run_timestamp"] + value_columns
    available = [column for column in required if column in frame.columns]
    if len(available) < len(required):
        return pd.DataFrame(columns=required)
    valid = frame.dropna(subset=["selected_mode", "summary_date", "run_timestamp"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=required)
    valid = valid.sort_values(["selected_mode", "summary_date", "run_timestamp"])
    latest = valid.groupby(["selected_mode", "summary_date"], as_index=False).tail(1)
    return latest.reset_index(drop=True)


def _latest_allocation_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate allocation history by mode, summary date, and asset."""
    required = ["selected_mode", "summary_date", "run_timestamp", "asset", "confidence"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=required)
    valid = frame.dropna(subset=["selected_mode", "summary_date", "run_timestamp", "asset"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=required)
    valid = valid.sort_values(["selected_mode", "summary_date", "asset", "run_timestamp"])
    latest = valid.groupby(["selected_mode", "summary_date", "asset"], as_index=False).tail(1)
    return latest.reset_index(drop=True)


def _load_return_proxies(
    processed_dir: str = "data/processed",
    manual_dir: str = "data/raw/manual/returns",
) -> dict[str, pd.DataFrame]:
    """Load optional return proxy series from processed or manual folders."""
    candidates = [
        Path(processed_dir) / "returns",
        Path(manual_dir),
    ]
    proxies: dict[str, pd.DataFrame] = {}
    for folder in candidates:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            frame = _load_csv(path)
            if frame.empty or "value" not in frame.columns:
                continue
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["date", "value"]).sort_values("date")
            if frame.empty:
                continue
            proxies[path.stem] = frame.loc[:, ["date", "value"]].copy()
    return proxies


def _add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute forward returns over the standard research windows."""
    output = frame.copy()
    for window in FORWARD_WINDOWS:
        output[f"forward_{window}m"] = output["value"].shift(-window) / output["value"] - 1
    return output


def _summarize_forward_metric(
    merged: pd.DataFrame,
    group_column: str,
    group_value: str,
    asset: str,
    selected_mode: str,
) -> list[dict[str, object]]:
    """Summarize forward returns for one state and one asset."""
    subset = merged.loc[merged[group_column] == group_value]
    rows: list[dict[str, object]] = []
    for window in FORWARD_WINDOWS:
        column = f"forward_{window}m"
        series = pd.to_numeric(subset[column], errors="coerce").dropna()
        rows.append(
            {
                "selected_mode": selected_mode,
                "state_type": group_column,
                "state": group_value,
                "asset": asset,
                "window_months": window,
                "count": int(series.count()),
                "average_forward_return": series.mean() if not series.empty else pd.NA,
                "median_forward_return": series.median() if not series.empty else pd.NA,
                "hit_ratio": (series.gt(0).mean() if not series.empty else pd.NA),
            }
        )
    return rows


def compute_regime_frequency(summary_history: pd.DataFrame) -> pd.DataFrame:
    """Compute simple regime and investment-clock frequencies from summary history."""
    summary = _latest_by_summary_date(
        summary_history,
        value_columns=["global_regime", "investment_clock"],
    )
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame(columns=REGIME_FREQUENCY_COLUMNS)

    for selected_mode, mode_frame in summary.groupby("selected_mode"):
        total = len(mode_frame)
        for state_type in ["global_regime", "investment_clock"]:
            counts = mode_frame[state_type].value_counts(dropna=False)
            for state, count in counts.items():
                rows.append(
                    {
                        "selected_mode": selected_mode,
                        "state_type": state_type,
                        "state": state,
                        "count": int(count),
                        "share": float(count) / total if total else pd.NA,
                    }
                )
    return pd.DataFrame(rows, columns=REGIME_FREQUENCY_COLUMNS)


def compute_regime_transition_matrix(summary_history: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple global regime transition matrix."""
    summary = _latest_by_summary_date(summary_history, value_columns=["global_regime"])
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame(columns=TRANSITION_MATRIX_COLUMNS)

    for selected_mode, mode_frame in summary.groupby("selected_mode"):
        ordered = mode_frame.sort_values("summary_date").copy()
        ordered["from_regime"] = ordered["global_regime"].shift(1)
        transitions = ordered.dropna(subset=["from_regime", "global_regime"])
        if transitions.empty:
            continue
        counts = (
            transitions.groupby(["from_regime", "global_regime"])
            .size()
            .reset_index(name="count")
            .rename(columns={"global_regime": "to_regime"})
        )
        totals = counts.groupby("from_regime")["count"].transform("sum")
        counts["transition_probability"] = counts["count"] / totals
        counts.insert(0, "selected_mode", selected_mode)
        rows.extend(counts.to_dict(orient="records"))
    return pd.DataFrame(rows, columns=TRANSITION_MATRIX_COLUMNS)


def compute_forward_return_summary(
    summary_history: pd.DataFrame,
    return_proxies: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute forward return summaries by regime and investment clock."""
    summary = _latest_by_summary_date(
        summary_history,
        value_columns=["global_regime", "investment_clock"],
    )
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame(columns=FORWARD_RETURN_COLUMNS)

    if not return_proxies:
        return pd.DataFrame(columns=FORWARD_RETURN_COLUMNS)

    for selected_mode, mode_frame in summary.groupby("selected_mode"):
        for asset, proxy in return_proxies.items():
            merged = mode_frame.merge(
                _add_forward_returns(proxy),
                left_on="summary_date",
                right_on="date",
                how="left",
            )
            for state_type in ["global_regime", "investment_clock"]:
                for state in merged[state_type].dropna().unique().tolist():
                    rows.extend(
                        _summarize_forward_metric(
                            merged=merged,
                            group_column=state_type,
                            group_value=str(state),
                            asset=asset,
                            selected_mode=str(selected_mode),
                        )
                    )
    return pd.DataFrame(rows, columns=FORWARD_RETURN_COLUMNS)


def compute_confidence_bucket_summary(
    allocation_history: pd.DataFrame,
    return_proxies: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize forward returns by confidence bucket and asset."""
    allocation = _latest_allocation_history(allocation_history)
    rows: list[dict[str, object]] = []
    if allocation.empty or not return_proxies:
        return pd.DataFrame(columns=CONFIDENCE_BUCKET_COLUMNS)

    for selected_mode, mode_frame in allocation.groupby("selected_mode"):
        for asset, proxy in return_proxies.items():
            merged = mode_frame.merge(
                _add_forward_returns(proxy),
                left_on="summary_date",
                right_on="date",
                how="left",
            )
            for confidence in merged["confidence"].dropna().unique().tolist():
                subset = merged.loc[merged["confidence"] == confidence]
                for window in FORWARD_WINDOWS:
                    column = f"forward_{window}m"
                    series = pd.to_numeric(subset[column], errors="coerce").dropna()
                    rows.append(
                        {
                            "selected_mode": selected_mode,
                            "asset": asset,
                            "confidence": confidence,
                            "window_months": window,
                            "count": int(series.count()),
                            "average_forward_return": series.mean() if not series.empty else pd.NA,
                            "median_forward_return": series.median() if not series.empty else pd.NA,
                            "hit_ratio": (series.gt(0).mean() if not series.empty else pd.NA),
                        }
                    )
    return pd.DataFrame(rows, columns=CONFIDENCE_BUCKET_COLUMNS)


def build_regime_evaluation_outputs(
    processed_dir: str = "data/processed",
    manual_returns_dir: str = "data/raw/manual/returns",
    history_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all descriptive evaluation outputs and save them to CSV files."""
    processed = Path(processed_dir)
    history_root = Path(history_dir) if history_dir is not None else processed
    summary_history = _load_csv(history_root / Path(SUMMARY_HISTORY_PATH).name)
    allocation_history = _load_csv(history_root / Path(ALLOCATION_HISTORY_PATH).name)
    return_proxies = _load_return_proxies(
        processed_dir=processed_dir,
        manual_dir=manual_returns_dir,
    )

    frequency = compute_regime_frequency(summary_history)
    transitions = compute_regime_transition_matrix(summary_history)
    forward_summary = compute_forward_return_summary(summary_history, return_proxies)
    confidence_summary = compute_confidence_bucket_summary(allocation_history, return_proxies)

    outputs = {
        "regime_frequency_summary.csv": frequency,
        "regime_transition_matrix.csv": transitions,
        "regime_forward_return_summary.csv": forward_summary,
        "confidence_bucket_summary.csv": confidence_summary,
    }
    for filename, frame in outputs.items():
        path = processed / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return outputs
