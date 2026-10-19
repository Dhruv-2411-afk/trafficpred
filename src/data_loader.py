"""Streaming parser for headerless Caltrans PeMS station 5-minute files."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import pandas as pd

# Verified against supplied d07_text_station_5min_2026_01_01.txt.gz (52 fields).
BASE_COLUMNS = ["timestamp", "station_id", "district", "freeway", "direction",
                "lane_type", "station_length", "samples", "percent_observed",
                "flow", "avg_occupancy", "avg_speed"]
LANE_COLUMNS = [item for lane in range(1, 9) for item in
                (f"lane_{lane}_samples", f"lane_{lane}_flow", f"lane_{lane}_occupancy", f"lane_{lane}_speed", f"lane_{lane}_observed")]
COLUMNS = BASE_COLUMNS + LANE_COLUMNS

def raw_files(raw_dir: str | Path) -> list[Path]:
    return sorted(Path(raw_dir).glob("d*_text_station_5min_*.txt.gz"))

def _clean(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.iloc[:, :len(COLUMNS)].copy()
    chunk.columns = COLUMNS[:len(chunk.columns)]
    if len(chunk.columns) < 12:
        return pd.DataFrame(columns=BASE_COLUMNS)
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    for col in ["station_id", "district", "flow", "avg_occupancy", "avg_speed", "percent_observed"]:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    return chunk.dropna(subset=["timestamp", "station_id"]).query("flow >= 0 and avg_speed >= 0")

def read_pems(raw_dir: str | Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield validated chunks without loading all input files into memory."""
    files = raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No PeMS .txt.gz files found under {raw_dir}")
    for path in files:
        for chunk in pd.read_csv(path, header=None, chunksize=chunksize, dtype=str, on_bad_lines="skip"):
            cleaned = _clean(chunk)
            if not cleaned.empty:
                yield cleaned

def load_station_data(raw_dir: str | Path, chunksize: int = 200_000) -> pd.DataFrame:
    frames = list(read_pems(raw_dir, chunksize))
    if not frames:
        raise ValueError("No valid records after PeMS validation.")
    return pd.concat(frames, ignore_index=True).sort_values(["station_id", "timestamp"])
