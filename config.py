"""Shared configuration for the VOICE decoder, build script and web API."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# 6 x 6 speller grid (same layout as the notebook).
GRID: list[list[str]] = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ0123"),
    list("456789"),
]
CHARACTERS: list[str] = [c for row in GRID for c in row]

# BNCI2014-008 "Flash stim" codes: 3-8 = rows, 9-14 = columns.
FLASH_CODES: list[int] = list(range(3, 15))
ROW_CODES: list[int] = list(range(3, 9))
COLUMN_CODES: list[int] = list(range(9, 15))

# Recording structure of BNCI2014-008.
SUBJECTS: list[int] = list(range(1, 9))
TRIALS_PER_SUBJECT = 35
REPETITIONS_PER_TRIAL = 10
EVENTS_PER_TRIAL = len(FLASH_CODES) * REPETITIONS_PER_TRIAL  # 120

# Preprocessing (matches notebook Module 2: MOABB P300 paradigm).
SAMPLING_RATE = 256.0
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 10.0
EPOCH_START_S = -0.2
EPOCH_END_S = 0.8
DECIMATION = 4            # 256 Hz -> 64 Hz (safe after a 10 Hz low-pass)
SCALE_TO_MICROVOLTS = 1e6  # MOABB returns volts

# First N trials of each subject are calibration data; the rest are spelled.
CALIBRATION_TRIALS = 10

# Adaptive stopping (matches notebook Module 6 VOICE_CONFIG).
STOPPING = {
    "minimum_repetitions": 3,
    "maximum_repetitions": REPETITIONS_PER_TRIAL,
    "confidence_threshold": 0.75,
    "uncertainty_threshold": 0.45,
    "stability_window": 2,
}

MAX_ALTERNATIVES = 3  # next-best characters offered after a rejection

# Runtime locations (override with environment variables on Render).
ARTIFACT_DIR = Path(os.environ.get("VOICE_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
SESSION_DIR = Path(os.environ.get("VOICE_SESSION_DIR", ROOT_DIR / "data" / "sessions"))


def code_to_grid_position(row_code: int, column_code: int) -> tuple[int, int]:
    return ROW_CODES.index(int(row_code)), COLUMN_CODES.index(int(column_code))


def pair_to_character(row_code: int, column_code: int) -> str:
    r, c = code_to_grid_position(row_code, column_code)
    return GRID[r][c]
