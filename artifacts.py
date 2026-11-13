"""Reading and writing the files produced by scripts/build_artifacts.py.

artifacts/
  manifest.json        dataset source, preprocessing, per-subject evaluation
  models.npz           weights_<s>, bias_<s> for each subject's decoder
  subject_<s>.npz      held-out trials: epochs (µV, float16), flash_codes, labels
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .decoder import LinearP300Model

MANIFEST = "manifest.json"
MODELS = "models.npz"
FORMAT_VERSION = 1


def subject_file(subject: int) -> str:
    return f"subject_{int(subject)}.npz"


def artifacts_present(directory: Path) -> bool:
    directory = Path(directory)
    if not (directory / MANIFEST).is_file() or not (directory / MODELS).is_file():
        return False
    try:
        manifest = json.loads((directory / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("format_version") != FORMAT_VERSION:
        return False
    return all((directory / subject_file(s)).is_file() for s in manifest.get("subjects", {}))


@dataclass
class SubjectTrials:
    subject: int
    trial_numbers: np.ndarray     # (n_trials,) original trial index in the recording
    epochs: np.ndarray            # (n_trials, 120, n_channels, n_times) float16 µV
    flash_codes: np.ndarray       # (n_trials, 120)
    labels: np.ndarray            # (n_trials, 120) 1 = target flash

    def __len__(self) -> int:
        return len(self.trial_numbers)

    def repetitions(self, i: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split trial i into 10 repetitions: the r-th flash of every code, in time order."""
        codes = self.flash_codes[i].astype(int)
        positions = {c: np.flatnonzero(codes == c) for c in config.FLASH_CODES}
        n_reps = min(len(p) for p in positions.values())
        reps = []
        for r in range(n_reps):
            local = np.sort([positions[c][r] for c in config.FLASH_CODES])
            reps.append((self.epochs[i, local].astype(np.float32), codes[local]))
        return reps

    def intended_character(self, i: int) -> str | None:
        targets = set(self.flash_codes[i][self.labels[i] == 1].astype(int).tolist())
        rows = [c for c in targets if c in config.ROW_CODES]
        cols = [c for c in targets if c in config.COLUMN_CODES]
        if len(rows) == 1 and len(cols) == 1:
            return config.pair_to_character(rows[0], cols[0])
        return None


class ArtifactStore:
    """Loads the manifest and models eagerly and subject trials on first use."""

    def __init__(self, directory: Path = config.ARTIFACT_DIR) -> None:
        self.directory = Path(directory)
        if not artifacts_present(self.directory):
            raise FileNotFoundError(
                f"No VOICE artifacts in '{self.directory}'. "
                "Run: python scripts/build_artifacts.py"
            )
        self.manifest = json.loads((self.directory / MANIFEST).read_text())
        with np.load(self.directory / MODELS) as data:
            self.models = {
                int(s): LinearP300Model(
                    weights=data[f"weights_{s}"].astype(np.float64),
                    bias=float(data[f"bias_{s}"]),
                    n_channels=int(self.manifest["n_channels"]),
                    n_times=int(self.manifest["n_times"]),
                )
                for s in self.manifest["subjects"]
            }
        self._trials: dict[int, SubjectTrials] = {}
        self._lock = threading.Lock()

    @property
    def subjects(self) -> list[int]:
        return sorted(self.models)

    @property
    def source(self) -> str:
        return self.manifest["source"]

    def model(self, subject: int) -> LinearP300Model:
        if int(subject) not in self.models:
            raise KeyError(f"Unknown subject {subject}.")
        return self.models[int(subject)]

    def trials(self, subject: int) -> SubjectTrials:
        subject = int(subject)
        self.model(subject)
        with self._lock:
            if subject not in self._trials:
                with np.load(self.directory / subject_file(subject)) as d:
                    self._trials[subject] = SubjectTrials(
                        subject=subject,
                        trial_numbers=d["trial_numbers"].astype(int),
                        epochs=d["epochs"],
                        flash_codes=d["flash_codes"].astype(np.int16),
                        labels=d["labels"].astype(np.int8),
                    )
            return self._trials[subject]


def write_artifacts(
    directory: Path,
    manifest: dict,
    models: dict[int, tuple[np.ndarray, float]],
    trials: dict[int, dict[str, np.ndarray]],
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / MODELS,
        **{f"weights_{s}": w.astype(np.float64) for s, (w, _) in models.items()},
        **{f"bias_{s}": np.float64(b) for s, (_, b) in models.items()},
    )
    for s, d in trials.items():
        np.savez_compressed(
            directory / subject_file(s),
            trial_numbers=d["trial_numbers"].astype(np.int16),
            epochs=d["epochs"].astype(np.float16),
            flash_codes=d["flash_codes"].astype(np.int8),
            labels=d["labels"].astype(np.int8),
        )
    manifest = {**manifest, "format_version": FORMAT_VERSION}
    # Manifest last: its presence marks a complete build.
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2))
