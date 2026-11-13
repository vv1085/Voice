"""VOICE core: configuration, P300 decoder, model files and speller sessions."""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__version__ = "1.0.0"

# `config.X` refers to the constants defined in this module.
config = sys.modules[__name__]

# ==========================================================================
# CONFIG
# ==========================================================================
ROOT_DIR = Path(__file__).resolve().parent

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


# ==========================================================================
# DECODER
# ==========================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic


@dataclass
class LinearP300Model:
    """Binary target / non-target classifier: p = sigmoid(w . x + b)."""

    weights: np.ndarray            # (n_channels * n_times,)
    bias: float
    n_channels: int
    n_times: int

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_times

    def features(self, epochs: np.ndarray) -> np.ndarray:
        epochs = np.asarray(epochs, dtype=np.float64)
        if epochs.ndim == 2:
            epochs = epochs[None]
        if epochs.ndim != 3 or epochs.shape[1:] != (self.n_channels, self.n_times):
            raise ValueError(
                f"Expected epochs shaped (n, {self.n_channels}, {self.n_times}); "
                f"got {tuple(epochs.shape)}."
            )
        if not np.all(np.isfinite(epochs)):
            raise ValueError("EEG epochs contain NaN or infinite values.")
        return epochs.reshape(len(epochs), -1)

    def predict_proba(self, epochs: np.ndarray) -> np.ndarray:
        return _sigmoid(self.features(epochs) @ self.weights + self.bias)


class EvidenceAccumulator:
    """Accumulates log-odds evidence per flash code across repetitions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.evidence = {code: 0.0 for code in config.FLASH_CODES}
        self.repetitions = 0
        self.previous_pair: tuple[int, int] | None = None
        self.stability = 0

    def update(self, flash_probabilities: dict[int, float]) -> dict:
        if set(flash_probabilities) != set(config.FLASH_CODES):
            raise ValueError("Each repetition must contain all 12 flash codes (3-14).")
        for code, p in flash_probabilities.items():
            p = float(np.clip(p, 1e-5, 1 - 1e-5))
            self.evidence[code] += float(np.log(p / (1 - p)))
        self.repetitions += 1
        return self.state()

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        e = np.exp(scores - scores.max())
        return e / e.sum()

    @staticmethod
    def _normalized_entropy(p: np.ndarray) -> float:
        h = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
        return float(np.clip(h, 0.0, 1.0))

    def state(self) -> dict:
        row_p = self._softmax(np.array([self.evidence[c] for c in config.ROW_CODES]))
        col_p = self._softmax(np.array([self.evidence[c] for c in config.COLUMN_CODES]))
        r, c = int(np.argmax(row_p)), int(np.argmax(col_p))
        pair = (config.ROW_CODES[r], config.COLUMN_CODES[c])

        self.stability = self.stability + 1 if pair == self.previous_pair else 1
        self.previous_pair = pair

        return {
            "pair": list(pair),
            "character": config.GRID[r][c],
            "confidence": float(min(row_p.max(), col_p.max())),
            "uncertainty": (self._normalized_entropy(row_p) + self._normalized_entropy(col_p)) / 2,
            "row_probabilities": row_p.round(6).tolist(),
            "column_probabilities": col_p.round(6).tolist(),
            "repetitions": self.repetitions,
            "stability": self.stability,
        }


@dataclass
class AdaptiveStopping:
    minimum_repetitions: int = config.STOPPING["minimum_repetitions"]
    maximum_repetitions: int = config.STOPPING["maximum_repetitions"]
    confidence_threshold: float = config.STOPPING["confidence_threshold"]
    uncertainty_threshold: float = config.STOPPING["uncertainty_threshold"]
    stability_window: int = config.STOPPING["stability_window"]

    def should_stop(self, state: dict) -> tuple[bool, str]:
        reps = state["repetitions"]
        if reps >= self.maximum_repetitions:
            return True, "MAXIMUM_REPETITIONS"
        if reps < self.minimum_repetitions:
            return False, "MINIMUM_REPETITIONS_NOT_REACHED"
        if state["confidence"] < self.confidence_threshold:
            return False, "LOW_CONFIDENCE"
        if state["uncertainty"] > self.uncertainty_threshold:
            return False, "HIGH_UNCERTAINTY"
        if state["stability"] < self.stability_window:
            return False, "PAIR_NOT_STABLE"
        return True, "ADAPTIVE_CONFIDENCE_STOP"


def ranked_characters(state: dict, limit: int) -> list[tuple[str, float]]:
    """Characters ranked by joint row x column probability."""
    joint = np.outer(state["row_probabilities"], state["column_probabilities"])
    order = np.argsort(-joint, axis=None)[:limit]
    rows, cols = np.unravel_index(order, joint.shape)
    return [(config.GRID[r][c], float(joint[r, c])) for r, c in zip(rows, cols)]


@dataclass
class TrialDecoder:
    """Runs one trial repetition by repetition until the stopping rule fires."""

    model: LinearP300Model
    stopping: AdaptiveStopping = field(default_factory=AdaptiveStopping)

    def repetition_probabilities(self, epochs: np.ndarray, flash_codes) -> dict[int, float]:
        flash_codes = [int(c) for c in np.asarray(flash_codes).ravel()]
        if len(flash_codes) != 12 or sorted(flash_codes) != config.FLASH_CODES:
            raise ValueError("A repetition needs 12 epochs with flash codes 3-14 exactly once.")
        epochs = np.asarray(epochs)
        if len(epochs) != 12:
            raise ValueError("A repetition needs exactly 12 epochs.")
        probs = self.model.predict_proba(epochs)
        return {code: float(p) for code, p in zip(flash_codes, probs)}

    def decode(self, repetitions: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        accumulator = EvidenceAccumulator()
        trace = []
        state, stop, reason = None, False, "NO_DATA"
        for epochs, codes in repetitions:
            probs = self.repetition_probabilities(epochs, codes)
            state = accumulator.update(probs)
            stop, reason = self.stopping.should_stop(state)
            trace.append({
                "flash_order": [int(c) for c in codes],
                "flash_probabilities": {str(k): round(v, 5) for k, v in probs.items()},
                "character": state["character"],
                "confidence": state["confidence"],
                "uncertainty": state["uncertainty"],
                "row_probabilities": state["row_probabilities"],
                "column_probabilities": state["column_probabilities"],
                "stop": stop,
                "reason": reason,
            })
            if stop:
                break
        if state is None:
            raise ValueError("No repetitions supplied.")
        return {"state": state, "stopped": stop, "reason": reason, "trace": trace}


# ==========================================================================
# ARTIFACTS
# ==========================================================================


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


# ==========================================================================
# SESSION
# ==========================================================================


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MessageEngine:
    """Composed message with undo / redo."""

    def __init__(self) -> None:
        self.text = ""
        self._undo: list[str] = []
        self._redo: list[str] = []
        self.history: list[dict] = []

    def _set(self, text: str, action: str, character: str | None = None, source: str = "USER") -> None:
        if text == self.text:
            return
        self._undo.append(self.text)
        self._redo.clear()
        self.text = text
        self.history.append({"time": _now(), "action": action, "character": character,
                             "source": source, "message": text})

    def add(self, character: str, source: str) -> None:
        if character not in config.CHARACTERS:
            raise ValueError(f"'{character}' is not on the speller grid.")
        self._set(self.text + character, "ADD", character, source)

    def apply(self, action: str) -> None:
        if action == "space":
            self._set(self.text + " ", "SPACE", " ")
        elif action == "backspace":
            self._set(self.text[:-1], "BACKSPACE")
        elif action == "clear":
            self._set("", "CLEAR")
        elif action == "undo":
            if self._undo:
                self._redo.append(self.text)
                self.text = self._undo.pop()
                self.history.append({"time": _now(), "action": "UNDO", "message": self.text})
        elif action == "redo":
            if self._redo:
                self._undo.append(self.text)
                self.text = self._redo.pop()
                self.history.append({"time": _now(), "action": "REDO", "message": self.text})
        else:
            raise ValueError(f"Unknown edit action '{action}'.")

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


class SpellerSession:
    """Replays a subject's held-out recorded trials through the adaptive decoder."""

    def __init__(self, store: ArtifactStore, subject: int) -> None:
        self.id = secrets.token_urlsafe(12)
        self.subject = int(subject)
        self.store = store
        self.trials = store.trials(self.subject)
        self.decoder = TrialDecoder(store.model(self.subject))
        self.message = MessageEngine()
        self.created = _now()
        self.finished_at: str | None = None
        self.lock = threading.Lock()

        self.position = 0                       # next held-out trial to decode
        self.candidate: str | None = None
        self.alternatives: list[tuple[str, float]] = []
        self.rejections = 0
        self.current: dict | None = None        # trial being confirmed
        self.trial_log: list[dict] = []
        self.events: list[dict] = [{"time": self.created, "event": "SESSION_STARTED"}]

    # -------------------------------------------------------------- state
    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    @property
    def awaiting_confirmation(self) -> bool:
        return self.candidate is not None

    def _require_open(self) -> None:
        if self.finished:
            raise RuntimeError("This session is finished. Start a new one.")

    def _log(self, event: str, **details) -> None:
        self.events.append({"time": _now(), "event": event, **details})

    # ------------------------------------------------------------ actions
    def decode_next(self) -> dict:
        self._require_open()
        if self.awaiting_confirmation:
            raise RuntimeError("Accept or reject the current candidate first.")
        if self.position >= len(self.trials):
            raise RuntimeError("All held-out trials for this subject have been decoded.")

        i = self.position
        self.position += 1
        result = self.decoder.decode(self.trials.repetitions(i))
        state = result["state"]

        self.alternatives = ranked_characters(state, config.MAX_ALTERNATIVES + 1)
        self.candidate = state["character"]
        self.rejections = 0
        self.current = {
            "trial": int(self.trials.trial_numbers[i]),
            "intended": self.trials.intended_character(i),
            "decoded": state["character"],
            "repetitions": state["repetitions"],
            "stopping_reason": result["reason"],
            "confidence": round(state["confidence"], 4),
            "uncertainty": round(state["uncertainty"], 4),
            "selected": None,
            "outcome": None,
        }
        self._log("TRIAL_DECODED", trial=self.current["trial"], decoded=self.candidate,
                  repetitions=state["repetitions"], reason=result["reason"])
        return {"trial": self.current, "trace": result["trace"]}

    def accept(self) -> str:
        self._require_open()
        if not self.awaiting_confirmation:
            raise RuntimeError("There is no candidate to accept.")
        character = self.candidate
        self.message.add(character, source="EEG DECODER")
        self._close_trial(character, "ACCEPTED")
        return character

    def reject(self) -> str | None:
        self._require_open()
        if not self.awaiting_confirmation:
            raise RuntimeError("There is no candidate to reject.")
        self._log("CANDIDATE_REJECTED", character=self.candidate)
        self.rejections += 1
        if self.rejections < len(self.alternatives):
            self.candidate = self.alternatives[self.rejections][0]
            return self.candidate
        self._close_trial(None, "SKIPPED")
        return None

    def _close_trial(self, selected: str | None, outcome: str) -> None:
        self.current.update(selected=selected, outcome=outcome, rejections=self.rejections)
        self.trial_log.append(self.current)
        self._log("TRIAL_" + outcome, trial=self.current["trial"], selected=selected)
        self.current, self.candidate, self.alternatives, self.rejections = None, None, [], 0

    def edit(self, action: str, character: str | None = None) -> None:
        self._require_open()
        if action == "add":
            if not character:
                raise ValueError("'add' needs a character.")
            self.message.add(character, source="MANUAL")
        else:
            self.message.apply(action)

    def finish(self) -> dict:
        if self.awaiting_confirmation:
            self._close_trial(None, "UNCONFIRMED")
        if not self.finished:
            self.finished_at = _now()
            self._log("SESSION_FINISHED")
        return self.report()

    # ------------------------------------------------------------ outputs
    def summary(self) -> dict:
        decoded = [t for t in self.trial_log if t["intended"] is not None]
        accepted = [t for t in self.trial_log if t["outcome"] == "ACCEPTED" and t["intended"]]
        return {
            "trials_decoded": len(self.trial_log) + (1 if self.current else 0),
            "trials_remaining": len(self.trials) - self.position,
            "first_guess_accuracy": (
                round(sum(t["decoded"] == t["intended"] for t in decoded) / len(decoded), 4)
                if decoded else None
            ),
            "accepted_accuracy": (
                round(sum(t["selected"] == t["intended"] for t in accepted) / len(accepted), 4)
                if accepted else None
            ),
            "mean_repetitions": (
                round(float(np.mean([t["repetitions"] for t in self.trial_log])), 2)
                if self.trial_log else None
            ),
        }

    def state(self) -> dict:
        return {
            "session_id": self.id,
            "subject": self.subject,
            "finished": self.finished,
            "message": self.message.text,
            "can_undo": self.message.can_undo,
            "can_redo": self.message.can_redo,
            "candidate": self.candidate,
            "alternatives": [{"character": c, "probability": round(p, 4)} for c, p in self.alternatives],
            "rejections": self.rejections,
            "current_trial": self.current,
            "trial_log": self.trial_log,
            "summary": self.summary(),
        }

    def report(self) -> dict:
        return {
            "project": "PROJECT VOICE",
            "session_id": self.id,
            "subject": self.subject,
            "data_source": self.store.source,
            "created": self.created,
            "finished": self.finished_at,
            "message": self.message.text,
            "summary": self.summary(),
            "trials": self.trial_log,
            "events": self.events,
            "message_history": self.message.history,
            "stopping_rule": config.STOPPING,
        }

    def save(self, directory: Path = config.SESSION_DIR) -> Path | None:
        """Best effort: Render's free disk is temporary, so the client also downloads it."""
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"VOICE_{self.id}.json"
            path.write_text(json.dumps(self.report(), indent=2))
            return path
        except OSError:
            return None
