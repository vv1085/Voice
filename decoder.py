"""P300 decoding: linear classifier, evidence accumulation and adaptive stopping.

Runtime code uses NumPy only. The build script trains a scikit-learn
StandardScaler + shrinkage-LDA pipeline and folds it into one weight
vector and bias, so the web server does not need scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


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
