"""Speller sessions: message editing, trial decoding, accept / reject, reports."""

from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .artifacts import ArtifactStore
from .decoder import TrialDecoder, ranked_characters


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
