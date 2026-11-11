"""VOICE backend: P300 speller classifier, evidence accumulation, session
management and the ipywidgets communication interface.

This module has no import-time side effects: it defines classes, functions
and configuration only. Training and launching are done by the caller (see
app.py).
"""

import json
import os
from datetime import datetime

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VOICE_GRID = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "1", "2", "3", "4"],
    ["5", "6", "7", "8", "9", "0"],
]
VOICE_FINAL_GRID = VOICE_GRID

# Output directory for exported sessions/messages (override with VOICE_OUTPUT_DIR).
VOICE_OUTPUT_DIR = os.environ.get("VOICE_OUTPUT_DIR", "voice_output")
VOICE_FINAL_DIR = VOICE_OUTPUT_DIR
os.makedirs(VOICE_OUTPUT_DIR, exist_ok=True)

# Flash codes 3-14: 3-8 flash the six rows, 9-14 flash the six columns.
# NOTE: stopping thresholds below are defaults; the original values were
# defined in a notebook session that is not part of this repository.
VOICE_CONFIG = {
    "flash_codes": list(range(3, 15)),
    "row_codes": list(range(3, 9)),
    "column_codes": list(range(9, 15)),
    "minimum_repetitions": 2,
    "maximum_repetitions": 10,
    "confidence_threshold": 0.80,
    "uncertainty_threshold": 0.50,
    "stability_window": 2,
    "session_directory": VOICE_OUTPUT_DIR,
}
VOICE_FINAL_CONFIG = VOICE_CONFIG
VOICE_APP_CONFIG = VOICE_CONFIG


def prepare_eeg_features(X):
    """Return a 2D (n_epochs, n_features) array; flattens (epochs, channels, times)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X.reshape(len(X), -1)


class VOICECommunicationInterface:

    def __init__(self):

        self.message = ""
        self.history = []
        self.selection_count = 0
        self.decoder_connected = False
        self.last_confidence = 0.0
        self.last_uncertainty = 1.0

        self.message_display = widgets.Textarea(
            value="",
            placeholder="Your message will appear here...",
            description="Message:",
            layout=widgets.Layout(width="100%", height="110px"),
            disabled=False
        )

        self.status = widgets.HTML(
            value="<b>VOICE:</b> Interface initialized."
        )

        self.confidence_display = widgets.HTML(
            value="Confidence: Awaiting decoder"
        )

        self.uncertainty_display = widgets.HTML(
            value="Uncertainty: Awaiting decoder"
        )

        self.history_display = widgets.Output()

        self.grid_buttons = []

        self.grid_widget = widgets.GridBox(
            children=[],
            layout=widgets.Layout(
                grid_template_columns="repeat(6, 1fr)",
                grid_gap="5px",
                width="100%"
            )
        )

        self._build_grid()

        # Controls
        self.undo_button = widgets.Button(
            description="Undo",
            button_style="warning",
            icon="undo"
        )

        self.space_button = widgets.Button(
            description="Space",
            button_style="info",
            icon="plus"
        )

        self.backspace_button = widgets.Button(
            description="Backspace",
            icon="backspace"
        )

        self.clear_button = widgets.Button(
            description="Clear",
            button_style="danger",
            icon="trash"
        )

        self.export_button = widgets.Button(
            description="Export Message",
            button_style="success",
            icon="download"
        )

        self.undo_button.on_click(self._undo)
        self.space_button.on_click(self._space)
        self.backspace_button.on_click(self._backspace)
        self.clear_button.on_click(self._clear)
        self.export_button.on_click(self._export)

        self.message_display.observe(
            self._message_changed,
            names="value"
        )

        self._render()

    # --------------------------------------------------------------------
    # 3. CHARACTER GRID
    # --------------------------------------------------------------------

    def _build_grid(self):

        for row in range(6):

            for col in range(6):

                character = VOICE_GRID[row][col]

                button = widgets.Button(
                    description=character,
                    tooltip=f"UI preview: {character}",
                    layout=widgets.Layout(
                        width="100%",
                        height="48px"
                    ),
                    style={"font_weight": "bold"}
                )

                button.on_click(
                    lambda b, c=character:
                    self._preview_character(c)
                )

                self.grid_buttons.append(button)

        self.grid_widget.children = self.grid_buttons

    # --------------------------------------------------------------------
    # 4. MESSAGE MANAGEMENT
    # --------------------------------------------------------------------

    def _message_changed(self, change):

        self.message = change["new"]

    def _add_character(self, character, source):

        if character not in [
            c for row in VOICE_GRID for c in row
        ]:
            raise ValueError("Invalid character.")

        self.message += character
        self.selection_count += 1

        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "character": character,
            "source": source,
            "message_length": len(self.message)
        })

        self.message_display.value = self.message

        self.status.value = (
            f"<b>VOICE:</b> Character {character} selected "
            f"via {source}."
        )

        self._update_history()

    def _preview_character(self, character):

        self._add_character(
            character,
            source="MANUAL UI PREVIEW"
        )

    def _space(self, button=None):

        self.message += " "
        self.message_display.value = self.message

        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "character": "SPACE",
            "source": "Interface control"
        })

        self._update_history()

    def _backspace(self, button=None):

        self.message = self.message[:-1]
        self.message_display.value = self.message

        self.status.value = "<b>VOICE:</b> Last character removed."

    def _undo(self, button=None):

        if not self.history:
            self.status.value = "<b>VOICE:</b> Nothing to undo."
            return

        previous = self.history.pop()

        if previous["character"] == "SPACE":
            self.message = self.message[:-1]
        elif previous["character"] in [
            c for row in VOICE_GRID for c in row
        ]:
            self.message = self.message[:-1]

        self.message_display.value = self.message

        self.status.value = "<b>VOICE:</b> Previous selection undone."

        self._update_history()

    def _clear(self, button=None):

        self.message = ""
        self.history = []

        self.message_display.value = ""

        self.status.value = "<b>VOICE:</b> Message cleared."

        self._update_history()

    # --------------------------------------------------------------------
    # 5. ADAPTIVE DECODER INTEGRATION
    # --------------------------------------------------------------------

    def connect_decoder(self):

        self.decoder_connected = True

        self.status.value = (
            "<b>VOICE:</b> Decoder integration endpoint enabled."
        )

    def submit_decoder_output(
        self,
        predicted_pair,
        confidence,
        uncertainty=0.0
    ):

        """
        Accepts a predicted row/column flash pair.

        Flash codes:
        1-6  = rows
        7-12 = columns

        The decoder must provide a complete valid pair.
        """

        if not self.decoder_connected:
            raise RuntimeError(
                "Connect the decoder before submitting predictions."
            )

        if len(predicted_pair) != 2:
            raise ValueError("A pair must contain two flash codes.")

        row_code, col_code = map(int, predicted_pair)

        if row_code not in range(1, 7):
            raise ValueError("Invalid row flash code.")

        if col_code not in range(7, 13):
            raise ValueError("Invalid column flash code.")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        if not 0 <= uncertainty <= 1:
            raise ValueError("Uncertainty must be between 0 and 1.")

        row_index = row_code - 1
        col_index = col_code - 7

        character = VOICE_GRID[row_index][col_index]

        self.last_confidence = confidence
        self.last_uncertainty = uncertainty

        self.confidence_display.value = (
            f"<b>Decoder confidence:</b> {confidence:.1%}"
        )

        self.uncertainty_display.value = (
            f"<b>Decoder uncertainty:</b> {uncertainty:.1%}"
        )

        self._add_character(
            character,
            source="EEG DECODER"
        )

        return character

    # --------------------------------------------------------------------
    # 6. HISTORY AND EXPORT
    # --------------------------------------------------------------------

    def _update_history(self):

        with self.history_display:

            self.history_display.clear_output(wait=True)

            print("SELECTION HISTORY")

            for item in self.history[-10:]:

                print(
                    item["timestamp"][11:19],
                    "|",
                    item["character"],
                    "|",
                    item["source"]
                )

    def _export(self, button=None):

        export_data = {
            "project": "VOICE",
            "message": self.message,
            "selection_count": self.selection_count,
            "history": self.history,
            "exported_at": datetime.now().isoformat()
        }

        path = os.path.join(
            VOICE_OUTPUT_DIR,
            "VOICE_MESSAGE.json"
        )

        with open(path, "w") as file:
            json.dump(export_data, file, indent=4)

        self.status.value = (
            f"<b>VOICE:</b> Message exported to {path}"
        )

    # --------------------------------------------------------------------
    # 7. RENDER APPLICATION
    # --------------------------------------------------------------------

    def _render(self):

        controls = widgets.HBox([
            self.undo_button,
            self.backspace_button,
            self.space_button,
            self.clear_button,
            self.export_button
        ])

        display(HTML("""
        <h2>VOICE | Adaptive Communication Interface</h2>
        <p>Neurotechnology communication application</p>
        <hr>
        """))

        display(self.status)

        display(HTML("<h3>Character Selection Grid</h3>"))
        display(self.grid_widget)

        display(HTML("<h3>Message Composer</h3>"))
        display(self.message_display)
        display(controls)

        display(HTML("<h3>Decoder Feedback</h3>"))
        display(self.confidence_display)
        display(self.uncertainty_display)

        display(HTML("<h3>Recent Selections</h3>"))
        display(self.history_display)


class VOICEP300Classifier:

    def __init__(self):

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            ))
        ])

        self.target_label = None
        self.non_target_label = None
        self.trained = False

        self.feature_count = None

    def fit(self, X, y):

        X_features = prepare_eeg_features(X)

        y = np.asarray(y).reshape(-1)

        if len(X_features) != len(y):

            raise ValueError(
                "X and y have different sample counts."
            )

        labels, counts = np.unique(
            y,
            return_counts=True
        )

        if len(labels) != 2:

            raise ValueError(
                "Binary P300 classification requires two classes."
            )

        # Identify target class.
        # For this dataset, target events are the minority class.

        if y.dtype.kind in ("U", "S", "O"):

            target_candidates = [
                label for label in labels
                if (
                    "target" in str(label).lower()
                    and "non" not in str(label).lower()
                )
            ]

            if target_candidates:

                self.target_label = target_candidates[0]

            else:

                self.target_label = labels[
                    np.argmin(counts)
                ]

        else:

            self.target_label = labels[
                np.argmin(counts)
            ]

        self.non_target_label = next(
            label for label in labels
            if label != self.target_label
        )

        y_binary = (
            y == self.target_label
        ).astype(int)

        self.model.fit(
            X_features,
            y_binary
        )

        self.feature_count = X_features.shape[1]

        self.trained = True

        print("\nP300 CLASSIFIER TRAINED")
        print("Training epochs:", len(X_features))
        print("Feature count:", self.feature_count)
        print("Target label:", self.target_label)
        print("Non-target label:", self.non_target_label)

        return self

    def predict_target_probability(self, X):

        if not self.trained:

            raise RuntimeError(
                "Train classifier before prediction."
            )

        X_features = prepare_eeg_features(X)

        if X_features.shape[1] != self.feature_count:

            raise ValueError(
                "Input feature dimensions do not match training."
            )

        probabilities = self.model.predict_proba(
            X_features
        )

        return probabilities[:, 1]

    def predict_target_logits(self, X):

        probabilities = self.predict_target_probability(X)

        probabilities = np.clip(
            probabilities,
            1e-6,
            1 - 1e-6
        )

        return np.log(
            probabilities / (1 - probabilities)
        )


class VOICEEvidenceAccumulator:

    def __init__(self):

        self.reset()

    def reset(self):

        self.evidence = {
            code: 0.0
            for code in VOICE_CONFIG["flash_codes"]
        }

        self.observations = {
            code: []
            for code in VOICE_CONFIG["flash_codes"]
        }

        self.repetition_count = 0

        self.previous_pair = None
        self.stable_count = 0

        self.pair_history = []

    def update(self, flash_probabilities):

        if set(flash_probabilities.keys()) != set(
            VOICE_CONFIG["flash_codes"]
        ):

            raise ValueError(
                "Each repetition must contain all 12 flash codes."
            )

        for code, probability in flash_probabilities.items():

            probability = float(probability)

            if not 0 <= probability <= 1:

                raise ValueError(
                    "Flash probabilities must be between 0 and 1."
                )

            probability = np.clip(
                probability,
                1e-5,
                1 - 1e-5
            )

            # Accumulate log-odds evidence.
            log_odds = np.log(
                probability / (1 - probability)
            )

            self.evidence[code] += log_odds

            self.observations[code].append(
                probability
            )

        self.repetition_count += 1

        return self.get_evidence_state()

    def get_evidence_state(self):

        row_codes = VOICE_CONFIG["row_codes"]
        column_codes = VOICE_CONFIG["column_codes"]

        row_scores = np.array([
            self.evidence[code]
            for code in row_codes
        ])

        column_scores = np.array([
            self.evidence[code]
            for code in column_codes
        ])

        row_probabilities = softmax(row_scores)
        column_probabilities = softmax(column_scores)

        row_index = int(
            np.argmax(row_probabilities)
        )

        column_index = int(
            np.argmax(column_probabilities)
        )

        row_code = row_codes[row_index]
        column_code = column_codes[column_index]

        pair = (row_code, column_code)

        confidence = float(
            min(
                np.max(row_probabilities),
                np.max(column_probabilities)
            )
        )

        def normalized_entropy(probabilities):

            entropy = -np.sum(
                probabilities * np.log(
                    probabilities + 1e-12
                )
            )

            return float(
                entropy / np.log(len(probabilities))
            )

        uncertainty = (
            normalized_entropy(row_probabilities)
            + normalized_entropy(column_probabilities)
        ) / 2

        if pair == self.previous_pair:

            self.stable_count += 1

        else:

            self.stable_count = 1

        self.previous_pair = pair

        self.pair_history.append(pair)

        return {
            "pair": pair,
            "character": VOICE_GRID[
                row_index
            ][
                column_index
            ],

            "confidence": confidence,
            "uncertainty": float(uncertainty),

            "row_probabilities": row_probabilities,
            "column_probabilities": column_probabilities,

            "repetitions": self.repetition_count,
            "stability": self.stable_count
        }


class VOICEAdaptiveStopping:

    def __init__(self):

        self.minimum = VOICE_CONFIG[
            "minimum_repetitions"
        ]

        self.maximum = VOICE_CONFIG[
            "maximum_repetitions"
        ]

        self.confidence_threshold = VOICE_CONFIG[
            "confidence_threshold"
        ]

        self.uncertainty_threshold = VOICE_CONFIG[
            "uncertainty_threshold"
        ]

        self.stability_window = VOICE_CONFIG[
            "stability_window"
        ]

    def should_stop(self, state):

        repetitions = state["repetitions"]

        if repetitions >= self.maximum:

            return True, "MAXIMUM_REPETITIONS"

        if repetitions < self.minimum:

            return False, "MINIMUM_REPETITIONS_NOT_REACHED"

        if state["confidence"] < self.confidence_threshold:

            return False, "LOW_CONFIDENCE"

        if state["uncertainty"] > self.uncertainty_threshold:

            return False, "HIGH_UNCERTAINTY"

        if state["stability"] < self.stability_window:

            return False, "PAIR_NOT_STABLE"

        return True, "ADAPTIVE_CONFIDENCE_STOP"


class VOICEAdaptiveDecoder:

    def __init__(self, classifier, communication_interface):

        self.classifier = classifier
        self.interface = communication_interface

        self.accumulator = VOICEEvidenceAccumulator()
        self.stopping_engine = VOICEAdaptiveStopping()

        self.session_log = []

        self.active = False
        self.current_trial = None

        self.interface.connect_decoder()

        print("\nADAPTIVE DECODER INITIALIZED")

    def start_trial(self, trial_id):

        if self.active:

            raise RuntimeError(
                "Previous trial must finish or be reset."
            )

        self.accumulator.reset()

        self.current_trial = trial_id

        self.active = True

        print("\nTrial started:", trial_id)

    def submit_repetition(
        self,
        flash_probabilities
    ):

        if not self.active:

            raise RuntimeError(
                "Call start_trial() first."
            )

        state = self.accumulator.update(
            flash_probabilities
        )

        stop, reason = self.stopping_engine.should_stop(
            state
        )

        record = {
            "trial": str(self.current_trial),
            "repetition": state["repetitions"],
            "predicted_pair": list(state["pair"]),
            "predicted_character": state["character"],
            "confidence": state["confidence"],
            "uncertainty": state["uncertainty"],
            "stability": state["stability"],
            "stopping_reason": reason,
            "timestamp": datetime.now().isoformat()
        }

        self.session_log.append(record)

        print(
            f"Repetition {state['repetitions']:02d} | "
            f"Character: {state['character']} | "
            f"Confidence: {state['confidence']:.3f} | "
            f"Uncertainty: {state['uncertainty']:.3f} | "
            f"Stop: {stop}"
        )

        if stop:

            self._complete_trial(state, reason)

        return {
            "state": state,
            "stop": stop,
            "reason": reason
        }

    def submit_eeg_repetition(
        self,
        eeg_epochs,
        flash_codes
    ):

        """
        eeg_epochs:
            12 EEG epochs, one for each flash.

        flash_codes:
            Actual event codes corresponding to each epoch.
            Expected codes: 3 through 14.
        """

        if not self.classifier.trained:

            raise RuntimeError(
                "Classifier is not trained."
            )

        eeg_epochs = np.asarray(eeg_epochs)

        flash_codes = np.asarray(
            flash_codes
        ).astype(int)

        if len(eeg_epochs) != 12:

            raise ValueError(
                "Expected exactly 12 epochs per repetition."
            )

        if sorted(flash_codes.tolist()) != sorted(
            VOICE_CONFIG["flash_codes"]
        ):

            raise ValueError(
                "Flash codes must contain each code from 3 to 14 once."
            )

        probabilities = (
            self.classifier.predict_target_probability(
                eeg_epochs
            )
        )

        flash_probabilities = {
            int(code): float(probability)
            for code, probability in zip(
                flash_codes,
                probabilities
            )
        }

        return self.submit_repetition(
            flash_probabilities
        )

    def _complete_trial(self, state, reason):

        actual_pair = state["pair"]

        # Convert dataset event codes to interface codes.
        normalized_pair = (
            actual_pair[0] - 2,
            actual_pair[1] - 2
        )

        character = self.interface.submit_decoder_output(
            predicted_pair=normalized_pair,
            confidence=state["confidence"],
            uncertainty=state["uncertainty"]
        )

        print("\nTRIAL COMPLETED")
        print("Selected character:", character)
        print("Stopping reason:", reason)
        print("Repetitions:", state["repetitions"])

        self.session_log[-1]["selected_character"] = character

        self.active = False

        self.current_trial = None

        return character

    def export_session(self):

        path = os.path.join(
            VOICE_OUTPUT_DIR,
            "VOICE_ADAPTIVE_DECODER_LOG.json"
        )

        with open(path, "w") as file:

            json.dump(
                self.session_log,
                file,
                indent=4
            )

        print("\nSession exported:", path)

        return path


def voice_convert_event_codes(event_codes):

    event_codes = np.asarray(event_codes).astype(int)

    converted = []

    for code in event_codes:

        if code not in VOICE_CONFIG["flash_codes"]:

            raise ValueError(
                f"Unknown flash code: {code}"
            )

        normalized = code - 2

        converted.append(normalized)

    return np.asarray(converted)


class VOICEEventManager:

    REQUIRED_COLUMNS = [
        "epoch_index",
        "trial_id",
        "repetition_id",
        "flash_code"
    ]

    CONTEXT_COLUMNS = [
        "subject",
        "session",
        "run"
    ]

    def __init__(self):

        self.event_table = None
        self.trial_groups = {}
        self.ready = False

    def load(self, event_table):

        table = event_table.copy()

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in table.columns
        ]

        if missing:

            raise ValueError(
                "Missing event metadata columns: "
                + str(missing)
                + ". Explicit event-to-epoch mapping is required."
            )

        table["epoch_index"] = pd.to_numeric(
            table["epoch_index"],
            errors="raise"
        ).astype(int)

        table["flash_code"] = pd.to_numeric(
            table["flash_code"],
            errors="raise"
        ).astype(int)

        table["repetition_id"] = pd.to_numeric(
            table["repetition_id"],
            errors="raise"
        )

        valid_codes = set(
            VOICE_APP_CONFIG["flash_codes"]
        )

        if not set(table["flash_code"]).issubset(valid_codes):

            raise ValueError(
                "Unknown flash codes detected."
            )

        if table["epoch_index"].duplicated().any():

            raise ValueError(
                "An epoch cannot appear twice in the event table."
            )

        self.event_table = table

        self._build_groups()

        self.ready = True

        return self

    def _build_groups(self):

        context = [
            col for col in self.CONTEXT_COLUMNS
            if col in self.event_table.columns
        ]

        grouping_columns = context + ["trial_id"]

        for key, trial in self.event_table.groupby(
            grouping_columns,
            sort=False
        ):

            repetitions = {}

            for repetition_id, repetition in trial.groupby(
                "repetition_id",
                sort=True
            ):

                codes = repetition["flash_code"].tolist()

                repetitions[repetition_id] = {
                    "data": repetition.copy(),
                    "complete": (
                        len(codes) == 12
                        and len(set(codes)) == 12
                        and set(codes) == set(
                            VOICE_APP_CONFIG["flash_codes"]
                        )
                    )
                }

            self.trial_groups[str(key)] = repetitions

    def summary(self):

        if not self.ready:

            raise RuntimeError(
                "Load event metadata first."
            )

        total_repetitions = 0
        complete_repetitions = 0

        for repetitions in self.trial_groups.values():

            total_repetitions += len(repetitions)

            complete_repetitions += sum(
                item["complete"]
                for item in repetitions.values()
            )

        return {
            "trials": len(self.trial_groups),
            "total_repetitions": total_repetitions,
            "complete_repetitions": complete_repetitions,
            "incomplete_repetitions": (
                total_repetitions - complete_repetitions
            )
        }


class VOICECommunicationSession:

    def __init__(self, decoder, interface):

        self.decoder = decoder
        self.interface = interface

        self.event_manager = VOICEEventManager()

        self.session_id = datetime.now().strftime(
            "VOICE_%Y%m%d_%H%M%S"
        )

        self.session_active = False

        self.processed_trials = []
        self.skipped_trials = []

        self.communication_log = []

        self.session_start = None
        self.session_end = None

        self.message_snapshot = ""

        print("Communication session manager initialized.")

    # --------------------------------------------------------------------
    # LOAD EVENT METADATA
    # --------------------------------------------------------------------

    def load_events(self, event_table):

        self.event_manager.load(event_table)

        summary = self.event_manager.summary()

        print("\nEVENT METADATA LOADED")

        for key, value in summary.items():

            print(f"{key}: {value}")

        return summary

    # --------------------------------------------------------------------
    # START SESSION
    # --------------------------------------------------------------------

    def start(self):

        if not self.event_manager.ready:

            raise RuntimeError(
                "Load event metadata before starting."
            )

        if self.session_active:

            raise RuntimeError(
                "Session is already active."
            )

        self.session_active = True

        self.session_start = datetime.now().isoformat()

        print("\nVOICE COMMUNICATION SESSION STARTED")
        print("Session ID:", self.session_id)

    # --------------------------------------------------------------------
    # PROCESS ONE REPETITION
    # --------------------------------------------------------------------

    def process_repetition(
        self,
        repetition_table,
        eeg_data
    ):

        if not self.session_active:

            raise RuntimeError(
                "Start the session first."
            )

        if len(repetition_table) != 12:

            raise ValueError(
                "A complete repetition requires 12 flash epochs."
            )

        repetition_table = repetition_table.sort_values(
            "flash_code"
        )

        if set(
            repetition_table["flash_code"]
        ) != set(VOICE_APP_CONFIG["flash_codes"]):

            raise ValueError(
                "Repetition does not contain all 12 flash codes."
            )

        epoch_indices = repetition_table[
            "epoch_index"
        ].to_numpy()

        flash_codes = repetition_table[
            "flash_code"
        ].to_numpy()

        if np.any(epoch_indices < 0) or np.any(
            epoch_indices >= len(eeg_data)
        ):

            raise IndexError(
                "Epoch index outside EEG array."
            )

        epochs = eeg_data[epoch_indices]

        # The existing decoder receives the 12 EEG epochs.
        result = self.decoder.submit_eeg_repetition(
            eeg_epochs=epochs,
            flash_codes=flash_codes
        )

        return result

    # --------------------------------------------------------------------
    # PROCESS COMPLETE TRIAL
    # --------------------------------------------------------------------

    def process_trial(
        self,
        trial_key,
        repetitions,
        eeg_data
    ):

        if not self.session_active:

            raise RuntimeError(
                "Session is not active."
            )

        print("\n" + "-" * 65)
        print("PROCESSING TRIAL:", trial_key)
        print("-" * 65)

        valid_repetitions = [
            (rep_id, item["data"])
            for rep_id, item in repetitions.items()
            if item["complete"]
        ]

        if not valid_repetitions:

            self.skipped_trials.append({
                "trial": trial_key,
                "reason": "No complete repetitions"
            })

            print("Trial skipped: no complete repetitions.")

            return None

        if len(valid_repetitions) < (
            VOICE_APP_CONFIG["minimum_repetitions"]
        ):

            self.skipped_trials.append({
                "trial": trial_key,
                "reason": "Insufficient complete repetitions"
            })

            print("Trial skipped: insufficient repetitions.")

            return None

        self.decoder.start_trial(trial_key)

        results = []

        for rep_id, repetition_table in valid_repetitions:

            if not self.decoder.active:

                break

            result = self.process_repetition(
                repetition_table,
                eeg_data
            )

            results.append({
                "repetition": rep_id,
                "result": result
            })

            if result["stop"]:

                break

        if self.decoder.active:

            # Do not force a character selection if the available
            # evidence never satisfied the stopping conditions.
            self.decoder.active = False
            self.decoder.current_trial = None

            self.skipped_trials.append({
                "trial": trial_key,
                "reason": "Decoder did not reach a stopping decision"
            })

            print("Trial ended without character selection.")

            return None

        final_record = {
            "trial": trial_key,
            "results": results,
            "completed_at": datetime.now().isoformat()
        }

        self.processed_trials.append(final_record)

        self.communication_log.append(final_record)

        return final_record

    # --------------------------------------------------------------------
    # PROCESS ALL AVAILABLE TRIALS
    # --------------------------------------------------------------------

    def process_all(self, eeg_data):

        if not self.session_active:

            raise RuntimeError(
                "Start the session first."
            )

        if not isinstance(eeg_data, np.ndarray):

            eeg_data = np.asarray(eeg_data)

        print("\nSTARTING COMPLETE COMMUNICATION PIPELINE")

        for trial_key, repetitions in (
            self.event_manager.trial_groups.items()
        ):

            self.process_trial(
                trial_key,
                repetitions,
                eeg_data
            )

        print("\nTRIAL PROCESSING FINISHED")

        print("Processed trials:", len(self.processed_trials))
        print("Skipped trials:", len(self.skipped_trials))

    # --------------------------------------------------------------------
    # SESSION EXPORT
    # --------------------------------------------------------------------

    def finish(self):

        self.session_active = False

        self.session_end = datetime.now().isoformat()

        self.message_snapshot = self.interface.message

        report = {

            "project": "PROJECT VOICE",

            "module": "MODULE 6",

            "session_id": self.session_id,

            "session_start": self.session_start,

            "session_end": self.session_end,

            "message": self.message_snapshot,

            "processed_trials": self.processed_trials,

            "skipped_trials": self.skipped_trials,

            "decoder_log": self.decoder.session_log,

            "selection_history": self.interface.history

        }

        path = os.path.join(
            VOICE_APP_CONFIG["session_directory"],
            self.session_id + "_SESSION.json"
        )

        with open(path, "w") as file:

            json.dump(
                report,
                file,
                indent=4,
                default=str
            )

        print("\nSESSION FINISHED")
        print("Message:", self.message_snapshot)
        print("Export:", path)

        return path


def voice_start_communication(session):

    if not session.event_manager.ready:

        raise RuntimeError(
            "Complete event metadata is required."
        )

    session.start()

    print("Communication engine active.")


def voice_process_communication(session, X):

    if not session.session_active:

        raise RuntimeError(
            "Start communication first."
        )

    if X is None:

        raise RuntimeError(
            "EEG epochs X are unavailable."
        )

    session.process_all(
        np.asarray(X)
    )


def voice_finish_communication(session):

    return session.finish()


def voice_show_status(session, classifier, decoder, app):

    print("\nPROJECT VOICE STATUS")
    print("-" * 50)

    print("Interface initialized:", app is not None)

    print("Classifier trained:", classifier.trained)

    print("Decoder initialized:", decoder is not None)

    print(
        "Event metadata ready:",
        session.event_manager.ready
    )

    print(
        "Session active:",
        session.session_active
    )

    print(
        "Characters composed:",
        len(app.message) if app is not None else 0
    )

    print(
        "Processed trials:",
        len(session.processed_trials)
    )

    print(
        "Skipped trials:",
        len(session.skipped_trials)
    )


class VOICEMessagingEngine:

    def __init__(self):

        self.message = ""
        self.history = []
        self.redo_history = []

        self.created_at = datetime.now().isoformat()

    def add_character(self, character, source="EEG"):

        character = str(character).upper()

        valid_characters = [
            c for row in VOICE_FINAL_GRID for c in row
        ]

        if character not in valid_characters:

            raise ValueError("Invalid character.")

        previous_message = self.message

        self.message += character

        self._record(
            "CHARACTER",
            character,
            previous_message,
            source
        )

        return self.message

    def add_space(self):

        previous_message = self.message

        self.message += " "

        self._record(
            "SPACE",
            " ",
            previous_message,
            "INTERFACE"
        )

        return self.message

    def backspace(self):

        if not self.message:
            return self.message

        previous_message = self.message

        self.message = self.message[:-1]

        self._record(
            "BACKSPACE",
            None,
            previous_message,
            "INTERFACE"
        )

        return self.message

    def undo(self):

        if not self.history:
            return self.message

        operation = self.history.pop()

        self.redo_history.append(operation)

        self.message = operation["previous_message"]

        return self.message

    def redo(self):

        if not self.redo_history:
            return self.message

        operation = self.redo_history.pop()

        self.message = operation["resulting_message"]

        self.history.append(operation)

        return self.message

    def clear(self):

        previous_message = self.message

        self.message = ""

        self._record(
            "CLEAR",
            None,
            previous_message,
            "INTERFACE"
        )

        return self.message

    def edit(self, new_message):

        previous_message = self.message

        self.message = str(new_message)

        self._record(
            "EDIT",
            None,
            previous_message,
            "USER"
        )

        return self.message

    def _record(
        self,
        action,
        value,
        previous_message,
        source
    ):

        self.history.append({

            "timestamp": datetime.now().isoformat(),

            "action": action,

            "value": value,

            "source": source,

            "previous_message": previous_message,

            "resulting_message": self.message

        })

        self.redo_history.clear()

    def get_state(self):

        return {

            "message": self.message,

            "length": len(self.message),

            "history_count": len(self.history),

            "created_at": self.created_at

        }


class VOICESessionManager:

    def __init__(self, messaging_engine):

        self.messaging = messaging_engine

        self.session_id = datetime.now().strftime(
            "VOICE_%Y%m%d_%H%M%S"
        )

        self.active = False

        self.start_time = None
        self.end_time = None

        self.trials = []
        self.selections = []
        self.events = []

        self.subject_id = None

        self.session_metadata = {}

    def start(self, subject_id=None):

        if self.active:

            raise RuntimeError(
                "A VOICE session is already active."
            )

        self.active = True

        self.start_time = datetime.now().isoformat()

        self.subject_id = subject_id

        self.events.append({

            "event": "SESSION_STARTED",

            "timestamp": self.start_time

        })

        print("VOICE SESSION STARTED")
        print("Session ID:", self.session_id)

    def record_trial(self, trial):

        if not self.active:

            raise RuntimeError(
                "Session is not active."
            )

        self.trials.append(trial)

    def record_selection(self, selection):

        if not self.active:

            raise RuntimeError(
                "Session is not active."
            )

        self.selections.append(selection)

    def finish(self):

        self.active = False

        self.end_time = datetime.now().isoformat()

        self.events.append({

            "event": "SESSION_FINISHED",

            "timestamp": self.end_time

        })

        return self.export()

    def export(self):

        report = {

            "project": "VOICE",

            "version": "2.0",

            "session_id": self.session_id,

            "subject_id": self.subject_id,

            "start_time": self.start_time,

            "end_time": self.end_time,

            "active": self.active,

            "message": self.messaging.message,

            "message_history": self.messaging.history,

            "trials": self.trials,

            "selections": self.selections,

            "events": self.events

        }

        path = os.path.join(
            VOICE_FINAL_DIR,
            self.session_id + "_SESSION.json"
        )

        with open(path, "w") as file:

            json.dump(
                report,
                file,
                indent=4,
                default=str
            )

        return path


class VOICEErrorRecovery:

    def __init__(self):

        self.errors = []

        self.recovery_count = 0

    def validate_flash_set(self, flash_codes):

        codes = list(
            np.asarray(flash_codes).astype(int)
        )

        expected = set(
            VOICE_FINAL_CONFIG["flash_codes"]
        )

        if len(codes) != 12:

            return False, "INCORRECT_FLASH_COUNT"

        if len(set(codes)) != 12:

            return False, "DUPLICATE_FLASH_CODE"

        if set(codes) != expected:

            return False, "INVALID_FLASH_MAPPING"

        return True, "VALID"

    def validate_epochs(self, epochs):

        epochs = np.asarray(epochs)

        if epochs.ndim != 3:

            return False, "EXPECTED_3D_EEG_ARRAY"

        if epochs.shape[0] != 12:

            return False, "EXPECTED_12_FLASH_EPOCHS"

        if not np.all(np.isfinite(epochs)):

            return False, "NONFINITE_EEG_VALUES"

        return True, "VALID"

    def recover(self, error, context=None):

        record = {

            "timestamp": datetime.now().isoformat(),

            "error": str(error),

            "context": context

        }

        self.errors.append(record)

        self.recovery_count += 1

        print("\nVOICE RECOVERY EVENT")
        print("Error:", error)

        return record

    def get_report(self):

        return {

            "total_errors": len(self.errors),

            "recovery_count": self.recovery_count,

            "errors": self.errors

        }


class VOICETrialReconstructor:

    REQUIRED_COLUMNS = [
        "epoch_index",
        "trial_id",
        "repetition_id",
        "flash_code"
    ]

    def __init__(self, event_table):

        self.table = event_table.copy()

        missing = [

            column for column in self.REQUIRED_COLUMNS

            if column not in self.table.columns

        ]

        if missing:

            raise ValueError(
                "Missing event metadata: " + str(missing)
            )

        self.table["epoch_index"] = self.table[
            "epoch_index"
        ].astype(int)

        self.table["flash_code"] = self.table[
            "flash_code"
        ].astype(int)

        self.table["repetition_id"] = self.table[
            "repetition_id"
        ].astype(int)

        if self.table["epoch_index"].duplicated().any():

            raise ValueError(
                "Duplicate epoch indices detected."
            )

        self.trials = self._reconstruct()

    def _reconstruct(self):

        result = {}

        context_columns = [

            c for c in [
                "subject",
                "session",
                "run"
            ]

            if c in self.table.columns

        ]

        group_columns = context_columns + ["trial_id"]

        for key, trial in self.table.groupby(
            group_columns,
            sort=False
        ):

            repetitions = {}

            for repetition_id, repetition in trial.groupby(
                "repetition_id",
                sort=True
            ):

                codes = repetition["flash_code"].tolist()

                complete = (

                    len(codes) == 12

                    and len(set(codes)) == 12

                    and set(codes) == set(
                        VOICE_FINAL_CONFIG["flash_codes"]
                    )

                )

                repetitions[repetition_id] = {

                    "table": repetition.copy(),

                    "complete": complete

                }

            result[str(key)] = repetitions

        return result

    def get_summary(self):

        total = sum(
            len(repetitions)
            for repetitions in self.trials.values()
        )

        complete = sum(

            item["complete"]

            for repetitions in self.trials.values()

            for item in repetitions.values()

        )

        return {

            "trials": len(self.trials),

            "repetitions": total,

            "complete_repetitions": complete,

            "incomplete_repetitions": total - complete

        }


class VOICEApplication:

    def __init__(self, classifier):

        if not classifier.trained:

            raise RuntimeError(
                "A trained P300 classifier is required."
            )

        self.classifier = classifier

        self.messaging = VOICEMessagingEngine()

        self.session = VOICESessionManager(
            self.messaging
        )

        self.recovery = VOICEErrorRecovery()

        self.evidence = None

        self.repetitions = 0

        self.previous_pair = None

        self.stability = 0

        self.current_candidate = None

        self.awaiting_confirmation = False

        self.last_state = None

        self.trial_history = []

        self.event_reconstructor = None

        self.eeg_data = None

        self._build_interface()

        print("VOICE APPLICATION CONTROLLER INITIALIZED")

    # -------------------------------------------------------------------------
    # 1. INTERFACE CONSTRUCTION
    # -------------------------------------------------------------------------

    def _build_interface(self):

        self.title = widgets.HTML(
            """
            <h1>VOICE</h1>
            <h3>Adaptive EEG Communication System</h3>
            <p>Personalized P300-based character selection</p>
            <hr>
            """
        )

        self.status = widgets.HTML(
            "<b>System status:</b> Initialized"
        )

        self.message_box = widgets.Textarea(

            value="",

            placeholder="Your composed message appears here...",

            description="Message:",

            layout=widgets.Layout(
                width="100%",
                height="110px"
            )

        )

        self.confidence_label = widgets.HTML(
            "<b>Decoder score:</b> Waiting"
        )

        self.uncertainty_label = widgets.HTML(
            "<b>Uncertainty:</b> Waiting"
        )

        self.repetition_label = widgets.HTML(
            "<b>Repetitions:</b> 0"
        )

        self.candidate_label = widgets.HTML(
            "<b>Candidate:</b> None"
        )

        self.grid_buttons = []

        for row in range(6):

            for column in range(6):

                character = VOICE_FINAL_GRID[row][column]

                button = widgets.Button(

                    description=character,

                    layout=widgets.Layout(
                        width="100%",
                        height="45px"
                    )

                )

                button.on_click(

                    lambda b, c=character:
                    self.manual_select(c)

                )

                self.grid_buttons.append(button)

        self.grid = widgets.GridBox(

            children=self.grid_buttons,

            layout=widgets.Layout(

                grid_template_columns="repeat(6, 1fr)",

                grid_gap="5px",

                width="100%"

            )

        )

        self.undo_button = widgets.Button(
            description="Undo",
            icon="undo"
        )

        self.redo_button = widgets.Button(
            description="Redo",
            icon="repeat"
        )

        self.space_button = widgets.Button(
            description="Space",
            icon="plus"
        )

        self.backspace_button = widgets.Button(
            description="Backspace",
            icon="backspace"
        )

        self.clear_button = widgets.Button(
            description="Clear",
            button_style="danger",
            icon="trash"
        )

        self.accept_button = widgets.Button(
            description="Accept candidate",
            button_style="success",
            icon="check"
        )

        self.reject_button = widgets.Button(
            description="Reject candidate",
            button_style="warning",
            icon="times"
        )

        self.start_button = widgets.Button(
            description="Start session",
            button_style="success"
        )

        self.finish_button = widgets.Button(
            description="Finish session"
        )

        self.export_button = widgets.Button(
            description="Export session",
            icon="download"
        )

        self.undo_button.on_click(
            lambda b: self._sync_message(self.messaging.undo())
        )

        self.redo_button.on_click(
            lambda b: self._sync_message(self.messaging.redo())
        )

        self.space_button.on_click(
            lambda b: self._sync_message(self.messaging.add_space())
        )

        self.backspace_button.on_click(
            lambda b: self._sync_message(self.messaging.backspace())
        )

        self.clear_button.on_click(
            lambda b: self._sync_message(self.messaging.clear())
        )

        self.accept_button.on_click(
            lambda b: self.confirm_candidate()
        )

        self.reject_button.on_click(
            lambda b: self.reject_candidate()
        )

        self.start_button.on_click(
            lambda b: self.start_session()
        )

        self.finish_button.on_click(
            lambda b: self.finish_session()
        )

        self.export_button.on_click(
            lambda b: self.export_session()
        )

        self.message_box.observe(
            self._on_message_edit,
            names="value"
        )

        self.history_output = widgets.Output()

        self.layout = widgets.VBox([

            self.title,

            self.status,

            widgets.HTML("<h3>Character Grid</h3>"),

            self.grid,

            widgets.HTML("<h3>Message Composer</h3>"),

            self.message_box,

            widgets.HBox([

                self.undo_button,

                self.redo_button,

                self.space_button,

                self.backspace_button,

                self.clear_button

            ]),

            widgets.HTML("<h3>Decoder Feedback</h3>"),

            self.confidence_label,

            self.uncertainty_label,

            self.repetition_label,

            self.candidate_label,

            widgets.HBox([

                self.accept_button,

                self.reject_button

            ]),

            widgets.HTML("<h3>Session Controls</h3>"),

            widgets.HBox([

                self.start_button,

                self.finish_button,

                self.export_button

            ]),

            widgets.HTML("<h3>Recent Activity</h3>"),

            self.history_output

        ])

    # -------------------------------------------------------------------------
    # 2. MESSAGE SYNCHRONIZATION
    # -------------------------------------------------------------------------

    def _sync_message(self, message):

        self.message_box.value = message

        self.status.value = (
            "<b>System status:</b> Message updated"
        )

        self._refresh_history()

    def _on_message_edit(self, change):

        if change["new"] != self.messaging.message:

            self.messaging.edit(change["new"])

            self._refresh_history()

    # -------------------------------------------------------------------------
    # 3. MANUAL GRID PREVIEW
    # -------------------------------------------------------------------------

    def manual_select(self, character):

        if self.awaiting_confirmation:

            self.status.value = (
                "<b>Confirm or reject the current candidate first.</b>"
            )

            return

        self.messaging.add_character(
            character,
            source="MANUAL PREVIEW"
        )

        self._sync_message(
            self.messaging.message
        )

    # -------------------------------------------------------------------------
    # 4. TRIAL INITIALIZATION
    # -------------------------------------------------------------------------

    def start_trial(self, trial_id):

        if not self.session.active:

            raise RuntimeError(
                "Start a communication session first."
            )

        if self.evidence is not None:

            raise RuntimeError(
                "A trial is already active."
            )

        self.evidence = {
            code: 0.0
            for code in VOICE_FINAL_CONFIG["flash_codes"]
        }

        self.repetitions = 0

        self.previous_pair = None

        self.stability = 0

        self.current_trial_id = trial_id

        self.trial_history.append({

            "trial_id": str(trial_id),

            "started_at": datetime.now().isoformat()

        })

        self.status.value = (
            f"<b>Trial active:</b> {trial_id}"
        )

    # -------------------------------------------------------------------------
    # 5. REAL EEG REPETITION PROCESSING
    # -------------------------------------------------------------------------

    def submit_eeg_repetition(
        self,
        eeg_epochs,
        flash_codes
    ):

        if self.evidence is None:

            raise RuntimeError(
                "Start a trial before submitting EEG."
            )

        try:

            valid, reason = self.recovery.validate_epochs(
                eeg_epochs
            )

            if not valid:

                raise ValueError(reason)

            valid, reason = self.recovery.validate_flash_set(
                flash_codes
            )

            if not valid:

                raise ValueError(reason)

            eeg_epochs = np.asarray(eeg_epochs)

            flash_codes = np.asarray(
                flash_codes
            ).astype(int)

            probabilities = (
                self.classifier.predict_target_probability(
                    eeg_epochs
                )
            )

            for code, probability in zip(
                flash_codes,
                probabilities
            ):

                probability = np.clip(
                    probability,
                    1e-5,
                    1 - 1e-5
                )

                evidence_value = np.log(
                    probability / (1 - probability)
                )

                self.evidence[int(code)] += float(
                    evidence_value
                )

            return self._update_decoder()

        except Exception as error:

            self.recovery.recover(
                error,
                context="EEG repetition processing"
            )

            raise

    # -------------------------------------------------------------------------
    # 6. ADAPTIVE EVIDENCE DECODER
    # -------------------------------------------------------------------------

    def _update_decoder(self):

        row_codes = VOICE_FINAL_CONFIG["row_codes"]

        column_codes = VOICE_FINAL_CONFIG["column_codes"]

        row_scores = np.array([
            self.evidence[code]
            for code in row_codes
        ])

        column_scores = np.array([
            self.evidence[code]
            for code in column_codes
        ])

        row_probabilities = softmax(row_scores)

        column_probabilities = softmax(column_scores)

        row_index = int(
            np.argmax(row_probabilities)
        )

        column_index = int(
            np.argmax(column_probabilities)
        )

        pair = (

            row_codes[row_index],

            column_codes[column_index]

        )

        character = VOICE_FINAL_GRID[
            row_index
        ][
            column_index
        ]

        score = float(
            min(
                np.max(row_probabilities),
                np.max(column_probabilities)
            )
        )

        def entropy(p):

            return -np.sum(
                p * np.log(p + 1e-12)
            ) / np.log(len(p))

        uncertainty = float(
            (
                entropy(row_probabilities)
                + entropy(column_probabilities)
            ) / 2
        )

        self.repetitions += 1

        if pair == self.previous_pair:

            self.stability += 1

        else:

            self.stability = 1

        self.previous_pair = pair

        self.last_state = {

            "pair": pair,

            "character": character,

            "score": score,

            "uncertainty": uncertainty,

            "repetitions": self.repetitions,

            "stability": self.stability

        }

        self.confidence_label.value = (
            f"<b>Decoder score:</b> {score:.2%}"
        )

        self.uncertainty_label.value = (
            f"<b>Uncertainty:</b> {uncertainty:.3f}"
        )

        self.repetition_label.value = (
            f"<b>Repetitions:</b> {self.repetitions}"
        )

        self.candidate_label.value = (
            f"<b>Candidate:</b> {character}"
        )

        minimum = VOICE_FINAL_CONFIG[
            "minimum_repetitions"
        ]

        maximum = VOICE_FINAL_CONFIG[
            "maximum_repetitions"
        ]

        stable = (
            self.stability
            >= VOICE_FINAL_CONFIG["stability_window"]
        )

        confident = (
            score
            >= VOICE_FINAL_CONFIG["confidence_threshold"]
        )

        certain = (
            uncertainty
            <= VOICE_FINAL_CONFIG["uncertainty_threshold"]
        )

        if self.repetitions < minimum:

            decision = "CONTINUE: MINIMUM REPETITIONS"

        elif confident and certain and stable:

            decision = "READY FOR CHARACTER CONFIRMATION"

            self._present_candidate(
                character,
                "ADAPTIVE CONFIDENCE"
            )

        elif self.repetitions >= maximum:

            decision = "MAXIMUM REACHED: CONFIRMATION REQUIRED"

            self._present_candidate(
                character,
                "MAXIMUM REPETITIONS / UNCERTAIN"
            )

        else:

            decision = "CONTINUE: GATHERING EVIDENCE"

        self.status.value = (
            f"<b>Decoder:</b> {decision}"
        )

        return {

            **self.last_state,

            "decision": decision

        }

    # -------------------------------------------------------------------------
    # 7. CHARACTER CONFIRMATION
    # -------------------------------------------------------------------------

    def _present_candidate(self, character, reason):

        self.current_candidate = character

        self.awaiting_confirmation = True

        self.status.value = (
            f"<b>Candidate {character} awaiting confirmation.</b> "
            f"Reason: {reason}"
        )

        self.accept_button.disabled = False

        self.reject_button.disabled = False

    def confirm_candidate(self):

        if not self.awaiting_confirmation:

            return

        character = self.current_candidate

        self.messaging.add_character(
            character,
            source="EEG DECODER CONFIRMED"
        )

        selection = {

            "character": character,

            "pair": list(self.last_state["pair"]),

            "score": self.last_state["score"],

            "uncertainty": self.last_state["uncertainty"],

            "repetitions": self.repetitions,

            "confirmed_at": datetime.now().isoformat()

        }

        self.session.record_selection(selection)

        self._reset_trial()

        self._sync_message(
            self.messaging.message
        )

        self.status.value = (
            f"<b>Character {character} accepted.</b>"
        )

    def reject_candidate(self):

        if not self.awaiting_confirmation:

            return

        self.status.value = (
            "<b>Candidate rejected. Awaiting correction.</b>"
        )

        self.current_candidate = None

        self.awaiting_confirmation = False

        self._reset_trial()

    def _reset_trial(self):

        if self.current_trial_id is not None:

            self.session.record_trial({

                "trial_id": str(self.current_trial_id),

                "repetitions": self.repetitions,

                "final_state": self.last_state,

                "finished_at": datetime.now().isoformat()

            })

        self.evidence = None

        self.repetitions = 0

        self.previous_pair = None

        self.stability = 0

        self.current_candidate = None

        self.awaiting_confirmation = False

        self.current_trial_id = None

        self.last_state = None

    # -------------------------------------------------------------------------
    # 8. SESSION MANAGEMENT
    # -------------------------------------------------------------------------

    def start_session(self):

        self.session.start()

        self.status.value = (
            "<b>VOICE session active.</b>"
        )

    def finish_session(self):

        if self.evidence is not None:

            raise RuntimeError(
                "Finish or discard the active trial first."
            )

        if not self.session.active:

            raise RuntimeError(
                "No active session."
            )

        path = self.session.finish()

        self.status.value = (
            f"<b>Session finished.</b> Saved to {path}"
        )

        return path

    # -------------------------------------------------------------------------
    # 9. EVENT TABLE INTEGRATION
    # -------------------------------------------------------------------------

    def load_event_table(self, event_table, eeg_data):

        self.event_reconstructor = VOICETrialReconstructor(
            event_table
        )

        self.eeg_data = np.asarray(eeg_data)

        if self.eeg_data.ndim != 3:

            raise ValueError(
                "EEG data must have shape "
                "(epochs, channels, time)."
            )

        if (
            self.event_reconstructor.table["epoch_index"].min() < 0
            or self.event_reconstructor.table["epoch_index"].max()
            >= len(self.eeg_data)
        ):

            raise IndexError(
                "Event metadata references unavailable EEG epochs."
            )

        return self.event_reconstructor.get_summary()

    # -------------------------------------------------------------------------
    # 10. APPLICATION EXPORT
    # -------------------------------------------------------------------------

    def export_session(self):

        path = self.session.export()

        message_path = os.path.join(
            VOICE_FINAL_DIR,
            self.session.session_id + "_MESSAGE.txt"
        )

        with open(message_path, "w") as file:

            file.write(self.messaging.message)

        print("Session export:", path)

        print("Message export:", message_path)

        return path

    # -------------------------------------------------------------------------
    # 11. ACTIVITY DISPLAY
    # -------------------------------------------------------------------------

    def _refresh_history(self):

        with self.history_output:

            self.history_output.clear_output(wait=True)

            for item in self.messaging.history[-10:]:

                print(
                    item["timestamp"],
                    "|",
                    item["action"],
                    "|",
                    item["source"]
                )

    # -------------------------------------------------------------------------
    # 12. DISPLAY APPLICATION
    # -------------------------------------------------------------------------

    def display(self):

        display(self.layout)


