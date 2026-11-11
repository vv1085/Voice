import os

import gradio as gr
import numpy as np

from voice_backend import (
    VOICE_FINAL_GRID,
    VOICEApplication,
    VOICEP300Classifier,
)

# --------------------------------------------------
# VOICE DEPLOYMENT INITIALIZATION
# --------------------------------------------------

# Calibration data (.npz with arrays "X" and "y") used to train the P300
# classifier at startup. Override the location with VOICE_CALIBRATION_PATH.
CALIBRATION_PATH = os.environ.get("VOICE_CALIBRATION_PATH", "calibration.npz")


def initialize_voice():
    """Train the classifier from calibration data and build the application.

    Returns (application, status_message). If calibration data is missing the
    application is None and the message explains what to provide, so the web
    UI can still start instead of crashing.
    """

    if not os.path.exists(CALIBRATION_PATH):
        return None, (
            f"⚠️ Calibration data not found at `{CALIBRATION_PATH}`. "
            "Provide an .npz file with arrays `X` and `y`, or set "
            "`VOICE_CALIBRATION_PATH`."
        )

    try:
        with np.load(CALIBRATION_PATH, allow_pickle=False) as data:
            X, y = data["X"], data["y"]

        classifier = VOICEP300Classifier().fit(X, y)
        application = VOICEApplication(classifier=classifier)
    except Exception as error:
        return None, f"⚠️ Backend initialization failed: {error}"

    return application, "✅ Classifier trained. Backend ready."


voice_application, voice_status = initialize_voice()


# --------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------

with gr.Blocks(title="VOICE | Adaptive EEG Communication") as demo:

    gr.Markdown(
        """
        # VOICE
        ### Adaptive EEG Communication System

        P300-based character selection and communication.
        """
    )

    status = gr.Markdown(voice_status)

    message = gr.Textbox(
        label="Composed message",
        value=getattr(voice_application, "message", "") if voice_application else "",
        lines=4,
        interactive=True
    )

    gr.Markdown("### Character Grid")

    grid = gr.Dataframe(
        value=VOICE_FINAL_GRID,
        interactive=False,
        headers=None
    )

    if voice_application is None:
        gr.Markdown(
            "The EEG decoder and session controls will be enabled "
            "after the calibration data is connected."
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
