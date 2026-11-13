#!/usr/bin/env bash
# Render build step: install the app, then make sure the model files exist.
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt

if python build_artifacts.py --check; then
  echo "Using committed artifacts."
else
  echo "Artifacts missing: building them (source: ${VOICE_BUILD_SOURCE:-bnci})."
  pip install -r requirements-build.txt
  python build_artifacts.py \
    --source "${VOICE_BUILD_SOURCE:-bnci}" \
    --data-dir "${VOICE_MNE_DATA:-/tmp/mne_data}"
fi

python build_artifacts.py --check
