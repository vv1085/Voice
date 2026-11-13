import os
import subprocess
import sys
from pathlib import Path

notebook = Path(__file__).parent / "PROJECT_VOICE_fixed.ipynb"

if not notebook.exists():
    raise FileNotFoundError(f"Notebook not found: {notebook}")

subprocess.run([
    sys.executable, "-m", "voila",
    str(notebook),
    "--port", os.environ.get("PORT", "10000"),
    "--no-browser",
    "--Voila.ip=0.0.0.0"
], check=True)
