from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
import subprocess


if __name__ == "__main__":
    env = os.environ.copy()
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"], env=env, check=True)
