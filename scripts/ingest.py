from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.build_indexes import build_all_indexes


if __name__ == "__main__":
    build_all_indexes()
