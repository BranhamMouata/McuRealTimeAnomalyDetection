from pathlib import Path


# Project root = parent of the python/ folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "model"
