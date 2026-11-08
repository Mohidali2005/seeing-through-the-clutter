import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT/"scripts"
# kaggle points these at its own folders through environment variables
DATA = Path(os.environ.get("TANGLE_DATA",ROOT/"data"))
RESULTS = Path(os.environ.get("TANGLE_RESULTS",ROOT/"runs"/"local"))
MODELS = Path(os.environ.get("TANGLE_MODELS",ROOT/"cache"/"models"))
NOTES = RESULTS/"notes"
SWAPS = RESULTS/"swap"
ANALYSIS = Path(os.environ.get("TANGLE_ANALYSIS",ROOT/"analysis"))
RUNS = ROOT/"runs"
