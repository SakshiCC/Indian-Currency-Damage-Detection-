import sys
from pathlib import Path

VINIT_DIR = Path(__file__).resolve().parent.parent.parent
if str(VINIT_DIR) not in sys.path:
    sys.path.insert(0, str(VINIT_DIR))
