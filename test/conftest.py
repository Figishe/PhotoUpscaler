import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Test files should be able to import using project root as base
sys.path.insert(0, str(ROOT))
