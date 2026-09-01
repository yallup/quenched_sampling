import sys
from pathlib import Path

# the spike-and-slab target lives with the example, not in the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
