# tests/conftest.py
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Use non-interactive backend for any plotting
os.environ.setdefault("MPLBACKEND", "Agg")

# Make optimize.py (in the repository root) importable from the tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(12345)
