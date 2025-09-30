# tests/conftest.py
import os
import numpy as np
import pytest

# Use non-interactive backend for any plotting
os.environ.setdefault("MPLBACKEND", "Agg")

@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(12345)
