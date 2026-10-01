# tests/test_configs.py
# Smoke test: every example config in configs/ must run through optimize.py without errors.
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.json"))

@pytest.mark.parametrize("config", CONFIGS, ids=[c.name for c in CONFIGS])
def test_config_runs(config):
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(ROOT / "optimize.py"), str(config)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
