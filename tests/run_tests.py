"""Run the workflow checks on Kaggle, with small synthetic assets only."""

from pathlib import Path
import unittest

test_root = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(test_root), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)

from demo.prepare_data import read_secret

print(f"Kaggle upload secret available: {bool(read_secret('KAGGLE_API_TOKEN'))}")
print(f"Hugging Face secret available: {bool(read_secret('HF_TOKEN'))}")
