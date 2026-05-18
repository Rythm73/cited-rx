"""
conftest.py — project root
Ensures project root is on sys.path for all tests.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # cited-rx/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
