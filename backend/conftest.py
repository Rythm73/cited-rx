"""
tests/conftest.py

Pytest configuration and shared fixtures.
Ensures project root is on sys.path for all tests.
"""
import sys
from pathlib import Path

# Always resolve imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))