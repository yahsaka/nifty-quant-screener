# tests/conftest.py
import sys
import os

# Add the 'src' directory to the Python path so tests can resolve absolute imports
# (e.g., `from indicators import calculate_indicators`)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
