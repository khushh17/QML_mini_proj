"""Small, inspectable California Housing regression experiment."""
import os
from pathlib import Path

# Keep plotting's font cache inside the project (also works in restricted accounts).
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'tmp' / 'matplotlib'))
