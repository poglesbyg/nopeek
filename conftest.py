"""Put the repository root on sys.path so tests can import the survey harness.

The library itself lives under src/ and is installed; the survey harness is a
repo-local tool and is deliberately not packaged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
