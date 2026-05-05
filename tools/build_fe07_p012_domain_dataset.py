"""Tool entrypoint for FE-07 dataset generation."""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from build_fe07_p012_domain_dataset import main  # noqa: E402


if __name__ == "__main__":
    main()
