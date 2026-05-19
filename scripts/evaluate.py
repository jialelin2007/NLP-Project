#!/usr/bin/env python
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("evaluation") / "evaluate_translation.py"
    runpy.run_path(target, run_name="__main__")
