#!/usr/bin/env python
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("models") / "inspect_local_model.py"
    runpy.run_path(target, run_name="__main__")
