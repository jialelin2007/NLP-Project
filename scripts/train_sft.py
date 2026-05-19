#!/usr/bin/env python
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("training") / "train_sft.py", run_name="__main__")
