#!/usr/bin/env python3
"""Supervised fine-tuning entry point. See ``python scripts/train_sft.py --help``.

Kaggle, two T4s:  accelerate launch --num_processes 2 scripts/train_sft.py --profile t4 ...
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from folioocr.train.sft import main  # noqa: E402

if __name__ == "__main__":
    main()
