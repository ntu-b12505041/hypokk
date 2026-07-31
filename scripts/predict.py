#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hypok_mimic3.config import load_config
from hypok_mimic3.inference import predict_cached_window


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only MIMIC-III ECG inference")
    parser.add_argument("--config", default="configs/mimic3.yaml")
    parser.add_argument("--input", required=True, help="Materialized ECG .npz")
    parser.add_argument("--checkpoint", help="best.pt or model_weights.h5")
    args = parser.parse_args()
    result = predict_cached_window(
        load_config(args.config),
        args.input,
        args.checkpoint,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
