#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic dev/test-final split from an evaluation set aligned with infomation.json."
    )
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument(
        "--source-eval-file",
        default="test.json",
        help="Source evaluation split filename. This should be the split aligned with infomation.json.",
    )
    parser.add_argument("--info-file", default="infomation.json", help="Info filename used for sandbox candidate lookup")
    parser.add_argument("--dev-out", default="splits/dev_eval.json", help="Output dev-eval split filename")
    parser.add_argument("--test-out", default="splits/test_final.json", help="Output held-out final test split filename")
    parser.add_argument(
        "--manifest-out",
        default="splits/eval_split_manifest.json",
        help="Output manifest filename",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dev-ratio", type=float, default=0.2, help="Fraction of source evaluation samples used for dev")
    group.add_argument("--dev-size", type=int, default=None, help="Absolute number of dev samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    source_path = data_dir / args.source_eval_file
    info_path = data_dir / args.info_file

    samples = json.loads(source_path.read_text(encoding="utf-8"))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"{source_path} must contain a JSON list")
    if not isinstance(info, dict):
        raise ValueError(f"{info_path} must contain a JSON object keyed by pid")
    if len(samples) < 2:
        raise ValueError(f"{source_path} must contain at least 2 samples")

    missing_info = [int(sample.get("pid")) for sample in samples if str(sample.get("pid")) not in info]
    if missing_info:
        preview = missing_info[:20]
        raise ValueError(
            f"{len(missing_info)} samples in {args.source_eval_file} do not exist in {args.info_file}. "
            f"First missing pids: {preview}"
        )

    dev_size = args.dev_size
    if dev_size is None:
        ratio = max(0.0, min(1.0, args.dev_ratio))
        dev_size = int(round(len(samples) * ratio))
    dev_size = max(1, min(len(samples) - 1, dev_size))

    indexed = list(enumerate(samples))
    rng = random.Random(args.seed)
    rng.shuffle(indexed)
    dev_indices = {idx for idx, _ in indexed[:dev_size]}

    dev_samples = [sample for idx, sample in enumerate(samples) if idx in dev_indices]
    test_samples = [sample for idx, sample in enumerate(samples) if idx not in dev_indices]

    dev_path = data_dir / args.dev_out
    test_path = data_dir / args.test_out
    manifest_path = data_dir / args.manifest_out
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dev_path.write_text(json.dumps(dev_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "source_eval_file": args.source_eval_file,
        "info_file": args.info_file,
        "dev_out": args.dev_out,
        "test_out": args.test_out,
        "seed": args.seed,
        "dev_size": len(dev_samples),
        "test_size": len(test_samples),
        "source_size": len(samples),
        "dev_ratio_effective": len(dev_samples) / len(samples),
        "dev_pids": [sample.get("pid") for sample in dev_samples],
        "test_pids": [sample.get("pid") for sample in test_samples],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Source evaluation samples: {len(samples)}")
    print(f"Dev-eval samples: {len(dev_samples)} -> {dev_path.resolve()}")
    print(f"Held-out test-final samples: {len(test_samples)} -> {test_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
