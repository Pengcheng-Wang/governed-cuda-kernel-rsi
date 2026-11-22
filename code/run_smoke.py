from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            sys.executable,
            str(code_root / "bench_fused_ops.py"),
            "--profile",
            "smoke",
            "--output",
            str(args.output_dir / "fused_ops_smoke.json"),
        ],
        [
            sys.executable,
            str(code_root / "bench_matmul.py"),
            "--profile",
            "smoke",
            "--output",
            str(args.output_dir / "matmul_smoke.json"),
        ],
        [
            sys.executable,
            str(code_root / "bench_attention.py"),
            "--profile",
            "smoke",
            "--output",
            str(args.output_dir / "attention_smoke.json"),
        ],
        [
            sys.executable,
            str(code_root / "bench_decode_graphs.py"),
            "--profile",
            "smoke",
            "--output",
            str(args.output_dir / "decode_graphs_smoke.json"),
        ],
        [
            sys.executable,
            str(code_root / "bench_cuda_extension.py"),
            "--profile",
            "smoke",
            "--source-root",
            str(code_root / "cuda"),
            "--output",
            str(args.output_dir / "cuda_extension_smoke.json"),
        ],
        [
            sys.executable,
            str(code_root / "bench_transformer_training.py"),
            "--profile",
            "smoke",
            "--output",
            str(args.output_dir / "transformer_training_smoke.json"),
        ],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=code_root)


if __name__ == "__main__":
    main()
