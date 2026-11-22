from __future__ import annotations

import argparse
import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SUITES = {
    "fused_ops": {
        "original": (
            "results/raw/full/"
            "fused_ops_full.json"
        ),
        "repetitions": (
            "rep1_fused_ops.json",
            "rep2_fused_ops.json",
        ),
        "key_fields": (
            "experiment",
            "implementation",
            "shape",
        ),
        "throughput": "effective_gb_s",
    },
    "cuda_extension": {
        "original": (
            "results/raw/full/"
            "cuda_extension_full.json"
        ),
        "repetitions": (
            "rep1_cuda_extension.json",
            "rep2_cuda_extension.json",
        ),
        "key_fields": (
            "experiment",
            "implementation",
            "shape",
        ),
        "throughput": "effective_gb_s",
    },
    "matmul": {
        "original": (
            "results/raw/full/"
            "matmul_full.json"
        ),
        "repetitions": (
            "rep1_matmul.json",
            "rep2_matmul.json",
        ),
        "key_fields": (
            "experiment",
            "implementation",
            "shape_mnk",
        ),
        "throughput": "tflop_s",
    },
    "attention": {
        "original": (
            "results/raw/full/"
            "attention_full.json"
        ),
        "repetitions": (
            "rep1_attention.json",
            "rep2_attention.json",
        ),
        "key_fields": (
            "experiment",
            "implementation",
            "shape_bshd",
        ),
        "throughput": (
            "tokens_per_second"
        ),
    },
    "transformer_training": {
        "original": (
            "results/raw/full/"
            "transformer_training_full.json"
        ),
        "repetitions": tuple(
            f"rep{index}_"
            "transformer_training.json"
            for index in range(1, 7)
        ),
        "key_fields": (
            "experiment",
            "implementation",
            "shape",
        ),
        "throughput": (
            "tokens_per_second"
        ),
    },
    "allreduce_8gpu": {
        "original": (
            "results/raw/distributed/"
            "allreduce_8gpu_full.json"
        ),
        "repetitions": (
            "rep1_allreduce_8gpu.json",
            "rep2_allreduce_8gpu.json",
        ),
        "key_fields": ("size_mb",),
        "throughput": "bus_gb_s",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(
            freeze(item)
            for item in value
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    key,
                    freeze(item),
                )
                for key, item
                in value.items()
            )
        )
    return value


def index_records(
    payload: dict[str, Any],
    key_fields: tuple[str, ...],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        tuple(
            freeze(record[field])
            for field in key_fields
        ): record
        for record in payload["records"]
    }


def median(values: list[float]) -> float:
    return float(
        statistics.median(values)
    )


def relative_spread(
    values: list[float],
) -> float:
    minimum = min(values)
    return (
        0.0
        if minimum == 0.0
        else max(values) / minimum - 1.0
    )


def key_payload(
    fields: tuple[str, ...],
    key: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        field: (
            list(value)
            if isinstance(value, tuple)
            else value
        )
        for field, value
        in zip(fields, key)
    }


def compare_suite(
    *,
    root: Path,
    input_dir: Path,
    name: str,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    original = load_json(
        root / config["original"]
    )
    repetitions = [
        load_json(
            input_dir / filename
        )
        for filename in config[
            "repetitions"
        ]
    ]
    fields = config["key_fields"]
    original_index = index_records(
        original,
        fields,
    )
    repetition_indices = [
        index_records(
            payload,
            fields,
        )
        for payload in repetitions
    ]
    keys_match = all(
        set(index)
        == set(original_index)
        for index in repetition_indices
    )
    if not keys_match:
        raise RuntimeError(
            f"{name} record keys differ"
        )
    throughput_field = config[
        "throughput"
    ]
    comparisons = []
    for key in sorted(
        original_index,
        key=str,
    ):
        baseline = original_index[key]
        rows = [
            index[key]
            for index
            in repetition_indices
        ]
        latency = [
            float(row["median_ms"])
            for row in rows
        ]
        throughput = [
            float(
                row[
                    throughput_field
                ]
            )
            for row in rows
        ]
        comparison = {
            "key": key_payload(
                fields,
                key,
            ),
            "median_ms": {
                "original": float(
                    baseline[
                        "median_ms"
                    ]
                ),
                "repetitions": latency,
                "new_median": median(
                    latency
                ),
                "new_over_original": (
                    median(latency)
                    / float(
                        baseline[
                            "median_ms"
                        ]
                    )
                ),
                "repeat_spread": (
                    relative_spread(
                        latency
                    )
                ),
            },
            throughput_field: {
                "original": float(
                    baseline[
                        throughput_field
                    ]
                ),
                "repetitions": (
                    throughput
                ),
                "new_median": median(
                    throughput
                ),
                "new_over_original": (
                    median(throughput)
                    / float(
                        baseline[
                            throughput_field
                        ]
                    )
                ),
                "repeat_spread": (
                    relative_spread(
                        throughput
                    )
                ),
            },
        }
        if "peak_incremental_memory_bytes" in baseline:
            memory = [
                int(
                    row[
                        "peak_incremental_memory_bytes"
                    ]
                )
                for row in rows
            ]
            comparison[
                "peak_incremental_memory_bytes"
            ] = {
                "original": int(
                    baseline[
                        "peak_incremental_memory_bytes"
                    ]
                ),
                "repetitions": memory,
                "new_median": median(
                    [
                        float(value)
                        for value in memory
                    ]
                ),
            }
        if "correctness" in baseline:
            comparison[
                "correctness"
            ] = {
                metric: {
                    "original": float(
                        baseline[
                            "correctness"
                        ][metric]
                    ),
                    "new_maximum": max(
                        float(
                            row[
                                "correctness"
                            ][metric]
                        )
                        for row in rows
                    ),
                }
                for metric in (
                    "max_abs_error",
                    "mean_abs_error",
                    "max_rel_error",
                    "mean_rel_error",
                )
            }
        if (
            name
            == "transformer_training"
        ):
            comparison[
                "training_correctness"
            ] = {
                "loss_abs_error_maximum": max(
                    float(
                        row[
                            "loss_abs_error"
                        ]
                    )
                    for row in rows
                ),
                "gradient_l2_relative_error_maximum": max(
                    float(
                        row[
                            "gradient_l2_relative_error"
                        ]
                    )
                    for row in rows
                ),
            }
        comparisons.append(comparison)
    return (
        comparisons,
        original,
        repetitions,
    )


def find_record(
    payload: dict[str, Any],
    **criteria: Any,
) -> dict[str, Any]:
    return next(
        record
        for record in payload["records"]
        if all(
            record.get(key) == value
            for key, value
            in criteria.items()
        )
    )


def pair_speedups(
    payloads: list[dict[str, Any]],
    shape: list[int],
) -> list[float]:
    return [
        float(
            find_record(
                payload,
                implementation=(
                    "torch_eager"
                ),
                shape=shape,
            )["median_ms"]
        )
        / float(
            find_record(
                payload,
                implementation=(
                    "torch_compile"
                ),
                shape=shape,
            )["median_ms"]
        )
        for payload in payloads
    ]


def run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    suite_comparisons = {}
    originals = {}
    repetitions = {}
    for name, config in SUITES.items():
        (
            suite_comparisons[name],
            originals[name],
            repetitions[name],
        ) = compare_suite(
            root=args.root,
            input_dir=args.input_dir,
            name=name,
            config=config,
        )

    original_environment = originals[
        "fused_ops"
    ]["environment"]
    new_environments = [
        payload["environment"]
        for suite, payloads
        in repetitions.items()
        if suite != "allreduce_8gpu"
        for payload in payloads
    ]
    environment_fields = (
        "gpu_name",
        "gpu_memory_bytes",
        "compute_capability",
        "torch",
        "torch_cuda",
        "triton",
        "flash_attn",
        "cuda_driver",
    )
    environment_match = {
        field: all(
            environment.get(field)
            == original_environment.get(
                field
            )
            for environment
            in new_environments
        )
        for field in environment_fields
    }

    fused_original = originals[
        "fused_ops"
    ]
    fused_new = repetitions[
        "fused_ops"
    ]
    cuda_original = originals[
        "cuda_extension"
    ]
    cuda_new = repetitions[
        "cuda_extension"
    ]
    attention_original = originals[
        "attention"
    ]
    attention_new = repetitions[
        "attention"
    ]
    matmul_original = originals[
        "matmul"
    ]
    matmul_new = repetitions[
        "matmul"
    ]
    training_original = originals[
        "transformer_training"
    ]
    training_new = repetitions[
        "transformer_training"
    ]
    allreduce_original = originals[
        "allreduce_8gpu"
    ]
    allreduce_new = repetitions[
        "allreduce_8gpu"
    ]

    def throughput_summary(
        *,
        original: dict[str, Any],
        new: list[dict[str, Any]],
        field: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        baseline = float(
            find_record(
                original,
                **criteria,
            )[field]
        )
        values = [
            float(
                find_record(
                    payload,
                    **criteria,
                )[field]
            )
            for payload in new
        ]
        return {
            "original": baseline,
            "repetitions": values,
            "new_median": median(
                values
            ),
            "new_over_original": (
                median(values)
                / baseline
            ),
            "repeat_spread": (
                relative_spread(values)
            ),
        }

    swiglu_triton = throughput_summary(
        original=fused_original,
        new=fused_new,
        field="effective_gb_s",
        criteria={
            "experiment": "swiglu",
            "implementation": "triton",
            "shape": [33554432],
        },
    )
    swiglu_cuda = throughput_summary(
        original=cuda_original,
        new=cuda_new,
        field="effective_gb_s",
        criteria={
            "experiment": "swiglu",
            "implementation": (
                "native_cuda_bf16x2"
            ),
            "shape": [33554432],
        },
    )
    rmsnorm_triton = throughput_summary(
        original=fused_original,
        new=fused_new,
        field="effective_gb_s",
        criteria={
            "experiment": "rmsnorm",
            "implementation": "triton",
            "shape": [4096, 8192],
        },
    )

    attention_shape = [
        2,
        2048,
        16,
        128,
    ]

    def attention_speedup(
        payload: dict[str, Any],
    ) -> float:
        math_row = find_record(
            payload,
            implementation=(
                "torch_sdpa_math"
            ),
            shape_bshd=attention_shape,
        )
        flash_row = find_record(
            payload,
            implementation=(
                "torch_sdpa_flash"
            ),
            shape_bshd=attention_shape,
        )
        return (
            float(math_row["median_ms"])
            / float(
                flash_row["median_ms"]
            )
        )

    def attention_memory_ratio(
        payload: dict[str, Any],
    ) -> float:
        math_row = find_record(
            payload,
            implementation=(
                "torch_sdpa_math"
            ),
            shape_bshd=attention_shape,
        )
        flash_row = find_record(
            payload,
            implementation=(
                "torch_sdpa_flash"
            ),
            shape_bshd=attention_shape,
        )
        return (
            float(
                math_row[
                    "peak_incremental_memory_bytes"
                ]
            )
            / float(
                flash_row[
                    "peak_incremental_memory_bytes"
                ]
            )
        )

    matmul_shapes = [
        [512, 4096, 4096],
        [4096, 4096, 4096],
        [4096, 11008, 4096],
        [8192, 4096, 11008],
    ]

    def matmul_advantage(
        payload: dict[str, Any],
        shape: list[int],
    ) -> float:
        cublas = find_record(
            payload,
            implementation="torch_cublas",
            shape_mnk=shape,
        )
        triton = find_record(
            payload,
            implementation=(
                "triton_autotuned"
            ),
            shape_mnk=shape,
        )
        return (
            float(cublas["tflop_s"])
            / float(triton["tflop_s"])
        )

    matmul_new_advantages = {
        "x".join(
            str(value)
            for value in shape
        ): [
            matmul_advantage(
                payload,
                shape,
            )
            for payload in matmul_new
        ]
        for shape in matmul_shapes
    }
    matmul_original_advantages = {
        "x".join(
            str(value)
            for value in shape
        ): matmul_advantage(
            matmul_original,
            shape,
        )
        for shape in matmul_shapes
    }
    matmul_negative_control = all(
        advantage > 1.0
        for values
        in matmul_new_advantages.values()
        for advantage in values
    )

    training_shapes = (
        [4, 1024, 1024, 16, 2816],
        [1, 4096, 1024, 16, 2816],
    )
    training = {}
    for shape in training_shapes:
        name = (
            f"{shape[0]}x{shape[1]}"
        )
        original_speedup = pair_speedups(
            [training_original],
            shape,
        )[0]
        speedups = pair_speedups(
            training_new,
            shape,
        )
        eager = [
            float(
                find_record(
                    payload,
                    implementation=(
                        "torch_eager"
                    ),
                    shape=shape,
                )["median_ms"]
            )
            for payload in training_new
        ]
        compiled = [
            float(
                find_record(
                    payload,
                    implementation=(
                        "torch_compile"
                    ),
                    shape=shape,
                )["median_ms"]
            )
            for payload in training_new
        ]
        original_eager = float(
            find_record(
                training_original,
                implementation=(
                    "torch_eager"
                ),
                shape=shape,
            )["median_ms"]
        )
        original_compiled = float(
            find_record(
                training_original,
                implementation=(
                    "torch_compile"
                ),
                shape=shape,
            )["median_ms"]
        )
        training[name] = {
            "original_speedup": (
                original_speedup
            ),
            "paired_speedups": speedups,
            "new_median_speedup": (
                median(speedups)
            ),
            "new_minimum_speedup": min(
                speedups
            ),
            "new_maximum_speedup": max(
                speedups
            ),
            "eager_ms": {
                "original": (
                    original_eager
                ),
                "repetitions": eager,
                "new_median": median(
                    eager
                ),
                "new_over_original": (
                    median(eager)
                    / original_eager
                ),
            },
            "compiled_ms": {
                "original": (
                    original_compiled
                ),
                "repetitions": compiled,
                "new_median": median(
                    compiled
                ),
                "new_over_original": (
                    median(compiled)
                    / original_compiled
                ),
                "repeat_spread": (
                    relative_spread(
                        compiled
                    )
                ),
            },
        }

    allreduce_256 = throughput_summary(
        original=allreduce_original,
        new=allreduce_new,
        field="bus_gb_s",
        criteria={"size_mb": 256},
    )

    correctness = {
        "fused_mean_abs_error_maximum": max(
            comparison[
                "correctness"
            ]["mean_abs_error"][
                "new_maximum"
            ]
            for comparison
            in suite_comparisons[
                "fused_ops"
            ]
        ),
        "cuda_mean_abs_error_maximum": max(
            comparison[
                "correctness"
            ]["mean_abs_error"][
                "new_maximum"
            ]
            for comparison
            in suite_comparisons[
                "cuda_extension"
            ]
        ),
        "attention_mean_abs_error_maximum": max(
            comparison[
                "correctness"
            ]["mean_abs_error"][
                "new_maximum"
            ]
            for comparison
            in suite_comparisons[
                "attention"
            ]
        ),
        "training_loss_abs_error_maximum": max(
            comparison[
                "training_correctness"
            ][
                "loss_abs_error_maximum"
            ]
            for comparison
            in suite_comparisons[
                "transformer_training"
            ]
        ),
        "training_gradient_relative_error_maximum": max(
            comparison[
                "training_correctness"
            ][
                "gradient_l2_relative_error_maximum"
            ]
            for comparison
            in suite_comparisons[
                "transformer_training"
            ]
        ),
    }
    correctness_passed = (
        correctness[
            "fused_mean_abs_error_maximum"
        ]
        <= 1e-3
        and correctness[
            "cuda_mean_abs_error_maximum"
        ]
        <= 1e-3
        and correctness[
            "attention_mean_abs_error_maximum"
        ]
        <= 1e-3
        and correctness[
            "training_loss_abs_error_maximum"
        ]
        <= 1e-4
        and correctness[
            "training_gradient_relative_error_maximum"
        ]
        <= 1e-3
    )
    representative_ratios = {
        "swiglu_triton": (
            swiglu_triton[
                "new_over_original"
            ]
        ),
        "swiglu_native_cuda": (
            swiglu_cuda[
                "new_over_original"
            ]
        ),
        "rmsnorm_triton": (
            rmsnorm_triton[
                "new_over_original"
            ]
        ),
        "attention_2048_speedup": (
            median(
                [
                    attention_speedup(
                        payload
                    )
                    for payload
                    in attention_new
                ]
            )
            / attention_speedup(
                attention_original
            )
        ),
        "allreduce_256_bus": (
            allreduce_256[
                "new_over_original"
            ]
        ),
    }
    representative_within_5_percent = all(
        abs(value - 1.0) <= 0.05
        for value
        in representative_ratios.values()
    )
    compiler_speedup_reproduced = all(
        abs(
            row["new_median_speedup"]
            / row["original_speedup"]
            - 1.0
        )
        <= 0.10
        for row in training.values()
    )

    if not (
        all(
            environment_match.values()
        )
        and correctness_passed
        and matmul_negative_control
    ):
        raise RuntimeError(
            "E110 validation failed"
        )

    result = {
        "suite": (
            "independent_a100_reproduction"
        ),
        "source": {
            "original": (
                "E1-E8 A100 benchmark "
                "artifacts"
            ),
            "new": (
                "two full independent-node "
                "repetitions plus four targeted "
                "transformer-training repeats"
            ),
        },
        "model": (
            "Re-run representative fusion, "
            "native CUDA, GEMM, attention, "
            "transformer-training, and 8-GPU "
            "all-reduce benchmarks on a fresh "
            "single-node 8xA100 allocation with "
            "the same software stack."
        ),
        "headline": {
            "environment_match": (
                environment_match
            ),
            "representative": {
                "swiglu_triton_gb_s": (
                    swiglu_triton
                ),
                "swiglu_native_cuda_gb_s": (
                    swiglu_cuda
                ),
                "rmsnorm_triton_gb_s": (
                    rmsnorm_triton
                ),
                "attention_2048": {
                    "original_math_to_flash_speedup": (
                        attention_speedup(
                            attention_original
                        )
                    ),
                    "new_math_to_flash_speedups": [
                        attention_speedup(
                            payload
                        )
                        for payload
                        in attention_new
                    ],
                    "new_median_math_to_flash_speedup": median(
                        [
                            attention_speedup(
                                payload
                            )
                            for payload
                            in attention_new
                        ]
                    ),
                    "original_memory_ratio": (
                        attention_memory_ratio(
                            attention_original
                        )
                    ),
                    "new_memory_ratios": [
                        attention_memory_ratio(
                            payload
                        )
                        for payload
                        in attention_new
                    ],
                },
                "matmul_cublas_over_triton": {
                    "original": (
                        matmul_original_advantages
                    ),
                    "new_repetitions": (
                        matmul_new_advantages
                    ),
                    "cublas_wins_every_new_case": (
                        matmul_negative_control
                    ),
                },
                "transformer_training": (
                    training
                ),
                "allreduce_256_bus_gb_s": (
                    allreduce_256
                ),
            },
            "representative_new_over_original": (
                representative_ratios
            ),
            "representative_within_5_percent": (
                representative_within_5_percent
            ),
            "compiler_speedup_reproduced_within_10_percent": (
                compiler_speedup_reproduced
            ),
            "correctness": correctness,
            "correctness_passed": (
                correctness_passed
            ),
        },
        "suite_comparisons": (
            suite_comparisons
        ),
        "validation": {
            "record_keys_match": True,
            "environment_fields_match": (
                all(
                    environment_match.values()
                )
            ),
            "matmul_negative_control_preserved": (
                matmul_negative_control
            ),
            "correctness_passed": (
                correctness_passed
            ),
            "full_repetitions": 2,
            "transformer_training_repetitions": 6,
        },
        "limitations": [
            (
                "Both allocations use the same "
                "A100 model and software versions; "
                "this is allocation-level rather "
                "than cross-architecture "
                "reproduction."
            ),
            (
                "The full panel has two process "
                "repetitions except transformer "
                "training, which has six after a "
                "targeted variability follow-up."
            ),
            (
                "GPU clocks and power were not "
                "sampled during every timed group; "
                "scheduler and clock-state effects "
                "can remain."
            ),
            (
                "The panel reproduces representative "
                "early kernel claims rather than "
                "every later MoE and serving study."
            ),
        ],
    }
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = args.output.with_suffix(
        args.output.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
