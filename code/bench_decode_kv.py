from __future__ import annotations

import argparse
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import flashinfer
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from common import (
    allclose_metrics,
    benchmark_cuda,
    environment_metadata,
    write_result,
)


def quality_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    normalized_rmse = (
        error.square().mean().sqrt()
        / expected_f.square().mean().sqrt().clamp_min(1e-8)
    )
    cosine = F.cosine_similarity(
        actual_f.flatten(),
        expected_f.flatten(),
        dim=0,
    )
    return {
        **allclose_metrics(actual, expected),
        "normalized_rmse": normalized_rmse.item(),
        "cosine_similarity": cosine.item(),
    }


def torch_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: SDPBackend,
) -> torch.Tensor:
    q_sdpa = q.unsqueeze(0).unsqueeze(2)
    k_sdpa = k.permute(1, 0, 2).unsqueeze(0)
    v_sdpa = v.permute(1, 0, 2).unsqueeze(0)
    enable_gqa = q.shape[0] != k.shape[1]
    with sdpa_kernel(backend):
        output = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
    return output.squeeze(0).squeeze(1)


def timed_implementation(
    name: str,
    fn: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    *,
    useful_kv_bytes: int,
    output_tokens: int,
    warmup: int,
    groups: int,
    iterations: int,
) -> dict[str, Any]:
    try:
        setup_started = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup_started
        correctness = quality_metrics(output, reference)
        timing = benchmark_cuda(
            fn,
            warmup=warmup,
            groups=groups,
            iterations_per_group=iterations,
        )
        seconds = timing["median_ms"] / 1_000.0
        return {
            "implementation": name,
            "supported": True,
            "setup_seconds": setup_seconds,
            "correctness": correctness,
            **timing,
            "output_tokens_per_second": output_tokens / seconds,
            "effective_kv_gb_s": useful_kv_bytes / seconds / 1e9,
        }
    except Exception as error:
        return {
            "implementation": name,
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def profiler_summary(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int = 3,
    top_k: int = 20,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        fn()
    torch.cuda.synchronize()

    events: list[dict[str, Any]] = []
    for event in prof.key_averages():
        self_device_us = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        device_us = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )
        if self_device_us <= 0.0 and device_us <= 0.0:
            continue
        events.append(
            {
                "name": event.key,
                "count": event.count,
                "self_device_time_us": self_device_us,
                "device_time_us": device_us,
            }
        )
    events.sort(key=lambda item: item["self_device_time_us"], reverse=True)
    return {
        "top_cuda_events": events[:top_k],
        "table": prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_k,
        ),
    }


def single_decode_records(
    profile_name: str,
    *,
    dtype: torch.dtype,
    num_qo_heads: int,
    head_dim: int,
) -> list[dict[str, Any]]:
    if profile_name == "smoke":
        contexts = [257]
        kv_head_counts = [8, 1]
        warmup, groups, iterations = 2, 3, 3
    else:
        contexts = [512, 2048, 8192, 32768]
        kv_head_counts = [32, 8, 1]
        warmup, groups, iterations = 10, 7, 20

    records: list[dict[str, Any]] = []
    for context in contexts:
        for num_kv_heads in kv_head_counts:
            q = torch.randn(
                (num_qo_heads, head_dim),
                device="cuda",
                dtype=dtype,
            )
            k = torch.randn(
                (context, num_kv_heads, head_dim),
                device="cuda",
                dtype=dtype,
            )
            v = torch.randn_like(k)
            reference = torch_decode(q, k, v, backend=SDPBackend.MATH)
            useful_kv_bytes = (
                2
                * context
                * num_kv_heads
                * head_dim
                * torch.empty((), dtype=dtype).element_size()
            )

            implementations = (
                (
                    "torch_sdpa_flash",
                    lambda: torch_decode(
                        q,
                        k,
                        v,
                        backend=SDPBackend.FLASH_ATTENTION,
                    ),
                ),
                (
                    "flashinfer_cuda_core",
                    lambda: flashinfer.single_decode_with_kv_cache(
                        q,
                        k,
                        v,
                        kv_layout="NHD",
                        use_tensor_cores=False,
                    ),
                ),
                (
                    "flashinfer_tensor_core",
                    lambda: flashinfer.single_decode_with_kv_cache(
                        q,
                        k,
                        v,
                        kv_layout="NHD",
                        use_tensor_cores=True,
                    ),
                ),
            )
            for name, fn in implementations:
                record = timed_implementation(
                    name,
                    fn,
                    reference,
                    useful_kv_bytes=useful_kv_bytes,
                    output_tokens=1,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                )
                record.update(
                    {
                        "experiment": "single_contiguous_decode",
                        "context": context,
                        "num_qo_heads": num_qo_heads,
                        "num_kv_heads": num_kv_heads,
                        "gqa_group_size": num_qo_heads // num_kv_heads,
                        "head_dim": head_dim,
                        "dtype": str(dtype),
                        "useful_kv_bytes": useful_kv_bytes,
                    }
                )
                records.append(record)

            del q, k, v, reference
            torch.cuda.empty_cache()
    return records


def build_paged_cache(
    lengths: Sequence[int],
    *,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pages_per_request = [
        math.ceil(length / page_size)
        for length in lengths
    ]
    total_pages = sum(pages_per_request)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    cache = torch.randn(
        (
            total_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
        ),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    physical_pages = torch.randperm(
        total_pages,
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    ).to(torch.int32)
    indptr_values = [0]
    for count in pages_per_request:
        indptr_values.append(indptr_values[-1] + count)
    indptr = torch.tensor(
        indptr_values,
        device="cuda",
        dtype=torch.int32,
    )
    last_page_len = torch.tensor(
        [
            ((length - 1) % page_size) + 1
            for length in lengths
        ],
        device="cuda",
        dtype=torch.int32,
    )
    return cache, indptr, physical_pages, last_page_len


def gather_request(
    cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    request: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(indptr[request].item())
    end = int(indptr[request + 1].item())
    page_ids = indices[start:end].to(torch.int64)
    request_pages = cache.index_select(0, page_ids)
    k = request_pages[:, 0].flatten(0, 1)[:length]
    v = request_pages[:, 1].flatten(0, 1)[:length]
    return k, v


def paged_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    lengths: Sequence[int],
    *,
    request_limit: int = 2,
) -> torch.Tensor:
    outputs = []
    for request in range(min(len(lengths), request_limit)):
        k, v = gather_request(
            cache,
            indptr,
            indices,
            request,
            lengths[request],
        )
        outputs.append(
            torch_decode(
                q[request],
                k,
                v,
                backend=SDPBackend.MATH,
            )
        )
    return torch.stack(outputs)


def make_decode_wrapper(
    *,
    workspace: torch.Tensor,
    use_tensor_cores: bool,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> tuple[Any, float]:
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=use_tensor_cores,
        backend="auto",
    )
    started = time.perf_counter()
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
    )
    torch.cuda.synchronize()
    return wrapper, time.perf_counter() - started


def paged_record(
    *,
    name: str,
    wrapper: Any,
    plan_seconds: float,
    q: torch.Tensor,
    cache: torch.Tensor,
    reference: torch.Tensor,
    useful_kv_bytes: int,
    warmup: int,
    groups: int,
    iterations: int,
) -> dict[str, Any]:
    output = torch.empty_like(q)
    fn = lambda: wrapper.run(
        q,
        cache,
        out=output,
    )[: reference.shape[0]]
    record = timed_implementation(
        name,
        fn,
        reference,
        useful_kv_bytes=useful_kv_bytes,
        output_tokens=q.shape[0],
        warmup=warmup,
        groups=groups,
        iterations=iterations,
    )
    record["plan_seconds"] = plan_seconds
    return record


def graph_record(
    *,
    workspace: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    q: torch.Tensor,
    cache: torch.Tensor,
    reference: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
    useful_kv_bytes: int,
    warmup: int,
    groups: int,
    iterations: int,
) -> dict[str, Any]:
    metadata_indptr = torch.empty_like(indptr)
    metadata_indices = torch.empty_like(indices)
    metadata_last_page_len = torch.empty_like(last_page_len)
    wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        workspace,
        metadata_indptr,
        metadata_indices,
        metadata_last_page_len,
        "NHD",
        use_tensor_cores=True,
    )
    plan_started = time.perf_counter()
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        disable_split_kv=True,
    )
    torch.cuda.synchronize()
    plan_seconds = time.perf_counter() - plan_started

    output = torch.empty_like(q)
    lse = torch.empty(
        (q.shape[0], num_qo_heads),
        device=q.device,
        dtype=torch.float32,
    )
    direct_fn = lambda: wrapper.run(
        q,
        cache,
        out=output,
        lse=lse,
    )
    for _ in range(3):
        direct_fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph):
        wrapper.run(q, cache, out=output, lse=lse)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - capture_started
    correctness = quality_metrics(output[: reference.shape[0]], reference)
    replay_fn = lambda: graph.replay()
    timing = benchmark_cuda(
        replay_fn,
        warmup=warmup,
        groups=groups,
        iterations_per_group=iterations,
    )
    seconds = timing["median_ms"] / 1_000.0
    return {
        "implementation": "flashinfer_tensor_core_cuda_graph",
        "supported": True,
        "plan_seconds": plan_seconds,
        "graph_capture_seconds": capture_seconds,
        "correctness": correctness,
        **timing,
        "output_tokens_per_second": q.shape[0] / seconds,
        "effective_kv_gb_s": useful_kv_bytes / seconds / 1e9,
    }


def paged_decode_records(
    profile_name: str,
    *,
    dtype: torch.dtype,
    num_qo_heads: int,
    head_dim: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if profile_name == "smoke":
        configurations = [
            {
                "name": "smoke_homogeneous",
                "lengths": [257, 257],
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": True,
            },
            {
                "name": "smoke_mqa",
                "lengths": [511, 389],
                "num_kv_heads": 1,
                "page_size": 16,
                "run_graph": False,
            },
        ]
        warmup, groups, iterations = 2, 3, 3
    else:
        configurations = [
            {
                "name": "mha_b16_l2048",
                "lengths": [2048] * 16,
                "num_kv_heads": 32,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "gqa_b1_l2048",
                "lengths": [2048],
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "gqa_b16_l2048",
                "lengths": [2048] * 16,
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "gqa_b64_l2048",
                "lengths": [2048] * 64,
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": True,
            },
            {
                "name": "gqa_b16_l8192",
                "lengths": [8192] * 16,
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "gqa_b64_l8192",
                "lengths": [8192] * 64,
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "mqa_b16_l8192",
                "lengths": [8192] * 16,
                "num_kv_heads": 1,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "mqa_b64_l8192",
                "lengths": [8192] * 64,
                "num_kv_heads": 1,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "gqa_b64_heterogeneous",
                "lengths": [
                    512 + ((request * 353) % 7681)
                    for request in range(64)
                ],
                "num_kv_heads": 8,
                "page_size": 16,
                "run_graph": False,
            },
            {
                "name": "gqa_b16_l8192_page64",
                "lengths": [8192] * 16,
                "num_kv_heads": 8,
                "page_size": 64,
                "run_graph": False,
            },
        ]
        warmup, groups, iterations = 5, 7, 10

    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for config_index, configuration in enumerate(configurations):
        lengths = configuration["lengths"]
        batch_size = len(lengths)
        num_kv_heads = configuration["num_kv_heads"]
        page_size = configuration["page_size"]
        q = torch.randn(
            (batch_size, num_qo_heads, head_dim),
            device="cuda",
            dtype=dtype,
        )
        cache, indptr, indices, last_page_len = build_paged_cache(
            lengths,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            seed=100 + config_index,
        )
        reference = paged_reference(
            q,
            cache,
            indptr,
            indices,
            lengths,
        )
        useful_kv_bytes = (
            2
            * sum(lengths)
            * num_kv_heads
            * head_dim
            * torch.empty((), dtype=dtype).element_size()
        )
        workspace_bytes = (
            512 * 1024 * 1024
            if num_kv_heads == 1
            else 128 * 1024 * 1024
        )
        workspace = torch.empty(
            workspace_bytes,
            device="cuda",
            dtype=torch.uint8,
        )

        wrapper_cuda = None
        wrapper_tensor = None
        config_records: list[dict[str, Any]] = []
        try:
            wrapper_cuda, plan_cuda = make_decode_wrapper(
                workspace=workspace,
                use_tensor_cores=False,
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                dtype=dtype,
            )
            config_records.append(
                paged_record(
                    name="flashinfer_paged_cuda_core",
                    wrapper=wrapper_cuda,
                    plan_seconds=plan_cuda,
                    q=q,
                    cache=cache,
                    reference=reference,
                    useful_kv_bytes=useful_kv_bytes,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                )
            )
        except Exception as error:
            config_records.append(
                {
                    "implementation": "flashinfer_paged_cuda_core",
                    "supported": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

        try:
            wrapper_tensor, plan_tensor = make_decode_wrapper(
                workspace=workspace,
                use_tensor_cores=True,
                indptr=indptr,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                dtype=dtype,
            )
            config_records.append(
                paged_record(
                    name="flashinfer_paged_tensor_core",
                    wrapper=wrapper_tensor,
                    plan_seconds=plan_tensor,
                    q=q,
                    cache=cache,
                    reference=reference,
                    useful_kv_bytes=useful_kv_bytes,
                    warmup=warmup,
                    groups=groups,
                    iterations=iterations,
                )
            )
        except Exception as error:
            config_records.append(
                {
                    "implementation": "flashinfer_paged_tensor_core",
                    "supported": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        if configuration["run_graph"]:
            try:
                config_records.append(
                    graph_record(
                        workspace=workspace,
                        indptr=indptr,
                        indices=indices,
                        last_page_len=last_page_len,
                        q=q,
                        cache=cache,
                        reference=reference,
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        page_size=page_size,
                        dtype=dtype,
                        useful_kv_bytes=useful_kv_bytes,
                        warmup=warmup,
                        groups=groups,
                        iterations=iterations,
                    )
                )
            except Exception as error:
                config_records.append(
                    {
                        "implementation": "flashinfer_tensor_core_cuda_graph",
                        "supported": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

        for record in config_records:
            record.update(
                {
                    "experiment": "batch_paged_decode",
                    "configuration": configuration["name"],
                    "batch_size": batch_size,
                    "context_min": min(lengths),
                    "context_mean": sum(lengths) / batch_size,
                    "context_max": max(lengths),
                    "total_kv_tokens": sum(lengths),
                    "num_qo_heads": num_qo_heads,
                    "num_kv_heads": num_kv_heads,
                    "gqa_group_size": num_qo_heads // num_kv_heads,
                    "head_dim": head_dim,
                    "page_size": page_size,
                    "dtype": str(dtype),
                    "useful_kv_bytes": useful_kv_bytes,
                    "physical_pages_randomized": True,
                    "workspace_bytes": workspace_bytes,
                }
            )
            records.append(record)

        if (
            configuration["name"] == "gqa_b64_l2048"
            and wrapper_tensor is not None
        ):
            output = torch.empty_like(q)
            profiles.append(
                {
                    "configuration": configuration["name"],
                    "implementation": "flashinfer_paged_tensor_core",
                    **profiler_summary(
                        lambda: wrapper_tensor.run(
                            q,
                            cache,
                            out=output,
                        )
                    ),
                }
            )

        del (
            q,
            cache,
            indptr,
            indices,
            last_page_len,
            reference,
            workspace,
        )
        del wrapper_cuda, wrapper_tensor
        torch.cuda.empty_cache()

    return records, profiles


def run(profile_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    dtype = torch.bfloat16
    num_qo_heads = 32
    head_dim = 128
    single_records = single_decode_records(
        profile_name,
        dtype=dtype,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
    )
    paged_records, profiles = paged_decode_records(
        profile_name,
        dtype=dtype,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
    )
    return {
        "suite": "decode_kv",
        "profile": profile_name,
        "environment": {
            **environment_metadata(),
            "flashinfer": getattr(
                flashinfer,
                "__version__",
                importlib.metadata.version("flashinfer-python"),
            ),
        },
        "notes": [
            "Decode attends to the entire supplied KV cache, so is_causal=False is correct for the one-token query.",
            "Effective KV bandwidth counts one read of K and V and excludes query/output and page-table traffic.",
            "Paged correctness is checked on the first two requests against PyTorch math SDPA.",
            "Paged physical page indices are randomized to exercise logical-to-physical indirection.",
            "Plan/JIT and CUDA Graph capture times are separated from steady-state replay.",
        ],
        "records": single_records + paged_records,
        "profiler_summaries": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()
