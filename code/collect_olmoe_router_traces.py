from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DOMAIN_TERMS = {
    "general_prose": (
        "question",
        "evidence",
        "example",
        "context",
        "summary",
        "revision",
    ),
    "software": (
        "compiler",
        "kernel",
        "buffer",
        "thread",
        "latency",
        "invariant",
    ),
    "mathematics": (
        "matrix",
        "theorem",
        "gradient",
        "probability",
        "sequence",
        "estimate",
    ),
    "physics": (
        "energy",
        "field",
        "particle",
        "momentum",
        "measurement",
        "uncertainty",
    ),
    "biology": (
        "cell",
        "protein",
        "organism",
        "pathway",
        "sample",
        "control",
    ),
    "economics": (
        "market",
        "incentive",
        "supply",
        "demand",
        "policy",
        "tradeoff",
    ),
    "operations": (
        "schedule",
        "inventory",
        "queue",
        "capacity",
        "failure",
        "recovery",
    ),
    "dialogue": (
        "speaker",
        "request",
        "response",
        "clarification",
        "decision",
        "feedback",
    ),
    "structured_data": (
        "record",
        "column",
        "schema",
        "identifier",
        "aggregate",
        "validation",
    ),
    "narrative": (
        "traveler",
        "workshop",
        "map",
        "weather",
        "observation",
        "choice",
    ),
}


def build_prompt(domain: str, index: int) -> str:
    terms = DOMAIN_TERMS[domain]
    sentences: list[str] = []
    for turn in range(18):
        a = terms[(index + turn) % len(terms)]
        b = terms[(index + 2 * turn + 1) % len(terms)]
        c = terms[(index + 3 * turn + 2) % len(terms)]
        sentences.append(
            f"Example {index:03d}.{turn:02d} in {domain} examines "
            f"{a}, relates it to {b}, and checks {c} against a "
            "clear baseline before drawing a limited conclusion."
        )
    return " ".join(sentences)


def build_corpus(
    samples_per_domain: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    prompts: list[str] = []
    ranges: list[dict[str, Any]] = []
    for domain in DOMAIN_TERMS:
        start = len(prompts)
        prompts.extend(
            build_prompt(domain, index)
            for index in range(samples_per_domain)
        )
        ranges.append(
            {
                "domain": domain,
                "start": start,
                "stop": len(prompts),
            }
        )
    return prompts, ranges


def sample_expert_counts(
    router_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    top_k: int,
    num_experts: int,
) -> torch.Tensor:
    batch_size, sequence_length = attention_mask.shape
    logits = router_logits.reshape(
        batch_size,
        sequence_length,
        num_experts,
    )
    topk_ids = torch.topk(
        logits.float(),
        top_k,
        dim=-1,
        sorted=False,
    ).indices
    counts = torch.zeros(
        (batch_size, num_experts),
        device=router_logits.device,
        dtype=torch.int32,
    )
    for sample in range(batch_size):
        valid = attention_mask[sample].bool()
        selected = topk_ids[sample][valid].reshape(-1)
        counts[sample].scatter_add_(
            0,
            selected,
            torch.ones_like(selected, dtype=torch.int32),
        )
    return counts


def collect(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(7_000)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    prompts, domain_ranges = build_corpus(
        args.samples_per_domain
    )
    corpus_sha256 = hashlib.sha256(
        "\n\n".join(prompts).encode("utf-8")
    ).hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        local_files_only=args.local_files_only,
    )
    model.eval()
    config = model.config
    num_layers = int(config.num_hidden_layers)
    num_experts = int(config.num_experts)
    top_k = int(config.num_experts_per_tok)

    all_counts = torch.empty(
        (
            len(prompts),
            num_layers,
            num_experts,
        ),
        dtype=torch.int16,
    )
    valid_tokens: list[int] = []
    batch_times_ms: list[float] = []
    first_batch_counts: torch.Tensor | None = None

    for start in range(0, len(prompts), args.batch_size):
        stop = min(start + args.batch_size, len(prompts))
        batch = tokenizer(
            prompts[start:stop],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.sequence_length,
        )
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                **batch,
                output_router_logits=True,
                use_cache=False,
                return_dict=True,
            )
        torch.cuda.synchronize()
        batch_times_ms.append(
            (time.perf_counter() - started) * 1_000.0
        )
        if outputs.router_logits is None:
            raise RuntimeError("Model did not return router logits")
        if len(outputs.router_logits) != num_layers:
            raise RuntimeError(
                "Router layer count mismatch: "
                f"{len(outputs.router_logits)} != {num_layers}"
            )

        layer_counts = []
        for logits in outputs.router_logits:
            layer_counts.append(
                sample_expert_counts(
                    logits,
                    batch["attention_mask"],
                    top_k=top_k,
                    num_experts=num_experts,
                )
            )
        stacked = torch.stack(
            layer_counts,
            dim=1,
        ).cpu()
        all_counts[start:stop] = stacked.to(torch.int16)
        valid_tokens.extend(
            int(value)
            for value in (
                batch["attention_mask"].sum(dim=1).cpu().tolist()
            )
        )
        if first_batch_counts is None:
            first_batch_counts = stacked.clone()
            with torch.inference_mode():
                repeat = model(
                    **batch,
                    output_router_logits=True,
                    use_cache=False,
                    return_dict=True,
                )
            repeat_counts = torch.stack(
                [
                    sample_expert_counts(
                        logits,
                        batch["attention_mask"],
                        top_k=top_k,
                        num_experts=num_experts,
                    )
                    for logits in repeat.router_logits
                ],
                dim=1,
            ).cpu()
            if not torch.equal(
                first_batch_counts,
                repeat_counts,
            ):
                raise RuntimeError(
                    "Router counts were not deterministic"
                )

    expected = (
        torch.tensor(valid_tokens, dtype=torch.int32)
        * top_k
    ).reshape(-1, 1)
    sums = all_counts.to(torch.int32).sum(dim=2)
    if not torch.equal(
        sums,
        expected.expand(-1, num_layers),
    ):
        raise RuntimeError("Router count totals are invalid")

    result = {
        "suite": "olmoe_trained_router_trace",
        "model": "allenai/OLMoE-1B-7B-0924",
        "model_commit": getattr(
            config,
            "_commit_hash",
            None,
        ),
        "corpus_generator": "domain_templates_v1",
        "corpus_sha256": corpus_sha256,
        "domain_ranges": domain_ranges,
        "samples_per_domain": args.samples_per_domain,
        "num_samples": len(prompts),
        "sequence_length": args.sequence_length,
        "valid_tokens": valid_tokens,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "expert_counts": all_counts.tolist(),
        "batch_size": args.batch_size,
        "batch_times_ms": batch_times_ms,
        "max_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version(
            "transformers"
        ),
        "gpu_name": torch.cuda.get_device_name(device),
        "correctness": {
            "deterministic_first_batch": True,
            "all_count_totals_exact": True,
            "expected_assignments_per_full_sample": (
                args.sequence_length * top_k
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            result,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
