#!/usr/bin/env python3
"""Immutable Recursive Self-Improvement (RSI) Kernel Verification & Vector-Pareto Ratchet Harness.

Governed by `PROGRAM.md`. Candidate kernel benchmarks import and invoke this module WITHOUT
modifying it. At runtime, `eval_harness_sha256()` hashes the LF-normalized bytes of this file
and embeds the digest in every ledger row and JSON report; `analyze_*.py` re-verifies the
on-disk digest offline to guarantee the evaluation boundary was never weakened.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

HARNESS_VERSION = "1.0.0"
HARNESS_PATH = Path(__file__).resolve()


def eval_harness_sha256() -> str:
    """LF-normalized SHA-256 of this immutable harness source file."""
    raw = HARNESS_PATH.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def hash_callable_sources(*items: Any) -> str:
    """Deterministic SHA-256 over one or more Python functions/classes/strings."""
    h = hashlib.sha256()
    for it in items:
        if isinstance(it, str):
            src = it
        else:
            try:
                src = inspect.getsource(it)
            except (OSError, TypeError):
                src = repr(it)
        h.update(src.replace("\r\n", "\n").encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()


def query_sm_telemetry(gpu_idx: int = 0) -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_idx}",
                "--query-gpu=clocks.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3.0,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return -1.0, -1.0


@dataclass(frozen=True)
class PrecisionStateSnapshot:
    allow_tf32_matmul: bool
    allow_tf32_cudnn: bool
    float32_matmul_precision: str
    grad_enabled: bool
    deterministic_algorithms: bool

    @staticmethod
    def capture() -> "PrecisionStateSnapshot":
        return PrecisionStateSnapshot(
            allow_tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
            allow_tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
            float32_matmul_precision=str(torch.get_float32_matmul_precision()),
            grad_enabled=bool(torch.is_grad_enabled()),
            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        )


def compare_tensor_to_fp64(
    cand: torch.Tensor | None,
    ref_fp64: torch.Tensor | None,
    *,
    rel_tol: float = 1.5e-2,
    zero_abs_tol: float = 1e-6,
    name: str = "tensor",
) -> dict[str, Any]:
    """Per-tensor FP64 parity check handling both nonzero and exact-zero reference regimes."""
    if ref_fp64 is None:
        ok = cand is None
        return {"name": name, "ok": ok, "rel_l2": 0.0 if ok else float("inf"), "max_abs": 0.0, "regime": "none"}
    if cand is None:
        return {"name": name, "ok": False, "rel_l2": float("inf"), "max_abs": float("inf"), "regime": "missing_grad"}

    cf = cand.detach().to(torch.float64)
    rf = ref_fp64.detach().to(torch.float64)
    if cf.shape != rf.shape:
        return {
            "name": name,
            "ok": False,
            "rel_l2": float("inf"),
            "max_abs": float("inf"),
            "regime": f"shape_mismatch:{tuple(cf.shape)}!={tuple(rf.shape)}",
        }
    if not bool(torch.isfinite(cf).all().item()):
        return {"name": name, "ok": False, "rel_l2": float("inf"), "max_abs": float("inf"), "regime": "non_finite"}

    diff = cf - rf
    max_abs = float(diff.abs().max().item()) if diff.numel() > 0 else 0.0
    ref_norm = float(torch.linalg.norm(rf).item())
    diff_norm = float(torch.linalg.norm(diff).item())

    if ref_norm <= 1e-12:
        ok = max_abs <= zero_abs_tol
        return {
            "name": name,
            "ok": ok,
            "rel_l2": 0.0 if max_abs <= zero_abs_tol else float("inf"),
            "max_abs": round(max_abs, 9),
            "regime": "exact_zero_oracle",
        }

    rel = diff_norm / ref_norm
    ok = rel <= rel_tol
    return {
        "name": name,
        "ok": ok,
        "rel_l2": round(rel, 8),
        "max_abs": round(max_abs, 8),
        "regime": "nonzero_oracle",
    }


def richardson_unit_rms_audit(
    scalar_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x0_fp64: torch.Tensor,
    analytic_grad_fp64: torch.Tensor,
    *,
    seed: int = 20261117,
    h_coarse: float = 2e-3,
    h_fine: float = 1e-3,
    ratio_bounds: tuple[float, float] = (3.90, 4.10),
    name: str = "grad",
) -> dict[str, Any]:
    """Unit-RMS central-difference O(h^2) Richardson extrapolation check (~4.000x ratio)."""
    gen = torch.Generator(device=x0_fp64.device)
    gen.manual_seed(seed)
    raw_dir = torch.randn(x0_fp64.shape, dtype=torch.float64, device=x0_fp64.device, generator=gen)
    rms = torch.sqrt(torch.mean(raw_dir * raw_dir)).clamp_min(1e-30)
    d_vec = raw_dir / rms

    analytic_dir = float(torch.sum(analytic_grad_fp64.detach().to(torch.float64) * d_vec).item())
    with torch.no_grad():
        f_p2 = float(scalar_loss_fn(x0_fp64.detach() + h_coarse * d_vec).item())
        f_m2 = float(scalar_loss_fn(x0_fp64.detach() - h_coarse * d_vec).item())
        f_p1 = float(scalar_loss_fn(x0_fp64.detach() + h_fine * d_vec).item())
        f_m1 = float(scalar_loss_fn(x0_fp64.detach() - h_fine * d_vec).item())

    fd_2h = (f_p2 - f_m2) / (2.0 * h_coarse)
    fd_1h = (f_p1 - f_m1) / (2.0 * h_fine)
    err_2h = abs(fd_2h - analytic_dir)
    err_1h = abs(fd_1h - analytic_dir)
    fd_rich = (4.0 * fd_1h - fd_2h) / 3.0
    rich_rel_err = abs(fd_rich - analytic_dir) / max(1e-12, abs(analytic_dir))

    if err_2h <= 1e-11 and err_1h <= 1e-11:
        # Exact linear/piecewise-linear derivative (zero 3rd-order term)
        ratio = 4.0
        ok = rich_rel_err <= 1e-9
        mode = "exact_linear_zero_curvature"
    else:
        ratio = err_2h / max(1e-30, err_1h)
        ok = (ratio_bounds[0] <= ratio <= ratio_bounds[1]) and (rich_rel_err <= 1e-6)
        mode = "nonlinear_oh2_richardson"

    return {
        "name": name,
        "ok": ok,
        "mode": mode,
        "err_2h": err_2h,
        "err_1h": err_1h,
        "ratio_2h_over_1h": round(ratio, 5),
        "richardson_rel_err": rich_rel_err,
    }


def benchmark_rotating_cuda(
    slot_fns: list[Callable[[], Any]],
    *,
    warmup: int = 4,
    groups: int = 7,
    iters_per_group: int = 6,
    gpu_idx: int = 0,
    l2_flush_buffer: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Stream-synchronized CUDA Event benchmark over rotating input slots with anti-cheat flag guard."""
    if not slot_fns:
        raise ValueError("slot_fns must contain at least one callable")

    snap_before = PrecisionStateSnapshot.capture()
    dev = torch.device(f"cuda:{gpu_idx}")
    n_slots = len(slot_fns)

    for w in range(warmup):
        if l2_flush_buffer is not None:
            l2_flush_buffer.zero_()
        slot_fns[w % n_slots]()
    torch.cuda.synchronize(dev)

    snap_warm = PrecisionStateSnapshot.capture()
    if snap_warm != snap_before:
        raise RuntimeError(f"Precision/autograd state mutated during warmup: {snap_before} -> {snap_warm}")

    torch.cuda.reset_peak_memory_stats(dev)
    slot_fns[0]()
    torch.cuda.synchronize(dev)
    peak_alloc_mib = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
    peak_reserved_mib = torch.cuda.max_memory_reserved(dev) / (1024.0 * 1024.0)

    samples_ms: list[float] = []
    step_counter = 0
    for _ in range(groups):
        if l2_flush_buffer is not None:
            l2_flush_buffer.zero_()
            torch.cuda.synchronize(dev)
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for _ in range(iters_per_group):
            slot_fns[step_counter % n_slots]()
            step_counter += 1
        e_ev.record()
        e_ev.synchronize()
        samples_ms.append(s_ev.elapsed_time(e_ev) / float(iters_per_group))

    torch.cuda.synchronize(dev)
    snap_after = PrecisionStateSnapshot.capture()
    if snap_after != snap_before:
        raise RuntimeError(f"Precision/autograd state mutated during timed loop: {snap_before} -> {snap_after}")

    ss = sorted(samples_ms)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    clk, pwr = query_sm_telemetry(gpu_idx)
    return {
        "median_ms": round(float(statistics.median(ss)), 4),
        "iqr_ms": round(float(q75 - q25), 4),
        "min_ms": round(float(ss[0]), 4),
        "max_ms": round(float(ss[-1]), 4),
        "peak_alloc_mib": round(float(peak_alloc_mib), 2),
        "peak_reserved_mib": round(float(peak_reserved_mib), 2),
        "sm_clock_mhz": clk,
        "power_w": pwr,
        "num_rotation_slots": n_slots,
    }


@dataclass
class RSITrialRecord:
    track: str
    trial_id: int
    candidate_id: str
    candidate_sha256: str
    eval_harness_sha256: str
    correctness_pass: bool
    anti_cheat_pass: bool
    max_fp64_rel_l2: float
    min_richardson_ratio: float
    max_richardson_ratio: float
    geomean_ms: float
    peak_alloc_mib: float
    worst_shape_ratio_vs_baseline: float
    speedup_vs_baseline: float
    speedup_vs_incumbent: float
    decision: str
    diagnosis: str
    per_shape_ms: dict[str, float] = field(default_factory=dict)


class RSIVectorRatchet:
    """Append-only multi-shape Pareto ratchet enforcing correctness, anti-cheat, and zero shape regressions."""

    def __init__(
        self,
        track: str,
        shape_keys: list[str],
        *,
        hbm_cap_mib: float = 76000.0,
        min_gain_tol: float = 0.01,
        max_shape_regression_tol: float = 0.02,
    ) -> None:
        self.track = track
        self.shape_keys = list(shape_keys)
        self.hbm_cap_mib = hbm_cap_mib
        self.min_gain_tol = min_gain_tol
        self.max_shape_regression_tol = max_shape_regression_tol
        self.harness_sha = eval_harness_sha256()
        self.baseline_record: RSITrialRecord | None = None
        self.incumbent_record: RSITrialRecord | None = None
        self.trials: list[RSITrialRecord] = []

    @staticmethod
    def _geomean(vals: list[float]) -> float:
        return math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / float(len(vals)))

    def record_compile_or_runtime_failure(
        self,
        *,
        trial_id: int,
        candidate_id: str,
        candidate_sha256: str,
        error_summary: str,
    ) -> RSITrialRecord:
        rec = RSITrialRecord(
            track=self.track,
            trial_id=trial_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            eval_harness_sha256=self.harness_sha,
            correctness_pass=False,
            anti_cheat_pass=True,
            max_fp64_rel_l2=float("inf"),
            min_richardson_ratio=0.0,
            max_richardson_ratio=0.0,
            geomean_ms=float("inf"),
            peak_alloc_mib=0.0,
            worst_shape_ratio_vs_baseline=float("inf"),
            speedup_vs_baseline=0.0,
            speedup_vs_incumbent=0.0,
            decision="REJECT_COMPILE_OR_RUNTIME",
            diagnosis=error_summary,
            per_shape_ms={},
        )
        self.trials.append(rec)
        return rec

    def evaluate_and_record(
        self,
        *,
        trial_id: int,
        candidate_id: str,
        candidate_sha256: str,
        parity_checks: list[dict[str, Any]],
        richardson_checks: list[dict[str, Any]],
        anti_cheat_checks: dict[str, bool],
        per_shape_metrics: dict[str, dict[str, Any]],
        hypothesis_note: str,
        allow_pruned_branch_stage: bool = False,
    ) -> RSITrialRecord:
        missing = [k for k in self.shape_keys if k not in per_shape_metrics]
        if missing:
            raise ValueError(f"Candidate {candidate_id} missing required target shapes: {missing}")

        parity_ok = all(bool(c.get("ok", False)) for c in parity_checks)
        rich_ok = all(bool(c.get("ok", False)) for c in richardson_checks) if richardson_checks else True
        correctness_ok = parity_ok and rich_ok
        anti_cheat_ok = all(bool(v) for v in anti_cheat_checks.values())

        finite_rels = [float(c["rel_l2"]) for c in parity_checks if math.isfinite(float(c.get("rel_l2", 0.0)))]
        max_rel = max(finite_rels) if (parity_ok and finite_rels) else (0.0 if parity_ok else float("inf"))
        if richardson_checks:
            r_vals = [float(c["ratio_2h_over_1h"]) for c in richardson_checks]
            min_r, max_r = min(r_vals), max(r_vals)
        else:
            min_r = max_r = 4.0

        shape_ms = {k: float(per_shape_metrics[k]["median_ms"]) for k in self.shape_keys}
        gmean_ms = round(self._geomean([shape_ms[k] for k in self.shape_keys]), 4)
        peak_mib = round(max(float(per_shape_metrics[k]["peak_alloc_mib"]) for k in self.shape_keys), 2)

        if self.baseline_record is None:
            if not (correctness_ok and anti_cheat_ok and peak_mib <= self.hbm_cap_mib):
                raise RuntimeError(
                    f"Baseline {candidate_id} failed verification gates: "
                    f"correctness={correctness_ok}, anti_cheat={anti_cheat_ok}, peak_mib={peak_mib}"
                )
            rec = RSITrialRecord(
                track=self.track,
                trial_id=trial_id,
                candidate_id=candidate_id,
                candidate_sha256=candidate_sha256,
                eval_harness_sha256=self.harness_sha,
                correctness_pass=True,
                anti_cheat_pass=True,
                max_fp64_rel_l2=round(max_rel, 6),
                min_richardson_ratio=round(min_r, 4),
                max_richardson_ratio=round(max_r, 4),
                geomean_ms=gmean_ms,
                peak_alloc_mib=peak_mib,
                worst_shape_ratio_vs_baseline=1.0,
                speedup_vs_baseline=1.0,
                speedup_vs_incumbent=1.0,
                decision="BASELINE",
                diagnosis=hypothesis_note,
                per_shape_ms=shape_ms,
            )
            self.baseline_record = rec
            self.incumbent_record = rec
            self.trials.append(rec)
            return rec

        assert self.incumbent_record is not None
        base_ms = self.baseline_record.per_shape_ms
        shape_ratios = {k: shape_ms[k] / max(base_ms[k], 1e-12) for k in self.shape_keys}
        worst_key = max(shape_ratios, key=lambda k: shape_ratios[k])
        worst_ratio = shape_ratios[worst_key]
        sp_base = self.baseline_record.geomean_ms / max(gmean_ms, 1e-12)
        sp_inc = self.incumbent_record.geomean_ms / max(gmean_ms, 1e-12)

        if not anti_cheat_ok:
            failed_ac = [k for k, v in anti_cheat_checks.items() if not v]
            decision = "REJECT_ANTI_CHEAT"
            diag = f"Failed anti-cheat invariants {failed_ac}; {hypothesis_note}"
        elif not correctness_ok:
            failed_p = [c["name"] for c in parity_checks if not c.get("ok", False)]
            failed_r = [c["name"] for c in richardson_checks if not c.get("ok", False)]
            decision = "REJECT_CORRECTNESS"
            diag = f"Failed FP64/Richardson gates parity={failed_p} richardson={failed_r}; {hypothesis_note}"
        elif peak_mib > self.hbm_cap_mib:
            decision = "REJECT_MEMORY"
            diag = f"Peak HBM {peak_mib:.1f} MiB > cap {self.hbm_cap_mib:.1f} MiB; {hypothesis_note}"
        elif worst_ratio > (1.0 + self.max_shape_regression_tol):
            decision = "REJECT_SHAPE_REGRESSION"
            diag = (
                f"Shape {worst_key} regressed {worst_ratio:.3f}x vs baseline "
                f"({shape_ms[worst_key]:.3f}ms vs {base_ms[worst_key]:.3f}ms); {hypothesis_note}"
            )
        elif gmean_ms <= self.incumbent_record.geomean_ms * (1.0 - self.min_gain_tol):
            decision = "KEEP" if not allow_pruned_branch_stage else "KEEP_TERMINAL_STAGE"
            diag = (
                f"Ratcheted {sp_inc:.3f}x over incumbent ({sp_base:.3f}x vs baseline, "
                f"worst shape {worst_key} {worst_ratio:.3f}x); {hypothesis_note}"
            )
        else:
            decision = "REVERT_SLOWER"
            diag = (
                f"Geomean {gmean_ms:.3f}ms did not beat incumbent "
                f"{self.incumbent_record.geomean_ms:.3f}ms by >={self.min_gain_tol * 100:.1f}%; {hypothesis_note}"
            )

        rec = RSITrialRecord(
            track=self.track,
            trial_id=trial_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            eval_harness_sha256=self.harness_sha,
            correctness_pass=correctness_ok,
            anti_cheat_pass=anti_cheat_ok,
            max_fp64_rel_l2=round(max_rel, 6) if math.isfinite(max_rel) else -1.0,
            min_richardson_ratio=round(min_r, 4),
            max_richardson_ratio=round(max_r, 4),
            geomean_ms=gmean_ms,
            peak_alloc_mib=peak_mib,
            worst_shape_ratio_vs_baseline=round(worst_ratio, 4),
            speedup_vs_baseline=round(sp_base, 4),
            speedup_vs_incumbent=round(sp_inc, 4),
            decision=decision,
            diagnosis=diag,
            per_shape_ms=shape_ms,
        )
        if decision in ("KEEP", "KEEP_TERMINAL_STAGE"):
            self.incumbent_record = rec
        self.trials.append(rec)
        return rec

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "harness_version": HARNESS_VERSION,
            "eval_harness_sha256": self.harness_sha,
            "track": self.track,
            "shape_keys": self.shape_keys,
            "hbm_cap_mib": self.hbm_cap_mib,
            "min_gain_tol": self.min_gain_tol,
            "max_shape_regression_tol": self.max_shape_regression_tol,
            "baseline_candidate": self.baseline_record.candidate_id if self.baseline_record else None,
            "final_incumbent_candidate": self.incumbent_record.candidate_id if self.incumbent_record else None,
            "final_speedup_vs_baseline": (
                round(self.baseline_record.geomean_ms / self.incumbent_record.geomean_ms, 4)
                if (self.baseline_record and self.incumbent_record)
                else 1.0
            ),
            "trials": [asdict(t) for t in self.trials],
        }


def write_rsi_ledger_tsv(ratchets: list[RSIVectorRatchet], out_path: Path) -> str:
    """Writes deterministic LF-terminated `rsi_ledger.tsv` across all tracks and returns its SHA-256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track",
        "trial_id",
        "candidate_id",
        "candidate_sha256",
        "eval_harness_sha256",
        "correctness_pass",
        "anti_cheat_pass",
        "max_fp64_rel_l2",
        "min_richardson_ratio",
        "max_richardson_ratio",
        "geomean_ms",
        "peak_alloc_mib",
        "worst_shape_ratio_vs_baseline",
        "speedup_vs_baseline",
        "speedup_vs_incumbent",
        "decision",
        "per_shape_ms_json",
        "diagnosis",
    ]
    lines: list[list[Any]] = []
    for r in ratchets:
        for t in r.trials:
            import json as _json

            lines.append(
                [
                    t.track,
                    t.trial_id,
                    t.candidate_id,
                    t.candidate_sha256[:16],
                    t.eval_harness_sha256[:16],
                    int(t.correctness_pass),
                    int(t.anti_cheat_pass),
                    t.max_fp64_rel_l2,
                    t.min_richardson_ratio,
                    t.max_richardson_ratio,
                    t.geomean_ms,
                    t.peak_alloc_mib,
                    t.worst_shape_ratio_vs_baseline,
                    t.speedup_vs_baseline,
                    t.speedup_vs_incumbent,
                    t.decision,
                    _json.dumps(t.per_shape_ms, sort_keys=True),
                    t.diagnosis,
                ]
            )
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(fieldnames)
        writer.writerows(lines)
    return hashlib.sha256(out_path.read_bytes()).hexdigest()
