#!/usr/bin/env python3
"""Extract ICR Probe features for the frozen SciFact comparison.

The ICR score follows the official ACL 2025 implementation at commit
40ec490e762cadbac6bcefdc24a8f0d5974e8448. This adapter preserves its
per-layer hidden-update/attention JSD, induction-head selection, response-only
mask, top-p setting, and token-mean pooling. The only protocol adaptation is a
single teacher-forced pass over the fixed candidate instead of autoregressive
generation, matching the ECLT candidate-verification input.

Upstream project: https://github.com/XavierZhang2002/ICR_Probe
Upstream license: Apache-2.0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SCHEMA_VERSION = "icassp-icr-features-v1"
UPSTREAM_COMMIT = "40ec490e762cadbac6bcefdc24a8f0d5974e8448"


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").strip().split())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield row


def resolve_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def exact_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    import re

    value = clean(value)
    if not value:
        return []
    return [(match.start(), match.end()) for match in re.finditer(re.escape(value), text, re.I)]


def candidate_centered_truncate(text: str, value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "\n[... evidence omitted by the fixed context budget ...]\n"
    occurrences = exact_occurrences(text, value)
    if not occurrences:
        keep = max(1, max_chars - len(marker))
        head = int(keep * 0.65)
        return text[:head] + marker + text[-(keep - head) :], True
    center = sum(occurrences[0]) // 2
    body = max(len(clean(value)), max_chars - 2 * len(marker))
    left = max(0, center - body // 2)
    right = min(len(text), left + body)
    left = max(0, right - body)
    return (marker if left else "") + text[left:right] + (marker if right < len(text) else ""), True


def build_prompt(evidence: str, candidate: str) -> str:
    return (
        "Assess whether the evidence supports the candidate claim.\n"
        "Use only the evidence. Do not rely on outside knowledge.\n"
        f"Candidate claim: {candidate}\n"
        "<EVIDENCE>\n"
        f"{evidence}\n"
        "</EVIDENCE>\n"
        "Reconstruct the candidate claim:"
    )


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    """Official ICR standardize-softmax Jensen-Shannon divergence."""

    left = (left - left.mean()) / torch.clamp(left.std(), min=1e-8)
    right = (right - right.mean()) / torch.clamp(right.std(), min=1e-8)
    left = F.softmax(left, dim=0)
    right = F.softmax(right, dim=0)
    middle = 0.5 * (left + right)
    left_kl = torch.sum(left * torch.log(torch.clamp(left / middle, min=1e-12)))
    right_kl = torch.sum(right * torch.log(torch.clamp(right / middle, min=1e-12)))
    return float((0.5 * left_kl + 0.5 * right_kl).item())


def block_average_skew(attention: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate official row-wise attention skew without materializing a mask."""

    heads = attention.shape[0]
    total = torch.zeros(heads, dtype=torch.float64, device=attention.device)
    count = torch.zeros(heads, dtype=torch.float64, device=attention.device)
    if end <= start:
        return total, count
    key_index = torch.arange(1, end - start + 1, device=attention.device, dtype=torch.float32)
    for query_start in range(start, end, 128):
        query_end = min(end, query_start + 128)
        block = attention[:, query_start:query_end, start:end].to(torch.float32)
        row_sum = block.sum(dim=-1, keepdim=True)
        valid = row_sum.squeeze(-1) > 0
        probability = block / (row_sum + 1e-12)
        mean = torch.sum(probability * key_index, dim=-1)
        centered = key_index.view(1, 1, -1) - mean.unsqueeze(-1)
        variance = torch.sum(centered.square() * probability, dim=-1)
        third = torch.sum(centered.pow(3) * probability, dim=-1)
        skew = third / (variance.pow(1.5) + 1e-12)
        total += torch.where(valid, skew, torch.zeros_like(skew)).sum(dim=-1).to(torch.float64)
        count += valid.sum(dim=-1).to(torch.float64)
        del block, row_sum, probability, mean, centered, variance, third, skew
    return total, count


def select_induction_heads(attention: torch.Tensor, response_start: int) -> torch.Tensor:
    """Match the official skew>=0 selection and top one-eighth fallback."""

    prompt_sum, prompt_count = block_average_skew(attention, 0, response_start)
    response_sum, response_count = block_average_skew(attention, response_start, attention.shape[-1])
    average = (prompt_sum + response_sum) / torch.clamp(prompt_count + response_count, min=1.0)
    selected = average >= 0
    minimum = max(1, attention.shape[0] // 8)
    if int(selected.sum().item()) < minimum:
        selected.zero_()
        selected[torch.topk(average, k=minimum).indices] = True
    return selected


class ICRExtractor:
    def __init__(
        self,
        model_path: str,
        *,
        dtype: str,
        max_input_tokens: int,
        max_context_chars: int,
        top_p: float,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": resolve_dtype(dtype),
            "device_map": {"": 0} if torch.cuda.is_available() else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self.base_model = getattr(self.model, "model", None)
        if self.base_model is None:
            raise ValueError("causal LM has no .model base transformer")
        self.device = next(self.model.parameters()).device
        self.max_input_tokens = max_input_tokens
        self.max_context_chars = max_context_chars
        self.top_p = top_p

    def encode(self, evidence: str, candidate: str) -> dict[str, Any]:
        evidence, truncated = candidate_centered_truncate(
            evidence, candidate, self.max_context_chars
        )
        prompt = build_prompt(evidence, candidate)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(" " + candidate, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError("candidate tokenization is empty")
        if len(prompt_ids) + len(target_ids) > self.max_input_tokens:
            ratio = self.max_input_tokens / max(1, len(prompt_ids) + len(target_ids))
            tighter = max(512, int(len(evidence) * ratio * 0.85))
            evidence, tightened = candidate_centered_truncate(evidence, candidate, tighter)
            truncated = truncated or tightened
            prompt = build_prompt(evidence, candidate)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) + len(target_ids) > self.max_input_tokens:
            raise ValueError(
                f"input over budget: prompt={len(prompt_ids)} target={len(target_ids)} "
                f"max={self.max_input_tokens}"
            )
        return {
            "input_ids": torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=self.device),
            "prompt_tokens": len(prompt_ids),
            "target_ids": [int(value) for value in target_ids],
            "context_chars": len(evidence),
            "context_truncated": truncated,
        }

    @torch.inference_mode()
    def extract(self, evidence: str, candidate: str) -> dict[str, Any]:
        encoded = self.encode(evidence, candidate)
        outputs = self.base_model(
            input_ids=encoded["input_ids"],
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if not hidden_states or not attentions or len(hidden_states) != len(attentions) + 1:
            raise ValueError("model did not return a complete hidden-state/attention stack")

        response_start = int(encoded["prompt_tokens"])
        response_positions = list(range(response_start, encoded["input_ids"].shape[1]))
        layer_scores: list[float] = []
        selected_head_counts: list[int] = []
        for layer_index, layer_attention in enumerate(attentions):
            attention = layer_attention[0]
            selected_heads = select_induction_heads(attention, response_start)
            selected_head_counts.append(int(selected_heads.sum().item()))
            pooled = attention[selected_heads].mean(dim=0)
            token_scores = []
            for token_position in response_positions:
                row = pooled[token_position].to(torch.float32).clone()
                row[:response_start] = 0
                if token_position + 1 < row.numel():
                    row[token_position + 1 :] = 0
                count = max(1, int(self.top_p * row.numel()))
                count = min(count, row.numel())
                attention_values, source_positions = torch.topk(row, k=count)
                previous = hidden_states[layer_index][0, token_position].to(torch.float32)
                current = hidden_states[layer_index + 1][0, token_position].to(torch.float32)
                update = current - previous
                source_hidden = hidden_states[layer_index][0].index_select(0, source_positions).to(torch.float32)
                contribution = torch.sum(update.unsqueeze(0) * source_hidden, dim=-1)
                contribution /= torch.linalg.vector_norm(source_hidden, dim=-1) + 1e-8
                token_scores.append(js_divergence(contribution, attention_values))
                del row, attention_values, source_positions, previous, current, update, source_hidden, contribution
            layer_scores.append(float(np.mean(token_scores)))
            del attention, selected_heads, pooled, token_scores

        predictor_positions = torch.tensor(
            list(range(response_start - 1, response_start + len(encoded["target_ids"]) - 1)),
            dtype=torch.long,
            device=self.device,
        )
        final_hidden = outputs.last_hidden_state[0].index_select(0, predictor_positions)
        logits = self.model.lm_head(final_hidden).to(torch.float32)
        target = torch.tensor(encoded["target_ids"], dtype=torch.long, device=self.device)
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs)
        token_logprob = log_probs.gather(1, target[:, None]).squeeze(1)
        entropy = torch.sum(-probabilities * log_probs, dim=-1)
        top2 = torch.topk(probabilities, k=min(2, probabilities.shape[-1]), dim=-1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]

        result = {
            "prompt_tokens": response_start,
            "target_tokens": len(encoded["target_ids"]),
            "context_chars": encoded["context_chars"],
            "context_truncated": encoded["context_truncated"],
            "icr_score_by_layer": layer_scores,
            "selected_head_count_by_layer": selected_head_counts,
            "confidence": {
                "avg_logprob": float(token_logprob.mean().item()),
                "min_logprob": float(token_logprob.min().item()),
                "avg_entropy": float(entropy.mean().item()),
                "avg_top2_margin": float(margin.mean().item()),
            },
        }
        del outputs, hidden_states, attentions, final_hidden, logits, target, log_probs, probabilities
        del token_logprob, entropy, top2, margin
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    args = parser.parse_args()

    run_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = set()
    if args.output.exists():
        processed = {clean(row.get("claim_id")) for row in iter_jsonl(args.output)}

    extractor = ICRExtractor(
        args.model_path,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_context_chars=args.max_context_chars,
        top_p=args.top_p,
    )
    written = 0
    errors = 0
    with args.output.open("a", encoding="utf-8") as output:
        for input_index, row in enumerate(iter_jsonl(args.input), 1):
            if args.max_records and input_index > args.max_records:
                break
            claim_id = clean(row.get("claim_id"))
            if not claim_id or claim_id in processed:
                continue
            result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "claim_id": claim_id,
                "record_id": clean(row.get("record_id")),
                "partition": clean(row.get("partition")),
                "attribute": clean(row.get("attribute")),
                "candidate_value": clean(row.get("candidate_value")),
                "labels": row.get("labels") or {},
                "audit_metadata": row.get("audit_metadata") or {},
                "status": "ok",
                "extractor_config": {
                    "model_path": args.model_path,
                    "prompt_mode": "claim_verification",
                    "source_method": "ICR Probe",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "attention_implementation": "eager",
                    "top_p": args.top_p,
                    "head_pooling": "mean",
                    "use_induction_heads": True,
                    "skew_threshold": 0.0,
                    "entropy_threshold": 100000.0,
                    "token_pooling": "mean",
                    "response_only_mask": True,
                    "teacher_forced_candidate": True,
                },
            }
            try:
                result["features"] = extractor.extract(
                    str(row.get("evidence_text") or ""), result["candidate_value"]
                )
            except Exception as exc:
                errors += 1
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
                if args.fail_fast:
                    raise
            output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            processed.add(claim_id)
            written += 1
            if written % args.checkpoint_every == 0:
                print(f"[checkpoint] written={written} errors={errors} output={args.output}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "written_this_run": written,
        "errors_this_run": errors,
        "processed_total": len(processed),
        "upstream_commit": UPSTREAM_COMMIT,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - run_started,
            "rows_per_second_this_run": written / max(1e-9, time.perf_counter() - run_started),
            "peak_cuda_allocated_mb": (
                torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
            ),
            "peak_cuda_reserved_mb": (
                torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else None
            ),
        },
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if errors / max(1, written) <= args.max_error_rate else 2


if __name__ == "__main__":
    raise SystemExit(main())
