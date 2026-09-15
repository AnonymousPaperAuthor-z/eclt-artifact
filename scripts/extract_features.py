#!/usr/bin/env python3
"""Extract memory-bounded evidence-conditioned layer trajectory features.

Forward hooks pool only selected layers and positions.
It performs one teacher-forced pass per claim and does not use source masking.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SCHEMA_VERSION = "icassp-eclt-features-v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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
    name = str(name).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(name)


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(getattr(model, "gpt_neox", None), "layers", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError("cannot locate decoder layers for selected-layer hooks")


def find_base_model(model: torch.nn.Module) -> torch.nn.Module:
    base = getattr(model, "model", None)
    if base is None:
        raise ValueError("causal LM has no .model base transformer")
    return base


def exact_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    value = clean(value)
    if not value:
        return []
    return [(match.start(), match.end()) for match in re.finditer(re.escape(value), text, re.IGNORECASE)]


def candidate_centered_truncate(text: str, value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    occurrences = exact_occurrences(text, value)
    marker = "\n[... evidence omitted by the fixed context budget ...]\n"
    if occurrences:
        center = (occurrences[0][0] + occurrences[0][1]) // 2
        # Reserve markers for both sides, then center the remaining body on the
        # exact candidate occurrence so tightening never removes the target span.
        body_budget = max(len(value), max_chars - 2 * len(marker))
        left = max(0, center - body_budget // 2)
        right = min(len(text), left + body_budget)
        left = max(0, right - body_budget)
        prefix = marker if left > 0 else ""
        suffix = marker if right < len(text) else ""
        available = max(len(value), max_chars - len(prefix) - len(suffix))
        left = max(0, center - available // 2)
        right = min(len(text), left + available)
        left = max(0, right - available)
        return prefix + text[left:right] + suffix, True
    keep = max(1, max_chars - len(marker))
    head = int(keep * 0.65)
    tail = keep - head
    return text[:head] + marker + text[-tail:], True


def overlap_positions(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    positions = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            positions.append(idx)
    return positions


def safe_mean(hidden: torch.Tensor, positions: list[int]) -> torch.Tensor | None:
    if not positions:
        return None
    valid = [pos for pos in positions if 0 <= pos < hidden.shape[1]]
    if not valid:
        return None
    index = torch.tensor(valid, dtype=torch.long, device=hidden.device)
    return hidden[0].index_select(0, index).to(torch.float32).mean(dim=0)


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 0:
        return None
    return float(torch.dot(a, b).item() / denom.item())


def normalized_l2(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float(torch.linalg.vector_norm(a - b).item() / math.sqrt(max(1, a.numel())))


def compact_topk(vector: torch.Tensor | None, k: int) -> dict[str, list[float] | list[int]]:
    if vector is None or k <= 0:
        return {"indices": [], "values": []}
    count = min(k, vector.numel())
    _, indices = torch.topk(torch.abs(vector), k=count)
    values = vector.index_select(0, indices)
    return {
        "indices": [int(x) for x in indices.detach().cpu().tolist()],
        "values": [round(float(x), 6) for x in values.detach().cpu().tolist()],
    }


def trajectory_summary(layers: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [(float(row["fraction"]), row.get(key)) for row in layers if row.get(key) is not None]
    if not values:
        return {"first": None, "last": None, "mean": None, "range": None, "slope": None, "auc": None}
    xs = torch.tensor([item[0] for item in values], dtype=torch.float64)
    ys = torch.tensor([float(item[1]) for item in values], dtype=torch.float64)
    slope = None
    if len(values) >= 2 and float(torch.var(xs, unbiased=False).item()) > 0:
        slope = float(torch.sum((xs - xs.mean()) * (ys - ys.mean())).item() / torch.sum((xs - xs.mean()) ** 2).item())
    auc = float(torch.trapz(ys, xs).item()) if len(values) >= 2 else float(ys[0].item())
    return {
        "first": float(ys[0].item()),
        "last": float(ys[-1].item()),
        "mean": float(ys.mean().item()),
        "range": float((ys.max() - ys.min()).item()),
        "slope": slope,
        "auc": auc,
    }


class SelectedLayerTrajectoryExtractor:
    def __init__(
        self,
        model_path: str,
        *,
        layer_fractions: list[float],
        dtype: str,
        projection_dim: int,
        projection_seed: int,
        top_neurons: int,
        max_input_tokens: int,
        max_context_chars: int,
        attn_implementation: str,
        prompt_mode: str,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": resolve_dtype(dtype),
            "device_map": {"": 0} if torch.cuda.is_available() else None,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except (TypeError, ValueError):
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.attn_implementation = load_kwargs.get("attn_implementation") or "model_default"
        self.model.eval()
        self.base_model = find_base_model(self.model)
        self.layers = find_decoder_layers(self.model)
        self.device = next(self.model.parameters()).device
        self.max_input_tokens = max_input_tokens
        self.max_context_chars = max_context_chars
        self.projection_dim = projection_dim
        self.top_neurons = top_neurons
        self.prompt_mode = prompt_mode

        selected = set()
        for fraction in layer_fractions:
            idx = int(round(max(0.0, min(1.0, fraction)) * (len(self.layers) - 1)))
            selected.add(idx)
        self.layer_indices = sorted(selected)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(projection_seed)
        hidden_size = int(getattr(self.model.config, "hidden_size"))
        projection = torch.randn(hidden_size, projection_dim, generator=generator, dtype=torch.float32)
        projection /= math.sqrt(max(1, projection_dim))
        self.projection = projection.to(self.device)
        self._capture_positions: dict[str, list[int]] = {}
        self._captured: dict[int, dict[str, Any]] = {}
        self._handles = []
        for idx in self.layer_indices:
            self._handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                return
            vectors = {
                name: safe_mean(hidden, positions)
                for name, positions in self._capture_positions.items()
            }
            predictor = vectors.get("predictor")
            candidate = vectors.get("candidate")
            evidence = vectors.get("evidence_value")
            context = vectors.get("evidence_context")
            projected: dict[str, list[float]] = {}
            for name in ("predictor", "candidate", "evidence_value"):
                vector = vectors.get(name)
                if vector is not None:
                    projected[name] = [
                        round(float(x), 6)
                        for x in torch.matmul(vector, self.projection).detach().cpu().tolist()
                    ]
            if predictor is not None and evidence is not None:
                projected["predictor_minus_evidence"] = [
                    round(float(x), 6)
                    for x in torch.matmul(predictor - evidence, self.projection).detach().cpu().tolist()
                ]
            self._captured[layer_idx] = {
                "predictor_evidence_cos": cosine(predictor, evidence),
                "candidate_evidence_cos": cosine(candidate, evidence),
                "predictor_context_cos": cosine(predictor, context),
                "candidate_context_cos": cosine(candidate, context),
                "predictor_evidence_l2": normalized_l2(predictor, evidence),
                "candidate_evidence_l2": normalized_l2(candidate, evidence),
                "predictor_norm": float(torch.linalg.vector_norm(predictor).item()) if predictor is not None else None,
                "candidate_norm": float(torch.linalg.vector_norm(candidate).item()) if candidate is not None else None,
                "evidence_norm": float(torch.linalg.vector_norm(evidence).item()) if evidence is not None else None,
                "projected": projected,
                "top_residual_dimensions": {
                    "predictor": compact_topk(predictor, self.top_neurons),
                    "candidate": compact_topk(candidate, self.top_neurons),
                    "evidence": compact_topk(evidence, self.top_neurons),
                },
            }
        return hook

    def _build_prompt(self, evidence: str, attribute: str, candidate: str) -> tuple[str, int, int]:
        if self.prompt_mode == "network_asset":
            prefix = (
                "Assess whether a network observation supports a candidate device attribute.\n"
                "Use only the observation. Component, server, library, and cache values do not "
                "automatically describe the endpoint.\n"
                f"Requested attribute: {attribute}\n"
                f"Candidate value: {candidate}\n"
                "<EVIDENCE>\n"
            )
            suffix = "\n</EVIDENCE>\nReconstruct the candidate value:"
        elif self.prompt_mode == "claim_verification":
            prefix = (
                "Assess whether the evidence supports the candidate claim.\n"
                "Use only the evidence. Do not rely on outside knowledge.\n"
                f"Candidate claim: {candidate}\n"
                "<EVIDENCE>\n"
            )
            suffix = "\n</EVIDENCE>\nReconstruct the candidate claim:"
        else:
            raise ValueError(f"unsupported prompt mode: {self.prompt_mode}")
        prompt = prefix + evidence + suffix
        return prompt, len(prefix), len(prefix) + len(evidence)

    def _encode(self, evidence: str, attribute: str, candidate: str) -> dict[str, Any]:
        evidence, char_truncated = candidate_centered_truncate(
            evidence, candidate, self.max_context_chars
        )
        prompt, evidence_start, evidence_end = self._build_prompt(evidence, attribute, candidate)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        target_ids = self.tokenizer(" " + candidate, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError("candidate tokenization is empty")

        if len(prompt_ids) + len(target_ids) > self.max_input_tokens:
            # The source datasets normally contain short, focused contexts. This
            # fallback tightens the character budget while preserving a value occurrence.
            ratio = self.max_input_tokens / max(1, len(prompt_ids) + len(target_ids))
            tighter = max(512, int(len(evidence) * ratio * 0.85))
            evidence, extra_truncated = candidate_centered_truncate(evidence, candidate, tighter)
            char_truncated = char_truncated or extra_truncated
            prompt, evidence_start, evidence_end = self._build_prompt(evidence, attribute, candidate)
            encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            prompt_ids = list(encoded["input_ids"])
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if len(prompt_ids) + len(target_ids) > self.max_input_tokens:
            raise ValueError(
                f"input remains over budget: prompt={len(prompt_ids)} target={len(target_ids)} "
                f"max={self.max_input_tokens}"
            )

        occurrence_spans = [
            (evidence_start + start, evidence_start + end)
            for start, end in exact_occurrences(evidence, candidate)
        ]
        context_span = [(evidence_start, evidence_end)]
        evidence_positions = overlap_positions(offsets, occurrence_spans)
        context_positions = overlap_positions(offsets, context_span)
        target_start = len(prompt_ids)
        candidate_positions = list(range(target_start, target_start + len(target_ids)))
        predictor_positions = list(range(target_start - 1, target_start + len(target_ids) - 1))
        input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=self.device)
        return {
            "input_ids": input_ids,
            "prompt_tokens": len(prompt_ids),
            "target_ids": [int(x) for x in target_ids],
            "candidate_positions": candidate_positions,
            "predictor_positions": predictor_positions,
            "evidence_positions": evidence_positions,
            "context_positions": context_positions,
            "occurrence_count": len(occurrence_spans),
            "context_chars": len(evidence),
            "context_truncated": char_truncated,
        }

    @torch.inference_mode()
    def extract(self, evidence: str, attribute: str, candidate: str) -> dict[str, Any]:
        encoded = self._encode(evidence, attribute, candidate)
        self._capture_positions = {
            "predictor": encoded["predictor_positions"],
            "candidate": encoded["candidate_positions"],
            "evidence_value": encoded["evidence_positions"],
            "evidence_context": encoded["context_positions"],
        }
        self._captured = {}
        outputs = self.base_model(
            input_ids=encoded["input_ids"],
            use_cache=False,
            return_dict=True,
        )
        final_hidden = outputs.last_hidden_state[0]
        predictor_index = torch.tensor(encoded["predictor_positions"], dtype=torch.long, device=self.device)
        predictor_hidden = final_hidden.index_select(0, predictor_index)
        logits = self.model.lm_head(predictor_hidden).to(torch.float32)
        target = torch.tensor(encoded["target_ids"], dtype=torch.long, device=self.device)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_logprob = log_probs.gather(1, target[:, None]).squeeze(1)
        entropy = torch.sum(-probs * log_probs, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]

        layers = []
        denom = max(1, len(self.layers) - 1)
        for idx in self.layer_indices:
            row = dict(self._captured.get(idx) or {})
            row["layer"] = idx
            row["fraction"] = idx / denom
            layers.append(row)
        scalar_keys = [
            "predictor_evidence_cos",
            "candidate_evidence_cos",
            "predictor_context_cos",
            "candidate_context_cos",
            "predictor_evidence_l2",
            "candidate_evidence_l2",
        ]
        return {
            "prompt_tokens": encoded["prompt_tokens"],
            "target_tokens": len(encoded["target_ids"]),
            "context_chars": encoded["context_chars"],
            "context_truncated": encoded["context_truncated"],
            "evidence_span_present": bool(encoded["evidence_positions"]),
            "evidence_occurrence_count": encoded["occurrence_count"],
            "confidence": {
                "avg_logprob": float(token_logprob.mean().item()),
                "min_logprob": float(token_logprob.min().item()),
                "avg_entropy": float(entropy.mean().item()),
                "avg_top2_margin": float(margin.mean().item()),
            },
            "layer_trajectory": layers,
            "trajectory_summary": {key: trajectory_summary(layers, key) for key in scalar_keys},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--layer-fractions", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260806)
    parser.add_argument("--top-neurons", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--prompt-mode",
        choices=("network_asset", "claim_verification"),
        default="network_asset",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    args = parser.parse_args()

    run_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = set()
    if args.output.exists():
        for row in iter_jsonl(args.output):
            claim_id = clean(row.get("claim_id"))
            if claim_id:
                processed.add(claim_id)

    fractions = [float(value) for value in args.layer_fractions.split(",") if value.strip()]
    extractor = SelectedLayerTrajectoryExtractor(
        args.model_path,
        layer_fractions=fractions,
        dtype=args.dtype,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        top_neurons=args.top_neurons,
        max_input_tokens=args.max_input_tokens,
        max_context_chars=args.max_context_chars,
        attn_implementation=args.attn_implementation,
        prompt_mode=args.prompt_mode,
    )
    written = 0
    errors = 0
    try:
        with args.output.open("a", encoding="utf-8") as out:
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
                        "layer_fractions": fractions,
                        "selected_layers": extractor.layer_indices,
                        "projection_dim": args.projection_dim,
                        "projection_seed": args.projection_seed,
                        "max_input_tokens": args.max_input_tokens,
                        "max_context_chars": args.max_context_chars,
                        "source_masking": False,
                        "attention_implementation": extractor.attn_implementation,
                        "prompt_mode": args.prompt_mode,
                        "model_input_fields": ["attribute", "candidate_value", "evidence_text"],
                    },
                }
                try:
                    result["features"] = extractor.extract(
                        str(row.get("evidence_text") or ""),
                        result["attribute"],
                        result["candidate_value"],
                    )
                except Exception as exc:
                    errors += 1
                    result["status"] = "error"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    result["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
                    if args.fail_fast:
                        raise
                out.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()
                processed.add(claim_id)
                written += 1
                if written % args.checkpoint_every == 0:
                    print(
                        f"[checkpoint] written={written} errors={errors} output={args.output}",
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                del result
    finally:
        extractor.close()

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "written_this_run": written,
        "errors_this_run": errors,
        "processed_total": len(processed),
        "selected_layers": extractor.layer_indices,
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
    error_rate = errors / max(1, written)
    return 0 if error_rate <= args.max_error_rate else 2


if __name__ == "__main__":
    raise SystemExit(main())
