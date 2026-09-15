#!/usr/bin/env python3
"""Build label-clean SciFact inputs for the ECLT public-transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "icassp-eclt-scifact-input-v1"
LABELS = {"SUPPORT": 1, "CONTRADICT": 0}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield row


def stable_shard(value: str, shards: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shards


def render_abstract(document: dict[str, Any]) -> str:
    title = str(document.get("title") or "").strip()
    abstract = [str(item).strip() for item in document.get("abstract") or [] if str(item).strip()]
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append("Abstract:\n" + "\n".join(f"[{idx}] {sentence}" for idx, sentence in enumerate(abstract)))
    return "\n".join(parts)


def build_split(
    path: Path,
    split: str,
    corpus: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for claim_row in iter_jsonl(path):
        claim_source_id = int(claim_row["id"])
        claim = str(claim_row.get("claim") or "").strip()
        evidence = claim_row.get("evidence") or {}
        if not claim or not evidence:
            continue
        for raw_doc_id, annotations in evidence.items():
            doc_id = int(raw_doc_id)
            document = corpus.get(doc_id)
            if document is None:
                raise ValueError(f"missing SciFact document {doc_id} for claim {claim_source_id}")
            labels = {str(item.get("label") or "") for item in annotations}
            unknown = labels - set(LABELS)
            if unknown or len(labels) != 1:
                raise ValueError(
                    f"ambiguous labels claim={claim_source_id} doc={doc_id}: {sorted(labels)}"
                )
            label_name = next(iter(labels))
            supported = LABELS[label_name]
            evidence_sets = sorted(
                {
                    tuple(int(index) for index in item.get("sentences") or [])
                    for item in annotations
                }
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "claim_id": f"scifact::{split}::{claim_source_id}::{doc_id}",
                    # Group every claim about the same abstract together.
                    "record_id": f"scifact-doc::{doc_id}",
                    "partition": split,
                    "attribute": "scientific_claim",
                    "candidate_value": claim,
                    "evidence_text": render_abstract(document),
                    "labels": {"supported": supported, "exact_accept": supported},
                    "audit_metadata": {
                        "dataset": "SciFact",
                        "official_split": split,
                        "source_claim_id": claim_source_id,
                        "source_doc_id": doc_id,
                        "label": label_name,
                        "evidence_sentence_sets": [list(indices) for indices in evidence_sets],
                        "full_abstract_input": True,
                        "label_or_rationale_in_model_input": False,
                        "source": "https://github.com/allenai/scifact",
                    },
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/scifact"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inputs/scifact"),
    )
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    corpus = {int(row["doc_id"]): row for row in iter_jsonl(args.data_dir / "corpus.jsonl")}
    train = build_split(args.data_dir / "claims_train.jsonl", "train", corpus)
    dev = build_split(args.data_dir / "claims_dev.jsonl", "dev", corpus)
    train_records = {row["record_id"] for row in train}
    for row in dev:
        row["audit_metadata"]["record_disjoint_from_train"] = row["record_id"] not in train_records

    rows = train + dev
    claim_ids = [row["claim_id"] for row in rows]
    if len(claim_ids) != len(set(claim_ids)):
        raise AssertionError("duplicate SciFact claim-document IDs")
    if any(not row["evidence_text"] for row in rows):
        raise AssertionError("empty SciFact evidence text")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (args.output_dir / f"scifact_claim_verification_shard{idx}.jsonl").open(
            "w", encoding="utf-8"
        )
        for idx in range(args.shards)
    ]
    shard_counts = Counter()
    try:
        for row in rows:
            shard = stable_shard(row["claim_id"], args.shards)
            handles[shard].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            shard_counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()

    counts = Counter(
        f"{row['partition']}::{(row.get('audit_metadata') or {}).get('label')}" for row in rows
    )
    disjoint_counts = Counter(
        str((row.get("audit_metadata") or {}).get("label"))
        for row in dev
        if (row.get("audit_metadata") or {}).get("record_disjoint_from_train")
    )
    manifest = {
        "schema_version": "icassp-eclt-scifact-manifest-v1",
        "source_data_dir": str(args.data_dir),
        "rows": len(rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_records": len(train_records),
        "dev_records": len({row["record_id"] for row in dev}),
        "dev_record_disjoint_rows": sum(
            bool((row.get("audit_metadata") or {}).get("record_disjoint_from_train"))
            for row in dev
        ),
        "dev_record_overlap_rows": sum(
            not bool((row.get("audit_metadata") or {}).get("record_disjoint_from_train"))
            for row in dev
        ),
        "label_counts": dict(sorted(counts.items())),
        "dev_record_disjoint_label_counts": dict(sorted(disjoint_counts.items())),
        "shard_counts": {str(idx): shard_counts[idx] for idx in range(args.shards)},
        "candidate_exact_occurrence_rows": sum(
            row["candidate_value"].lower() in row["evidence_text"].lower() for row in rows
        ),
        "evidence_chars": {
            "min": min(len(row["evidence_text"]) for row in rows),
            "max": max(len(row["evidence_text"]) for row in rows),
            "mean": round(sum(len(row["evidence_text"]) for row in rows) / len(rows), 2),
        },
        "model_input_fields": ["candidate_value", "evidence_text"],
        "label_fields_excluded_from_model_input": ["labels", "audit_metadata"],
        "evidence_policy": "complete title and abstract; rationale indices retained for audit only",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
