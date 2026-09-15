#!/usr/bin/env python3
"""Build a frozen, case-disjoint VitaminC claim/evidence contrast set.

Each pair keeps the claim fixed and changes only the official evidence. One row
is SUPPORTS and the other is either REFUTES or NOT ENOUGH INFO. Only real
Wikipedia revisions are used, and at most one pair is retained per case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "icassp-eclt-vitaminc-input-v1"
SOURCE_URL = "https://github.com/TalSchuster/VitaminC"
NEGATIVE_LABELS = ("REFUTES", "NOT ENOUGH INFO")


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


def stable_id(*values: Any) -> str:
    payload = "\x1f".join(clean(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def text_digest(value: Any) -> str:
    return hashlib.sha256(clean(value).casefold().encode("utf-8")).hexdigest()


def source_row_id(row: dict[str, Any]) -> str:
    return clean(row.get("unique_id")) or stable_id(
        row.get("case_id"), row.get("claim"), row.get("evidence"), row.get("label")
    )


def load_candidates(path: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in iter_jsonl(path):
        if clean(row.get("revision_type")).lower() != "real":
            continue
        case_id = clean(row.get("case_id"))
        claim = clean(row.get("claim"))
        label = clean(row.get("label")).upper()
        evidence = clean(row.get("evidence"))
        if not case_id or not claim or not evidence or label not in {"SUPPORTS", *NEGATIVE_LABELS}:
            continue
        grouped[(case_id, claim.casefold())][label].append(row)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (case_id, _), labels in grouped.items():
        if not labels["SUPPORTS"]:
            continue
        negative_label = next((label for label in NEGATIVE_LABELS if labels[label]), None)
        if negative_label is None:
            continue
        supports = sorted(labels["SUPPORTS"], key=source_row_id)
        negatives = sorted(labels[negative_label], key=source_row_id)
        group_seed = int(stable_id(seed, case_id, supports[0].get("claim")), 16)
        rng = random.Random(group_seed)
        by_case[case_id].append(
            {
                "case_id": case_id,
                "claim": clean(supports[0].get("claim")),
                "negative_label": negative_label,
                "support": supports[rng.randrange(len(supports))],
                "negative": negatives[rng.randrange(len(negatives))],
            }
        )

    # At most one pair per case. The fixed seed makes this choice reproducible.
    selected_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_id, candidates in sorted(by_case.items()):
        rng = random.Random(int(stable_id(seed, "case", case_id), 16))
        chosen = candidates[rng.randrange(len(candidates))]
        selected_by_label[chosen["negative_label"]].append(chosen)
    return selected_by_label


def select_stratified(
    pools: dict[str, list[dict[str, Any]]],
    *,
    total_pairs: int,
    refutes_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    refutes_n = round(total_pairs * refutes_fraction)
    targets = {"REFUTES": refutes_n, "NOT ENOUGH INFO": total_pairs - refutes_n}
    chosen: list[dict[str, Any]] = []
    for label, target in targets.items():
        pool = list(pools[label])
        if len(pool) < target:
            raise ValueError(f"insufficient {label} pairs: need={target} available={len(pool)}")
        random.Random(int(stable_id(seed, label), 16)).shuffle(pool)
        chosen.extend(pool[:target])
    random.Random(int(stable_id(seed, "pair-order"), 16)).shuffle(chosen)
    return chosen


def materialize_pair(pair: dict[str, Any], split: str, pair_index: int) -> list[dict[str, Any]]:
    pair_id = f"vitaminc::{split}::{pair_index:05d}::{stable_id(pair['case_id'], pair['claim'])}"
    record_id = f"vitaminc-case::{split}::{pair['case_id']}"
    rows = []
    for role, label, source in (
        ("support", "SUPPORTS", pair["support"]),
        ("contrast", pair["negative_label"], pair["negative"]),
    ):
        evidence = clean(source.get("evidence"))
        page = clean(source.get("page"))
        evidence_text = f"Page: {page}\nEvidence:\n{evidence}" if page else evidence
        supported = int(label == "SUPPORTS")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "claim_id": f"{pair_id}::{role}",
                "record_id": record_id,
                "partition": split,
                "attribute": "factual_claim",
                "candidate_value": pair["claim"],
                "evidence_text": evidence_text,
                "labels": {"supported": supported, "exact_accept": supported},
                "audit_metadata": {
                    "dataset": "VitaminC",
                    "official_split": split,
                    "pair_id": pair_id,
                    "pair_role": role,
                    "case_id": pair["case_id"],
                    "unique_id": source_row_id(source),
                    "wiki_revision_id": clean(source.get("wiki_revision_id")),
                    "page": page,
                    "revision_type": clean(source.get("revision_type")),
                    "label": label,
                    "claim_fixed_within_pair": True,
                    "real_revision_only": True,
                    "label_or_rationale_in_model_input": False,
                    "source": SOURCE_URL,
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/vitaminc"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inputs/vitaminc"),
    )
    parser.add_argument("--train-pairs", type=int, default=1500)
    parser.add_argument("--test-pairs", type=int, default=500)
    parser.add_argument("--refutes-fraction", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "selection": {
            "real_revision_only": True,
            "one_pair_per_case": True,
            "fixed_claim_within_pair": True,
            "refutes_fraction": args.refutes_fraction,
            "official_dev_unused": True,
        },
        "source": SOURCE_URL,
        "splits": {},
    }
    for offset, (split, pair_count) in enumerate(
        (("train", args.train_pairs), ("test", args.test_pairs))
    ):
        pools = load_candidates(args.raw_dir / f"{split}.jsonl", args.seed + offset)
        pairs = select_stratified(
            pools,
            total_pairs=pair_count,
            refutes_fraction=args.refutes_fraction,
            seed=args.seed + offset,
        )
        rows = [
            row
            for pair_index, pair in enumerate(pairs)
            for row in materialize_pair(pair, split, pair_index)
        ]
        write_jsonl(args.output_dir / f"vitaminc_contrastive_{split}.jsonl", rows)
        all_rows.extend(rows)
        manifest["splits"][split] = {
            "pairs": len(pairs),
            "rows": len(rows),
            "cases": len({row["record_id"] for row in rows}),
            "labels": dict(Counter(row["audit_metadata"]["label"] for row in rows)),
        }

    train_cases = {
        row["record_id"].split("::", 2)[-1] for row in all_rows if row["partition"] == "train"
    }
    test_cases = {
        row["record_id"].split("::", 2)[-1] for row in all_rows if row["partition"] == "test"
    }
    overlap = train_cases & test_cases
    if overlap:
        raise ValueError(f"official train/test case overlap detected: {len(overlap)}")

    write_jsonl(args.output_dir / "vitaminc_contrastive_full.jsonl", all_rows)
    shard_rows = [[] for _ in range(args.shards)]
    # Keep both rows from a pair on the same shard.
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        pair_groups[row["audit_metadata"]["pair_id"]].append(row)
    for index, pair_id in enumerate(sorted(pair_groups)):
        shard_rows[index % args.shards].extend(pair_groups[pair_id])
    for shard, rows in enumerate(shard_rows):
        write_jsonl(args.output_dir / f"vitaminc_contrastive_shard{shard}.jsonl", rows)

    split_rows = {
        split: [row for row in all_rows if row["partition"] == split]
        for split in ("train", "test")
    }
    overlap_audit = {}
    for name, getter in (
        ("claim", lambda row: row["candidate_value"]),
        ("evidence", lambda row: row["evidence_text"]),
        ("page", lambda row: row["audit_metadata"]["page"]),
        ("unique_id", lambda row: row["audit_metadata"]["unique_id"]),
    ):
        train_values = {text_digest(getter(row)) for row in split_rows["train"]}
        test_values = {text_digest(getter(row)) for row in split_rows["test"]}
        overlap_audit[name] = {
            "train_unique": len(train_values),
            "test_unique": len(test_values),
            "overlap": len(train_values & test_values),
        }
    same_page_pairs = sum(
        len({row["audit_metadata"]["page"] for row in rows}) == 1
        for rows in pair_groups.values()
    )

    manifest["total"] = {
        "pairs": len(pair_groups),
        "rows": len(all_rows),
        "train_test_case_overlap": len(overlap),
        "shards": [len(rows) for rows in shard_rows],
        "same_page_pairs": same_page_pairs,
        "overlap_audit": overlap_audit,
        "context_chars": {
            split: {
                "min": min(len(row["evidence_text"]) for row in rows),
                "mean": sum(len(row["evidence_text"]) for row in rows) / len(rows),
                "max": max(len(row["evidence_text"]) for row in rows),
            }
            for split, rows in split_rows.items()
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
