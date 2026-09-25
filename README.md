# ECLT

Code for Evidence-Conditioned Layer Trajectories: verifying a fixed claim
against supplied evidence using relations between pooled hidden states across
language model layers. The backbone stays frozen; a regularized linear probe
is trained on labeled claim-evidence pairs.

This is a code-only release. It contains public-data preparation, ECLT feature
extraction, probe evaluation, depth ablations, and an optional ICR baseline.
Datasets, model weights, cached predictions, papers, and experiment logs are
not included. The private network-observation audit and the separate HIDE
runner are outside this release.

## Setup

Use Python 3.10 or newer. Install PyTorch for your CUDA environment first
(the GPU experiments used PyTorch 2.7.1), then install the Python dependencies:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The scripts use Transformers 4.55.0 and scikit-learn 1.7.2. Keep these versions
when comparing scores. Feature extraction needs local or Hugging Face model
weights and access to hidden states. Use trusted weights and model code.
CPU evaluation does not require a GPU; extracting features from a 7B model
is intended for a GPU with sufficient memory.

## Prepare public data

Obtain [SciFact](https://github.com/allenai/scifact) and
[VitaminC](https://github.com/TalSchuster/VitaminC) from their original
repositories, following their respective licenses. Place SciFact's
`corpus.jsonl`, `claims_train.jsonl`, and `claims_dev.jsonl` under
`data/scifact/`, and VitaminC's `train.jsonl` and `test.jsonl` under
`data/vitaminc/`.

```bash
python scripts/prepare_scifact.py --shards 1
python scripts/prepare_vitaminc.py --shards 1
```

SciFact preparation retains labeled claim-abstract pairs and document IDs.
VitaminC preparation selects 1,500 training pairs and 500 test pairs, keeping
the claim fixed within each supporting/contrast pair and checking split
overlap. Seeds and selection rules are retained in the source.

## Extract ECLT features

For the matched Qwen2.5 configuration, use eager attention and the
claim-verification prompt explicitly:

```bash
python scripts/extract_features.py \
  --input inputs/scifact/scifact_claim_verification_shard0.jsonl \
  --output outputs/scifact_features.jsonl \
  --model-path ./models/Qwen2.5-7B-Instruct \
  --prompt-mode claim_verification --attn-implementation eager \
  --max-input-tokens 8192 --max-context-chars 24000 --fail-fast

python scripts/extract_features.py \
  --input inputs/vitaminc/vitaminc_contrastive_full.jsonl \
  --output outputs/vitaminc_features.jsonl \
  --model-path ./models/Qwen2.5-7B-Instruct \
  --prompt-mode claim_verification --attn-implementation eager \
  --max-input-tokens 2048 --max-context-chars 12000 --fail-fast
```

Choose the GPU with `CUDA_VISIBLE_DEVICES`. Each process uses one visible GPU;
the extractor performs one teacher-forced pass per input row. Nine sampled
depths include the first and last decoder layers. Random projections and
top-neuron diagnostics may appear in raw feature files but are not consumed
by the released paper evaluators.

For your own JSONL input, each row needs `claim_id`, `record_id`, `partition`,
`attribute`, `candidate_value`, and `evidence_text`. Supervised evaluation also
needs `labels.supported` (0 or 1). The preparers supply the additional metadata
required by the public-task evaluators. Labels and audit metadata are not fed
into the language model.

Resume uses the IDs already present in the output file. Start with a fresh
output path after changing inputs, model weights, or settings.

## Evaluate

```bash
python scripts/evaluate_scifact.py \
  --features outputs/scifact_features.jsonl \
  --output-json outputs/scifact_scores.json \
  --output-md outputs/scifact_scores.md --bootstrap 2000

python scripts/evaluate_pairs.py \
  --eclt-features outputs/vitaminc_features.jsonl \
  --output-json outputs/vitaminc_scores.json \
  --output-md outputs/vitaminc_scores.md --bootstrap 2000

python scripts/evaluate_depth.py \
  --features Qwen2.5=outputs/scifact_features.jsonl \
  --output-json outputs/depth_scores.json \
  --output-md outputs/depth_scores.md --bootstrap 2000
```

The depth evaluator also reports `raw_9`, which keeps the nine relation values
at fixed depth coordinates without summary statistics, and `raw_diff_9`, which
adds differences between adjacent sampled depths. Their grouped-bootstrap
comparison tests whether local trajectory changes add information beyond depth
identity alone.

SciFact uses document-grouped five-fold evaluation and a document-disjoint
train-to-dev comparison. VitaminC uses the selected official train/test split;
its additional pooled OOF output is a separate diagnostic. Paired direction
counts how often the supporting context scores strictly higher than the
contrast context; ties do not count as correct.

Feature names in the code map to the manuscript as follows:

| Code | Representation |
|---|---|
| `confidence` | Input indicators and token confidence |
| `last_layer` / `final_relation` | Final-layer relations and controls |
| `relation_trajectory` | Per-depth relations, summaries, and controls |
| `trajectory` / `full_trajectory` | Full ECLT, including per-depth state norms |
| `pure_final_relation` | Final-layer relations only |
| `pure_relation_trajectory` | Relational trajectory only |

## Optional ICR comparison

`extract_icr_features.py` implements the fixed-claim adaptation of
[ICR Probe](https://github.com/XavierZhang2002/ICR_Probe), following upstream
commit `40ec490e762cadbac6bcefdc24a8f0d5974e8448` (Apache-2.0).
It preserves the ICR scoring operations and changes the input procedure to
teacher forcing for the supplied claim. It does not reproduce the upstream
autoregressive QA task. `icr_probe.py` supplies the linear and MLP evaluators.

```bash
python scripts/extract_icr_features.py \
  --input inputs/vitaminc/vitaminc_contrastive_full.jsonl \
  --output outputs/vitaminc_icr.jsonl \
  --model-path ./models/Qwen2.5-7B-Instruct \
  --max-input-tokens 2048 --max-context-chars 12000 --fail-fast

python scripts/evaluate_pairs.py \
  --eclt-features outputs/vitaminc_features.jsonl \
  --icr-features outputs/vitaminc_icr.jsonl \
  --output-json outputs/vitaminc_comparison.json \
  --output-md outputs/vitaminc_comparison.md --bootstrap 2000
```

## License

Code is released under Apache-2.0. The ICR adaptation retains its upstream
attribution above and in the source. Public datasets and model weights remain
subject to their original terms and are not redistributed here.
