# Public dataset pipeline

The repository treats the **dataset as the first deliverable**. The goal is to
create a proxy benchmark that matches the competition I/O contract before
optimizing the agent.

## 1. Competition-shaped contract

The local runner consumes JSONL rows with exactly:

```json
{"idx": 0, "problem": "the complete math problem"}
```

The system returns a non-empty `final_response` for each problem. Gold answers
must never be placed in the agent input. For local evaluation they are stored
separately in `gold.jsonl`.

`scripts/export_competition_bundle.py` enforces this separation.

## 2. Normalized candidate schema

Each public item is normalized to:

```json
{
  "idx": 0,
  "problem": "...",
  "answer": "...",
  "answer_type": "numeric",
  "domain": "pde",
  "difficulty": "hard",
  "source_dataset": "deepmath_gap",
  "source_id": "...",
  "problem_hash": "...",
  "review_status": "pending",
  "source_meta": {}
}
```

Imported rows remain `review_status=pending`: a public gold label does not prove
that our domain mapping, self-containedness, answer type, or formatting is
appropriate for this competition.

## 3. Source policy

We use two source tiers. **Primary proxy sources** are comparatively clean and
are allowed to contribute broadly: TheoremQA (text-only Math rows),
ProofNet-Verified, SciBench `calculus`/`diff`, SuperGPQA's `Science ->
Mathematics` hierarchy, and ORQA for expert-authored operations-research
reasoning. **Gap-filling sources** are used only when primary coverage is
insufficient. `DeepMath-Magistral-stage1` is currently restricted to topic paths
for PDE, differential geometry, measure theory, functional analysis, and
regression; every such row requires human review before entering Benchmark-v1.

ProofNet rows remain `answer_type=proof` and are scored separately from
objective-answer QA. TheoremQA rows with a non-null `Picture` are excluded so
the local benchmark remains text-only. ORQA concatenates its context, question,
and options into one `problem` string and converts its zero-based target index
to an answer letter.

Source URLs, selection rules, and license notes are in
`data/source_manifest.json`.

## 4. Build the public candidate pool

```bash
pip install -r requirements.txt
pip install -r requirements-data.txt
python scripts/prepare_public_data.py --output-dir data/public_candidates
```

This writes one normalized JSONL per source, a merged deduplicated
`all_public_candidates.jsonl`, and `coverage.json`.

Fast network/schema check:

```bash
python scripts/prepare_public_data.py --smoke --output-dir tmp/public_smoke
```

## 5. Build a human review queue

Do not randomly sample the merged pool. Create a source-diverse queue:

```bash
python scripts/make_review_queue.py \
  data/public_candidates/all_public_candidates.jsonl \
  --per-domain 30 \
  --output data/review_queue_v1.jsonl \
  --csv-output data/review_queue_v1.csv
```

For each selected item verify at least:

- problem is complete and self-contained;
- gold answer is correct;
- `answer_type` is correct;
- domain mapping is correct;
- no image or inaccessible external context is required;
- no answer is leaked in the problem statement;
- for aggregated/gap-filling sources, the topic label genuinely matches the
  mathematics being tested.

Approved rows are changed to `review_status=approved`.

## 6. Build and export Benchmark-v1

Once each target domain has enough approved rows:

```bash
python scripts/build_balanced_benchmark.py \
  data/review_queue_v1.jsonl \
  --output data/benchmark_v1_full.jsonl \
  --per-domain 25 \
  --min-per-domain 20

python scripts/export_competition_bundle.py \
  data/benchmark_v1_full.jsonl \
  --output-dir data/benchmark_v1
```

The final bundle is:

```text
data/benchmark_v1/
  input.jsonl      # only idx + problem; feed this to main.py
  gold.jsonl       # local evaluator only; never feed to the agent
  manifest.json    # counts + SHA256 hashes
```

For exploratory engineering before review is complete, the builders expose
explicit `--allow-*` flags. Results produced with those flags should not be
reported as Benchmark-v1 results.

## 7. Evaluation split

Proof problems should not be mixed into one exact-match accuracy number. Report
at least:

- automatically scorable QA accuracy (numeric/symbolic/choice/etc.);
- proof subset score (manual or separately validated proof judge);
- per-domain accuracy;
- source-wise accuracy to detect dataset-specific prompt overfitting.
