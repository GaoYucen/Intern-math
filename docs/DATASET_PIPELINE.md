# Public dataset pipeline

The repository treats the **dataset as the first deliverable**. The current
frozen `Benchmark-v1` is a proxy benchmark that matches the competition I/O
contract before optimizing the agent. The competition material says there are
18 subfields but does not enumerate them, so our 17-domain taxonomy remains a
working proxy taxonomy rather than an official reconstruction of the hidden
test distribution.

## 1. Competition-shaped contract

The local runner consumes JSONL rows with exactly:

```json
{"idx": 0, "problem": "the complete math problem"}
```

The system returns a non-empty `final_response` for each problem. Gold answers
are never placed in the agent input; local evaluation uses a separate
`gold.jsonl`. `scripts/export_competition_bundle.py` enforces this separation.

## 2. Source set

The candidate pool combines TheoremQA, ProofNet-Verified, SciBench, SuperGPQA,
ORQA, MA-ProofBench, HARDMath2, a targeted MMLU regression subset, and a
strictly filtered DeepMath gap subset. The main gap-filling additions are:

- MA-ProofBench for functional analysis and measure/integration;
- HARDMath2 nonlinear-PDE problems for PDE;
- MMLU econometrics/statistics questions with explicit regression concepts;
- DeepMath only when the problem text itself semantically confirms the mapped
  PDE or differential-geometry topic.

Source URLs, licenses, and selection rules are recorded in
`data/source_manifest.json`.

## 3. Automatic review

Because Benchmark-v1 is intended to be immediately runnable, candidates are
reviewed deterministically by `scripts/auto_review_dataset.py` instead of a
manual CSV queue. The review checks:

- non-empty, reasonably sized, text-contained problem and gold answer;
- valid domain/hash fields and consistent multiple-choice labels;
- rejection of explicit figure/image dependencies;
- source-specific integrity rules for expert/verified sources;
- explicit regression semantics for the MMLU regression subset;
- problem-text semantic confirmation for the noisier DeepMath gap source.

The frozen build reviewed 4,656 normalized candidates, approved 4,374 and
rejected 282. The approved pool has at least 20 rows in every working target
domain; detailed counts are stored in
`data/benchmark_v1/auto_review_report.json` and
`data/benchmark_v1/approved_pool_audit.json`.

This is an automated source/schema/semantic review, not a claim that every
public gold solution was independently re-solved from scratch. Expert-reviewed
or peer-verified sources are preferred for previously under-covered domains,
and noisy gap-source rows are filtered conservatively.

## 4. Frozen Benchmark-v1

`Benchmark-v1` uses a deterministic seed and selects 20 approved rows from each
of the 17 working proxy domains, for **340 problems total**. The frozen files are
committed directly to `main`:

```text
data/benchmark_v1_full.jsonl       # full rows with gold/metadata

data/benchmark_v1/
  input.jsonl                      # only idx + problem; feed to main.py
  gold.jsonl                       # evaluator only; never feed to the agent
  manifest.json                    # counts + SHA256 hashes
  auto_review_report.json          # review decisions and rejection counts
  approved_pool_audit.json         # approved coverage by domain/source/type
  source_coverage.json             # normalized candidate-pool coverage
```

The current manifest contains 20 rows for every working proxy domain and records
SHA256 hashes for both the model input and gold file. Do not modify the frozen
input while comparing model variants; create a new benchmark version instead.

## 5. Rebuild command

The GitHub Actions workflow `.github/workflows/build-public-candidates.yml`
rebuilds the candidate pool, performs automatic review, audits coverage, freezes
the balanced dataset, validates the competition input contract, and commits the
result to `main` when the build succeeds.

For a local rebuild, the equivalent core commands are:

```bash
python scripts/prepare_public_data.py --output-dir tmp/public_candidates_full
python scripts/auto_review_dataset.py \
  tmp/public_candidates_full/all_public_candidates.jsonl \
  --output tmp/reviewed_candidates.jsonl \
  --approved-output tmp/approved_candidates.jsonl \
  --report tmp/auto_review_report.json
python scripts/build_balanced_benchmark.py \
  tmp/approved_candidates.jsonl \
  --output data/benchmark_v1_full.jsonl \
  --per-domain 20 --min-per-domain 20 --seed 20260829
python scripts/export_competition_bundle.py \
  data/benchmark_v1_full.jsonl --output-dir data/benchmark_v1
```

## 6. Evaluation split

The 340 problems include objective-answer items and proof/open-ended items.
Proof items must not be mixed into a naive exact-match score. Report at least:

- automatically scorable QA accuracy (choice/numeric/integer/float/etc.);
- proof/open-ended score using an appropriate judge or verification rule;
- per-domain accuracy;
- source-wise accuracy to detect dataset-specific prompt overfitting.
