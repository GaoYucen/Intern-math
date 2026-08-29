# Intern-math

A baseline-first codebase for the Intern mathematical reasoning challenge.

The main design choice is deliberate: **build a trustworthy competition-shaped
proxy dataset and measure a strong single-model baseline before adding
multi-agent complexity**. The default `ReasoningAgent` makes one model call with
thinking enabled; an optional two-call self-refinement mode exists only for
controlled ablation.

## Current priority: dataset first

The current work is focused on constructing `Benchmark-v1` before further agent
optimization. Public adapters are provided for **TheoremQA, ProofNet-Verified,
SciBench, and the Mathematics subset of SuperGPQA**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-data.txt

python scripts/prepare_public_data.py --output-dir data/public_candidates
python scripts/make_review_queue.py \
  data/public_candidates/all_public_candidates.jsonl \
  --per-domain 30 \
  --output data/review_queue_v1.jsonl \
  --csv-output data/review_queue_v1.csv
```

See `docs/DATASET_PIPELINE.md` for the complete review, balancing, and export
workflow.

## Competition-compatible benchmark bundle

After human review and balancing, export with:

```bash
python scripts/export_competition_bundle.py \
  data/benchmark_v1_full.jsonl \
  --output-dir data/benchmark_v1
```

This produces:

```text
input.jsonl   # exactly {idx, problem}; feed this to main.py
gold.jsonl    # local evaluation only; never exposed to the agent
manifest.json # counts and SHA256 hashes
```

## Run a baseline

```bash
export INTERN_API_KEY=YOUR_TOKEN
export INTERN_MODEL=intern-s2-preview
export INTERN_THINKING_MODE=1
export AGENT_MODE=direct
python main.py --input_file data/benchmark_v1/input.jsonl --output_dir outputs/direct
python scripts/evaluate_outputs.py \
  --benchmark data/benchmark_v1/gold.jsonl \
  --output_dir outputs/direct \
  --report_dir reports/direct
```

### A/B test self-refinement

```bash
export AGENT_MODE=self_refine
python main.py --input_file data/benchmark_v1/input.jsonl --output_dir outputs/self_refine
```

## Why this repository starts simple

The previous prototype mixed planner, solver, synthesizer and verifier modules
before a trustworthy benchmark and direct-model baseline were established.
That makes a poor leaderboard result hard to diagnose and can even turn a
correct solver output into a wrong final answer. Here every additional module
must beat the fixed held-out benchmark before it becomes part of the default.

## Competition compatibility

The runner follows the official interface: `ReasoningAgent.solve(problem,
metadata)` returns a dictionary containing a non-empty `final_response`. Traces
contain only aggregate metadata and do not copy the hidden problem or raw model
responses.

## Intended experimental order

1. import and normalize suitable public mathematical datasets;
2. manually audit and freeze a balanced held-out proxy benchmark;
3. run direct Intern model baselines;
4. perform per-domain and failure-type analysis;
5. add one targeted mechanism at a time;
6. keep only changes that improve the held-out benchmark, then verify on the
   official leaderboard.

Additional notes: `docs/EXPERIMENT_PLAN.md`, `docs/STUDENT_DATA_AUDIT.md`, and
`data/README.md`.
