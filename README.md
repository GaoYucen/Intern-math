# Intern-math

A baseline-first codebase for the Intern mathematical reasoning challenge.

The current priority is **official-evaluation robustness before agent complexity**: freeze a trustworthy competition-shaped proxy benchmark, restore a reproducible single-call Intern-S2 baseline, diagnose delivery/truncation failures, and only then add adaptive compute, verification, tools, or selective rescue when they demonstrate controlled gains.

## Current project stage: R1 Robust Baseline

The active research branch is `r1-robust-baseline`.

R1 is intentionally restricted to exactly one solver request per problem:

- model: `intern-s2-preview-397b` in the R1 evaluation workflows;
- thinking mode: on;
- temperature: `0.0`;
- maximum completion budget: `8192` tokens;
- Agent layer: exactly one `client.chat()` call;
- Client default: `retry=1`, i.e. one HTTP attempt;
- no second-pass finalizer;
- no verifier / judge in the participant path;
- no self-refine or multi-agent path;
- the complete non-empty primary response is preserved as `final_response`;
- the prompt asks the solver to terminate with `FINAL_ANSWER: ...` and to stop exploring after obtaining a well-justified answer.

This design follows the official-evaluation diagnosis: the previous H4 Hybrid used 204 requests for 112 problems, produced 42 invalid answers and scored 10.71%, whereas an earlier 397B single-call submission used 112 requests, had zero invalid answers and scored 13.39%. R1 therefore treats delivery stability and compute allocation as the first bottleneck to fix.

## Frozen proxy benchmark

`Benchmark-v1` is frozen and contains:

- 340 problems total;
- 17 working domains, exactly 20 problems per domain;
- competition-shaped model input containing only `idx` and `problem`;
- gold answers and source metadata stored separately for local evaluation;
- frozen input / gold hashes recorded in `data/benchmark_v1/manifest.json`.

```text
data/benchmark_v1/
  input.jsonl                 # feed this to main.py
  gold.jsonl                  # local evaluator only; never feed to the agent
  manifest.json               # frozen counts + SHA256 hashes
  auto_review_report.json     # automated review summary
  approved_pool_audit.json    # audited candidate coverage
  source_coverage.json        # source-pool coverage before balancing
```

Internal benchmark scores are diagnostic proxies, not estimates of the official hidden-set score. The main acceptance signals for R1 are API success, one-request invariants, final-answer delivery, truncation/closure behavior, and regression relative to the same frozen benchmark.

## Local R1 run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export INTERN_API_KEY=YOUR_TOKEN
export INTERN_MODEL=intern-s2-preview-397b
export INTERN_THINKING_MODE=1
export AGENT_MAX_TOKENS=8192
export AGENT_TEMPERATURE=0.0
export LOCAL_MAX_CONCURRENCY=2

python main.py \
  --input_file data/benchmark_v1/input.jsonl \
  --output_dir outputs/r1

python scripts/evaluate_outputs.py \
  --benchmark data/benchmark_v1/gold.jsonl \
  --output_dir outputs/r1 \
  --report_dir reports/r1
```

## R1 acceptance workflows

The R1 branch uses a gated evaluation chain:

1. `r1-robust-smoke` — 17-domain objective smoke test. It validates API success, unit tests, explicit final-answer behavior, non-empty responses, and the exactly-one-request trace invariant.
2. `r1-long-reasoning-stress` — 40-item trusted long-reasoning stress set with fixed 397B / thinking-on / 8K / temperature-0 configuration. It is explicitly triggered through `experiments/r1_long_stress.trigger`.
3. `r1-full-benchmark` — complete 340-item Benchmark-v1 run. It is intentionally gated behind `experiments/r1_full_benchmark.trigger` and should only be launched after the long-reasoning stress result is accepted.

For Actions-based runs, configure the repository secret:

```text
INTERN_API_KEY
```

Do not commit API keys to the repository.

## Experimental roadmap

The project follows four stages:

- **R1 — Robust Baseline:** single call, answer preservation, deterministic delivery diagnostics, invalid/truncation autopsy.
- **R2 — Adaptive Reasoning:** difficulty/risk profiling and controlled 2K/4K/8K compute allocation with active early closure.
- **R3 — Verified Rescue:** lightweight checking, selective fresh rescue, and targeted SymPy/Python/Z3 routing only for high-risk problems.
- **R4 — Test-Time Search:** adaptive sampling, candidate normalization/consensus, specialist solvers, and hidden-distribution calibration for the 50→70 regime.

Do not reintroduce default finalizers, solver+judge loops, or generic multi-agent orchestration before R1 has passed its acceptance chain.

Dataset construction details are in `docs/DATASET_PIPELINE.md`.