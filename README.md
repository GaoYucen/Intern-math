# Intern-math

A baseline-first codebase for the Intern mathematical reasoning challenge.

The main design choice is deliberate: **freeze a trustworthy competition-shaped proxy benchmark, establish a reproducible single-model baseline, and only then add routing / verification / multi-agent mechanisms that demonstrably improve the fixed benchmark.**

## Current status

`Benchmark-v1` is now frozen on `main`.

- 340 problems total;
- 17 working domains, exactly 20 problems per domain;
- competition-shaped model input contains only `idx` and `problem`;
- gold answers and source metadata are kept separately for local evaluation;
- the frozen input / gold hashes are recorded in `data/benchmark_v1/manifest.json`;
- automatic source/schema/semantic review and coverage audit reports are stored alongside the benchmark.

The active project stage is now **model evaluation**, not dataset construction.

```text
data/benchmark_v1/
  input.jsonl                 # feed this to main.py
  gold.jsonl                  # local evaluator only; never feed to the agent
  manifest.json               # frozen counts + SHA256 hashes
  auto_review_report.json     # automated review summary
  approved_pool_audit.json    # audited candidate coverage
  source_coverage.json        # source-pool coverage before balancing
```

## Agent baseline

The default `ReasoningAgent` is intentionally simple and reproducible:

- `AGENT_MODE=direct`: one reasoning call;
- `AGENT_MODE=self_refine`: direct solution followed by one independent audit/correction call;
- thinking mode is enabled by default;
- every successful response must contain a non-empty `final_response` and the prompt asks the model to end with `FINAL_ANSWER: ...`.

The current default model in `llm_client.py` is `intern-s2-preview`; it can be overridden with `INTERN_MODEL`.

## Local baseline run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export INTERN_API_KEY=YOUR_TOKEN
export INTERN_MODEL=intern-s2-preview
export INTERN_THINKING_MODE=1
export AGENT_MODE=direct
export LOCAL_MAX_CONCURRENCY=3

python main.py \
  --input_file data/benchmark_v1/input.jsonl \
  --output_dir outputs/direct

python scripts/evaluate_outputs.py \
  --benchmark data/benchmark_v1/gold.jsonl \
  --output_dir outputs/direct \
  --report_dir reports/direct
```

The evaluator reports automatically scorable accuracy plus per-domain counts. Proof items remain separated from exact-answer accuracy because they require a proof judge rather than string matching.

## GitHub Actions model runs

Two workflows are prepared:

- `baseline-smoke`: one representative problem from each of the 17 domains, used to validate API connectivity and the end-to-end runner before spending the full evaluation budget;
- `baseline-full`: the complete 340-problem Benchmark-v1 run, with selectable model, agent mode and local concurrency. Outputs and evaluation reports are uploaded as workflow artifacts.

For Actions-based runs, configure the repository secret:

```text
INTERN_API_KEY
```

Do not commit API keys to the repository.

## Experimental order from here

1. run the 17-domain direct smoke test;
2. if the API/run contract is healthy, run the 340-problem direct baseline;
3. inspect overall, per-domain and per-answer-type failures;
4. run the controlled `self_refine` ablation;
5. add one targeted mechanism at a time (for example routing, specialist prompts, candidate generation, verifier/selector);
6. retain only mechanisms that improve the same frozen Benchmark-v1, then verify the final candidate on the official evaluation platform.

Dataset construction details are in `docs/DATASET_PIPELINE.md`.
