# Intern-math

A baseline-first codebase for the Intern mathematical reasoning challenge.

The main design choice is deliberate: **measure a strong single-model baseline
before adding multi-agent complexity**. The default `ReasoningAgent` makes one
model call with thinking enabled and returns the full mathematical solution.
An optional two-call self-refinement mode is provided only for controlled
ablation.

## Why this repository starts simple

The previous prototype mixed planner, solver, synthesizer and verifier modules
before a trustworthy benchmark and direct-model baseline were established.
That makes a poor leaderboard result hard to diagnose and can even turn a
correct solver output into a wrong final answer. Here every additional module
must beat the fixed held-out benchmark before it becomes part of the default.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export INTERN_API_KEY=YOUR_TOKEN
export INTERN_MODEL=intern-s2-preview
export INTERN_THINKING_MODE=1
export AGENT_MODE=direct
python main.py --input_file sample_data/dev.jsonl --output_dir outputs/direct
python scripts/evaluate_outputs.py \
  --benchmark sample_data/dev.jsonl \
  --output_dir outputs/direct \
  --report_dir reports/direct
```

### A/B test self-refinement

```bash
export AGENT_MODE=self_refine
python main.py --input_file sample_data/dev.jsonl --output_dir outputs/self_refine
```

## Competition compatibility

The runner follows the official interface: `ReasoningAgent.solve(problem,
metadata)` returns a dictionary containing a non-empty `final_response`. Traces
contain only aggregate metadata and do not copy the hidden problem or raw model
responses.

## Benchmark workflow

See `docs/EXPERIMENT_PLAN.md` and `data/README.md`.

The intended order is:

1. build and audit a balanced proxy benchmark;
2. run direct model baselines;
3. perform domain/failure analysis;
4. add one targeted mechanism at a time;
5. keep only changes that improve the held-out benchmark and then verify on the
   official leaderboard.
