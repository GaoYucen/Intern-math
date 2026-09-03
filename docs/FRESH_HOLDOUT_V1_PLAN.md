# Fresh-Holdout-v1 plan

Purpose: build a frozen, low-contamination validation set that is used to decide whether a competition agent is worth an official submission. It must not be optimized item-by-item after results are observed.

## Why this exists

The existing public/proxy sets are useful for regression and mechanism diagnosis, but they are badly calibrated to the official hidden set. A configuration that looked excellent on the 40-item stress set scored only 13.39% officially. Therefore the old stress score is no longer a submission gate.

## Three-level evaluation policy

1. `benchmark_v1_full` — regression only. Detect engineering or broad capability regressions.
2. `long_reasoning_stress` — mechanism diagnosis only. Measure truncation, long-reasoning behavior, and proof/symbolic failure modes.
3. `Fresh-Holdout-v1` — primary local submission gate. Freeze after construction; do not tune prompts against individual items.

## Construction rules

- Prefer recently published 2025–2026 graduate qualifying-exam topics as *difficulty/style references*, but do not copy public exam text into the repository.
- Write independent, newly parameterized problems with deterministic or independently checkable gold answers.
- Favor exact, symbolic, numeric, Boolean, and finite-structure questions that can be scored without an LLM judge.
- Keep a smaller proof/derivation slice with concise reference rubrics; proof items must not dominate the gate score.
- Include algebra, analysis, complex analysis, topology/geometry, probability, PDE/ODE, numerical analysis, combinatorics, and linear algebra/optimization.
- Record provenance as `synthetic_fresh_v1`; recent public exams are only topic inspiration, not copied source text.
- Freeze the item list and gold answers before running candidate agents.

## Initial target

Phase A: 30–40 items, at least 70% automatically scorable.

Phase B: expand to 60–80 items after the first architecture comparison, without changing Phase-A items.

## Submission gate

A candidate should not be promoted because of a high score on the old 40-item stress set alone. Promotion requires:

- no material regression on `benchmark_v1_full`;
- acceptable completion/error/latency behavior;
- a clear paired improvement on the frozen Fresh-Holdout-v1;
- manual inspection of changed proof/symbolic cases.

## Current architecture lesson

The bounded dual-solver experiment showed a useful confidence signal: on the old stress set, all 8 cases where two independent short solvers produced the same normalized final answer were judged correct, while disagreement cases were much weaker. This suggests using short-solver agreement as a confidence gate, not using a short chooser as the main solver. The next architecture should preserve high-confidence agreement and route disagreement to stronger reasoning/tool paths.
