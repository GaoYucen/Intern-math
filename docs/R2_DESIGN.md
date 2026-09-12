# R2 Budgeted Mathematical Tools

Status: implemented; paired model validation pending. No official evaluation was submitted.

## Fixed deployment contract

- Official client injected into ReasoningAgent; participant uses only chat.
- Select intern-s2-preview-397b on the competition platform. Code cannot override it.
- thinking_mode=False is an explicit code default, independent of environment variables.
- Primary call: 4096 tokens, temperature 0.1. At most three calls, ceilings 4096/4096/2048.
- Mathematical Python: at most two calculations, separate sanitized subprocesses, 12 s wall limit and CPU/memory/output limits.
- Complete answers return immediately. Incomplete/code-only replies get calculation feedback or bounded closure rescue. No unconditional verifier or answer rewriting.
- Proofs retain essential arguments. Metadata other than the problem is unused; no cross-question state and no answer lookup.
- Traces contain counts/status/timing only. Model exception strings and private client fields are not recorded by the participant.

## Why this candidate

R1 thinking-off improved delivery but did not establish accuracy. R2 aims to retain delivery while replacing fragile arithmetic/algebra with executed computation. Domain reminders are inferred from the question, not gold metadata. Thinking-off tools are an empirical hypothesis: provider documentation favors thinking-on for tool-heavy agents, so this design must earn its place in paired testing.

## Validation frozen before inference

Two questions per domain from the existing 340-item Benchmark-v1 by SHA256 of r2-frozen-20260912: + question. This is a 34-item stratified regression, not an unseen holdout or hidden-score estimate. Compare R1 thinking-off (8192 tokens, temperature 0) with deployed R2 defaults. Different compute ceilings are reported, not described as equal budget.

Strict checks do not use substring containment. Remaining symbolic/proof cases receive two blinded, order-swapped judgments by the SAME 397B model. Agreement remains a proxy; disagreement/failed judgments stay unresolved in the full denominator. A theorem specification ending in sorry is NOT a proof reference. Report gains AND regressions, API errors, closure, calls, tokens and latency.

Unit tests cover fixed config, no gold leakage/cross-question state, nested boxes, tool execution, restricted imports, timeout and exception handling. The mathematical worker is defense in depth, not a replacement for the host OS sandbox.

## Promotion

A green workflow alone is not acceptance. Inspect full-denominator results and unresolved cases. Preserve existing submission branches. Package only user_agent.py, r2_agent.py, math_tools.py, requirements and instructions after acceptance; exclude datasets, golds and local judge scripts.
