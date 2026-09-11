# Intern-math — R2 budgeted tools candidate

This branch contains an implemented candidate, not a claimed high-scoring submission. No official evaluation is launched by its workflows.

## Deployment

The official entry is `user_agent.py:ReasoningAgent`. Select **intern-s2-preview-397b** in the competition UI. Thinking-off is fixed in code, not inherited from local environment settings. Primary/follow-up/closure budgets are 4096/4096/2048 tokens, at most three calls. Simple completed answers return immediately; numerical/symbolic computations may use two bounded Python workers. Proof arguments are preserved.

`math_tools.py` restricts generated code, sanitizes the worker environment and bounds time/memory/output. It is defense in depth, not a replacement for the host OS sandbox.

## Validation

`r2-paired-eval` compares R1 thinking-off against R2 on 34 SHA-selected questions covering all 17 Benchmark-v1 domains. The dataset and full denominator are fixed before inference. This is a previously used regression set, not an unseen holdout.

Strict checks do not accept substring matches. Remaining cases receive blinded, order-swapped judgments by the same model; agreement is a proxy, not proof of correctness. Unresolved cases remain in the denominator. A successful workflow is not evidence of a good score.

`r2-preflight` runs the full unit suite and creates a minimal candidate package. Historical R1 tests target the unchanged `r1_agent.py` snapshot. See `docs/R2_DESIGN.md` for the design and promotion rules.

Original main, R1 and submission branches remain untouched. Do not spend an official submission until the paired report has been reviewed.
