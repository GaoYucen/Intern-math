# Experiment status — 2026-09-03

## Official calibration facts

Both official submissions used `intern-s2-preview-397b` selected on the submission page.

- Direct, thinking OFF, 4K: 14/112 correct = 12.50%; 77/112 truncated.
- Direct, thinking ON, 8K: 15/112 correct = 13.39%; 85/112 truncated.

Therefore the official evidence does not support simply increasing one-shot reasoning length.

## Old stress40 bounded dual experiment

Run `33654981541`, 397B, two independent thinking-OFF 4K solvers, short 2K chooser on disagreement.

- 40/40 API success; final marker 39/40.
- Agreement gate: 8/40; all 8 agreement cases were reference-judge correct.
- Short chooser was used on 32/40 and was much weaker on the hard cases.
- Final reference adjudication: 20 correct, 13 wrong, 7 invalid.

Conclusion: short independent consensus is a useful confidence gate, but the short chooser is not a sufficient hard-problem solver.

## Fresh-Holdout-v1

A deterministic frozen set of 72 newly written problems was built: 18 subject areas × 4 questions, 36 Chinese and 36 English. It contains 53 exact-track, 16 tool-track, and 3 judge-track items. Frozen SHA-256:

`94e6f919d9ade2d823afb86c81d183f3a408c218ffd5a169eae0cc8ce69eed9f`

This set is for paired architecture comparison. Its absolute accuracy must not be interpreted as predicted official accuracy; many items are still more routine than the hidden competition set.

Calibration run `33657546970`:

### Direct A — 397B, thinking OFF, 4K

- 72/72 success.
- Strict: 35 correct, 11 wrong, 26 review.
- Reference adjudication: 48 correct, 19 wrong, 5 invalid.

### Bounded dual — A/B thinking OFF 4K + short chooser

- 72/72 success, final marker 72/72.
- Strict: 48 correct, 7 wrong, 17 review.
- Reference adjudication: 64 correct, 7 wrong, 1 invalid.
- Agreement gate: 40 cases. Paired artifact analysis found 39/40 agreement cases correct.
- Compared with the independent direct run, dual moved 22 prior wrong/invalid cases to correct while moving 6 prior correct cases to wrong/invalid; net +16 correct (48 → 64). Because model sampling is stochastic, treat this as strong directional evidence rather than an exact causal estimate.

Conclusion: independent short paths add value, and answer agreement is the most reliable inexpensive confidence signal seen so far. Fresh-Holdout is nevertheless too easy to be used as an official-score predictor.

## Active architecture: hybrid-consensus-v1

Branch: `hybrid-consensus-v1`

Workflow head before this status note: `dd153eb434fc48f220bd4e9987f50fc98749d2f3`

Run: `33699514225`

Architecture:

1. Solver A: thinking OFF, 4K, temperature 0.15.
2. Solver B: independent, thinking OFF, 4K, temperature 0.35.
3. If normalized final answers agree, accept immediately.
4. Otherwise escalate to a thinking-ON 4K deep solver using both fast attempts as evidence.
5. Only if the deep pass fails to close with a final marker, use a short thinking-OFF 2K chooser as a delivery fallback.

The workflow runs the frozen Fresh-Holdout-v1 and old stress40 in parallel. The important decision criterion is whether the hybrid recovers much of the long-thinking mathematical quality on stress40 while retaining the completion/latency benefits and strong consensus gate. A high Fresh-Holdout score alone is not enough to promote a submission.
