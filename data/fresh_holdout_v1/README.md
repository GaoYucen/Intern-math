# Fresh-Holdout-v1 (v0.1)

A fresh, frozen, gold-reliable benchmark for architecture selection after the old proxy set proved poorly calibrated to the official hidden evaluation.

- 30 items: 22 development + 8 never-touch.
- v0.1 uses objective gold only (integer/rational/symbolic/boolean).
- Problems are independently constructed/parameterized; recent public exams inform topic mix only.
- Gold is deterministically recomputed by `scripts/build_fresh_holdout_v1.py`.
- Freeze rule: after observing model results, never edit items in place.
- Never-touch rule: indices 22-29 are reserved for one-shot pre-submission evaluation.

Use `benchmark_v1` for regression, `long_reasoning_stress` for mechanism diagnostics, indices 0-21 here for architecture selection, and 22-29 only as the final holdout. This set is not claimed to predict the official leaderboard score.
