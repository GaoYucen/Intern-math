# Fresh-Holdout-v1

This directory separates three different roles that were previously conflated:

1. **Scored fresh holdout** — 72 newly written deterministic problems, exactly 4 per subject across 18 subject areas. This is the primary local architecture-comparison set. Generate it with `python scripts/build_fresh_holdout_v1.py` and then freeze the SHA256 in `scored_summary.json`.
2. **Public speculative / challenge-associated sets** — fetched at runtime with `python scripts/collect_public_speculative_sets.py`. They are useful for robustness checks, but they are *not* treated as official evaluation questions and are not committed into this repository.
3. **Regression sets** — the existing 340-item benchmark and stress40 remain useful for catching regressions and studying long-reasoning behavior, but should no longer be used as official-score predictors.

## Source hygiene

`source_manifest.json` records included public sources and exclusions. In particular, assets from a public repository whose recent commit history claims exact one-to-one correspondence with the 112 evaluation questions and direct standard-answer lookup are intentionally excluded from collection. The goal is to approximate difficulty and failure modes without contaminating the submission with hidden-evaluation fingerprints or answer tables.

## Freeze rule

Once `scored.jsonl` is generated and its hash is recorded, do not edit individual failed problems to improve a model's apparent score. Any substantive change creates a new dataset version (v2, v3, ...). This makes comparisons between solver architectures meaningful.

## Recommended use

- Use `scored.jsonl` as the main local gate for architecture comparisons.
- Use collected public sets as an **unscored or separately scored external robustness suite**.
- Report success/error, explicit-final rate, API timeouts, and token/latency behavior alongside accuracy.
- Do not infer an official leaderboard score directly from any local benchmark.
