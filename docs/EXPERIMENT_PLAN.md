# Experiment plan

The repository is intentionally baseline-first.

## Phase 1: establish a trustworthy proxy benchmark

1. Normalize public/self-constructed questions to the schema in `data/README.md`.
2. Build a balanced held-out set (rough target: 20–30 questions/domain).
3. Manually audit answers and `answer_type` before treating accuracy as valid.
4. Do not tune prompts on the held-out set.

## Phase 2: run baselines before agent engineering

Run the same benchmark with fixed decoding settings and record at least:

| ID | Model | Agent mode | Thinking | Calls/question |
|---|---|---|---|---:|
| B0 | intern-s1 | direct | off | 1 |
| B1 | intern-s1 | direct | on | 1 |
| B2 | intern-s1-pro (if available) | direct | on | 1 |
| B3 | intern-s2-preview | direct | off | 1 |
| B4 | intern-s2-preview | direct | on | 1 |
| B5 | intern-s2-preview | self_refine | on | 2 |

Do not keep an extra agent module merely because it sounds sophisticated. Keep
it only if it improves held-out accuracy enough to justify latency/cost.

## Phase 3: error-driven improvements

After B0–B5, tag failures by cause:

- wrong interpretation / missing condition
- algebra or arithmetic error
- theorem/definition recall error
- long proof loses state
- final answer extraction/formatting error
- solver was correct but later module corrupted it

Only then add targeted mechanisms (router, independent verifier, retry,
self-consistency, symbolic tool, etc.), one at a time with ablation.
