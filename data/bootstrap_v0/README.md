# bootstrap_v0

`bootstrap_v0` is an **exploratory, unreviewed** 160-question set used only to
validate the competition-shaped data/runner/evaluator pipeline before
`Benchmark-v1` is frozen.

It is built from the student's normalized public pool with:

```bash
python scripts/import_student_pool.py \
  /path/to/combined_math_all.jsonl \
  --output tmp/student_candidates.jsonl

python scripts/build_bootstrap_v0.py \
  tmp/student_candidates.jsonl \
  --output-dir data/bootstrap_v0
```

Selection policy:

- only MathBench + TheoremQA;
- Hendrycks MATH is excluded;
- 20 questions from each of 8 domains;
- deterministic seed 2026;
- source-diverse sampling within each domain.

Current 8 domains are `advanced_algebra`, `complex_analysis`,
`discrete_mathematics`, `mathematical_analysis`, `numerical_analysis`, `ode`,
`probability_theory`, and `statistical_inference`.

The generated `input.jsonl` contains only `idx` and `problem` and therefore
matches the local competition input contract. `gold.jsonl` is for local
scoring only.

Do **not** report results on this set as Benchmark-v1: its labels/domain mapping
have not received the required human audit and it does not cover the missing
advanced domains such as PDE and topology comprehensively.
