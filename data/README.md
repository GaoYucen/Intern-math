# Benchmark data

Do **not** put the competition hidden test set, hidden answers, or private judge
artifacts here.

Use public or self-constructed data and normalize each row to JSONL:

```json
{"idx": 0, "problem": "...", "answer": "...", "answer_type": "numeric", "domain": "pde", "difficulty": "hard", "source_dataset": "...", "review_status": "approved"}
```

Recommended `answer_type` values:

- `integer`, `float`, `numeric`, `rational`
- `symbolic`
- `set` / `multiple_values`
- `choice`
- `boolean`
- `text`
- `proof` (not auto-scored; send to manual or LLM-judge review)

## Candidate pool vs. held-out benchmark

A large imported pool is not automatically a benchmark. Imported rows should
start as `review_status=pending`. Before a row enters the held-out benchmark,
a human should verify at least the problem statement, gold answer, answer type,
domain label and whether the problem is self-contained.

The first target is a balanced proxy benchmark, not a huge imbalanced pile of
questions. Aim for 20--30 auditable questions per target domain, stratified by
difficulty where possible. Keep a separate held-out benchmark for comparing
agent changes.

For the student's existing combined public pool, run:

```bash
python scripts/import_student_pool.py \
  --input /path/to/combined_math_all.jsonl \
  --output data/student_pool_candidates.jsonl \
  --coverage reports/student_pool_coverage.json

python scripts/audit_dataset.py data/student_pool_candidates.jsonl \
  --output reports/student_pool_audit.json
```

See `docs/STUDENT_DATA_AUDIT.md` for the current coverage diagnosis.
