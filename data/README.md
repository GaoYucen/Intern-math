# Benchmark data

Do **not** put the competition hidden test set, hidden answers, or private judge
artifacts here.

Use public or self-constructed data and normalize each row to JSONL:

```json
{"idx": 0, "problem": "...", "answer": "...", "answer_type": "numeric", "domain": "pde", "difficulty": "hard", "source": "..."}
```

Recommended `answer_type` values:

- `integer`, `float`, `numeric`, `rational`
- `symbolic`
- `set` / `multiple_values`
- `choice`
- `boolean`
- `text`
- `proof` (not auto-scored; send to manual or LLM judge review)

The first target is a balanced proxy benchmark, not a huge imbalanced pile of
questions. Aim for 20–30 auditable questions per target domain, stratified by
difficulty where possible. Keep a separate held-out benchmark for comparing
agent changes.
