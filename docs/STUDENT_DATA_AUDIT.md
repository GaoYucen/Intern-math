# Audit of the student's current public-data pool

This report is based on the student package supplied on 2026-08-29, especially
`sample_data/external/combined_math/combined_math_all.jsonl` and its manifest.
It is a **data audit**, not a claim about the official hidden test distribution.
The competition material says that the organizer's set spans 18 mathematical
subfields, but the material available to us does not enumerate all 18, so the
repository's domain taxonomy is explicitly provisional.

## What is in the current pool

The student's combined candidate file contains 13,874 records from:

- Hendrycks MATH: 12,500 raw candidate records;
- MathBench: 932;
- TheoremQA: 442.

After retaining only records already marked `mapped`, normalizing answer types,
and removing exact normalized-problem duplicates, the importer keeps 11,026
candidate records. **All 11,026 are still `review_status=pending`**, so this is
not yet a trustworthy held-out benchmark.

### Coverage after normalization

| Working proxy domain | Count | Main source issue |
|---|---:|---|
| advanced algebra | 7,484 | 7,202 are Hendrycks MATH; this mapping is much broader/more elementary than a university advanced-algebra benchmark |
| mathematical analysis | 1,760 | 1,292 are Hendrycks MATH |
| discrete mathematics | 880 | 762 are Hendrycks MATH |
| probability theory | 725 | 483 are Hendrycks MATH |
| ODE | 63 | MathBench only |
| statistical inference | 46 | MathBench + TheoremQA |
| numerical analysis | 24 | TheoremQA only |
| complex analysis | 21 | TheoremQA only |
| stochastic processes | 12 | TheoremQA only |
| functional analysis | 10 | TheoremQA only |
| abstract algebra | 1 | TheoremQA only |
| measure theory | 0 | missing |
| PDE | 0 | missing |
| operations research | 0 | missing |
| topology | 0 | missing |
| differential geometry | 0 | missing |
| regression analysis | 0 | missing in the combined public pool |

The code therefore treats the 13,874-record file as a **candidate pool**, not as
`Benchmark-v1`.

## Why Hendrycks MATH should not dominate Benchmark-v1

The current subject mapper assigns most Hendrycks `algebra` questions to
`advanced_algebra`. This creates thousands of records but does not solve the
coverage problem: the task description explicitly mentions fields such as PDE,
complex analysis, topology and operations research. A very large elementary or
olympiad-style algebra pool can make aggregate accuracy look stable while still
being a poor proxy for the competition.

For initial university-level diagnostics, the safer first slice is MathBench +
TheoremQA. After deduplication that gives 1,287 mapped candidates, but it still
has serious gaps: abstract algebra (1), functional analysis (10), stochastic
processes (12), and no measure theory/PDE/operations research/topology/
differential geometry.

## Required next step before calling anything Benchmark-v1

1. Add public sources for the missing advanced domains.
2. Normalize them to the repository schema.
3. Manually audit at least 20--30 questions per target domain, including answer
   type and gold answer.
4. Mark audited rows `review_status=approved`.
5. Build the held-out set with `scripts/build_balanced_benchmark.py` **without**
   `--allow-pending` or `--allow-shortfall`.
6. Only then run B0--B5 and compare model/agent changes.

The scripts intentionally refuse to produce a supposedly trustworthy balanced
benchmark when audited domain coverage is insufficient.
