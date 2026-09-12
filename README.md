# Intern-math — R2.3 candidate

The current preferred engineering candidate is **R2.3**, tested at commit `ce56db565e1b3a3035e5cbbaf0f7c342ce31cb60`. This is not an official high-score claim. No official submission or AtomGit synchronization has been performed by these workflows.

## Deployment

Entry: `user_agent.py:ReasoningAgent`, initialized with the official injected client. Choose **intern-s2-preview-397b** in the competition UI. Thinking-off is fixed in code, independent of workflow environment variables. Temperature0.1; at most three calls with 4096/4096/2048 completion ceilings; at most two mathematical tool runs.

The candidate asks for numerical code early, executes selected standard mathematical functions, checks precision and requested numerical methods, and retains a remaining tool opportunity after an unfinished answer. Complete answers return immediately. There is no default verifier or cross-question state. A final bounded closure can retain the actual tool result and original problem without repeating an entire abandoned reasoning chain.

`math_tools.py` exposes restricted math/SymPy/NumPy/SciPy/mpmath facades with CPU, memory, wall-time and output limits, and a sanitized worker environment. This is defense in depth, not a replacement for the official OS sandbox. Requirements are pinned where relevant.

## Completed validation — keep versions and cohorts separate

- Original R2 `945f56e`, existing34-question paired regression: raw report21/34 versus R1-off21/34. Offline ChatGPT review of all saved answers found25/34 acceptable for originalR2 versus21/34 for R1, keeping ambiguous/conditional items in the denominator. This was post-hoc, not blind external verification or an official-score estimate.
- R2.1 run34661560207, nine development-selected tool cases: executor compatibility improved but numerical/closure failures remained.
- R2.2 `e1cae17`, run34662596440, thirteen development-selected questions:13 responses,13 explicit finals; reviewed core results/required arguments acceptable on11, one numerical error177, one ambiguous source192. Same11 snippets: old executor4 no-error executions versus new10 (including two no-printed-result runs).26model calls,27736completion tokens.
- R2.3 `ce56db5`, run34663369021: targeted item177 plus proof controls289 and33, all three reviewed as acceptable.177 now executes Brent root finding and returns10.0658, matching independent10.0657784357. Five model calls,6639completion tokens, one truncated primary call subsequently recovered. Do not merge these three replies with R2.2 replies to invent a single-version aggregate score.
- Final full unit suite: **139 passed**. Packaged entry and file hashes verified. Initial R2.3 CI failed before API use because a test captured mutable request lists; the test was fixed to snapshot at send time without changing solver logic.

All accuracy reviews above are ChatGPT offline reviews of actual saved model outputs, not external human certification. Some accepted requested answers have ancillary prose defects, documented in the reports. No fresh holdout or official hidden evaluation has been completed for this candidate.

## Reports

Current completion record: `docs/R23_COMPLETION_20260912.md`.
Full original34-question audit rationale: `docs/R2_OFFLINE_AUDIT_20260912.md` and `reports/r2_audit/decisions.json`.
Targeted runtime/review summaries: `reports/r2_audit/r22_r23_results.json`.
Independent mathematical checks: `python scripts/verify_r2_math.py`.
Explicit-reference offline scoring helpers: `scripts/audited_answer_checks.py` (not silently substituted into historical reports).

Historical R1 tests target its unchanged snapshot. `main`, R1 and existing submission branches were not modified by this continuation. PR#1 remains a draft. The minimal candidate contains only runtime files, requirements, deployment note and hashes; evaluation references and review decisions must never be included in the participant's inference path.
