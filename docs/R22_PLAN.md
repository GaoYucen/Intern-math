# R2.2 narrow continuation

R2.1 run 34661560207 completed: 9/9 API success, 8/9 explicit final, 22 model calls, 5 truncated calls, 40011 completion tokens. On 11 identical newly generated snippets, old worker accepted 2 and repaired worker accepted 5, with 3 new-only successes and no old-only success. The tools still rejected two harmless docstrings and numerical package imports. Code also used forbidden dir(), which must remain rejected.

More importantly, item 263 executed successfully but stopped bisection before its interval was wholly on one side of the rounding threshold, returning 2.36 rather than 2.37. Item 177 never requested code and repeated its derivation through three truncated responses. Item 250 still had inaccurate synthetic-division arithmetic after tool failures. Thus execution success does NOT imply a correct answer.

Changes for R2.2:
- Permit inert leading module/function docstrings, not arbitrary expression strings.
- Install pinned standard numpy/scipy/mpmath dependencies and expose only selected numeric functions through restricted facades. Continue rejecting IO, random state, introspection, sympify/parse_expr/lambdify, and dynamic format expressions.
- Request numerical code early, before long hand approximations; explicitly require the requested numerical method and verified rounding/residuals.
- Last-budget closure uses original question, last draft and actual tool result, requests the answer first and bounded essential justification. It does not append another full abandoned derivation chain.
- Same 397B, thinking-off, temperature 0.1, and 4096/4096/2048 ceilings; no unconditional verifier, no more calls, no official submission.

Validation is prospectively fixed to the same 9 numerical/tool cases [47,177,192,200,214,241,250,263,272] plus four accepted proof regressions [289,150,209,33]. This is a targeted development test, not a new holdout or aggregate official-score estimate. Item 192's source remains ambiguous. All newly generated snippets are replayed in old/new workers to compare compatibility on identical code. Final mathematics is reviewed separately. Original data and outputs stay intact.
