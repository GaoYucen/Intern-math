# R2.3: do not prohibit available calculation after an unfinished answer

R2.2 run 34662596440 completed: 13 successful outputs and explicit finals, 26 calls, 27736 completion tokens, 3 truncated calls, mean43.29s/item. On identical new snippets, old executor4/11 vs new10/11. Manual-style ChatGPT review found 11 acceptable requested answers, one inaccurate ODE numeric result177, and one ambiguous covariance question192. This is a development regression, not a hidden score.

Important remaining caveats: item241 makes an unsupported claim that its valid Hoeffding bound is the tightest possible; item250's numerical iterations now execute correctly, but it labels the residual of a full-precision iterate as if evaluated at the printed rounded number, and mentions a bracket not produced by Newton. These ancillary claims are not certified. Tool success alone is not proof correctness.

Root cause for177: the primary reply never emitted code. The generic incomplete-answer followup explicitly said No code, despite a remaining model/tool budget. This forced additional hand approximations. R2.3 removes that contradiction only on this branch; no new calls/budget/model changes.

A deterministic offline replay of all13 saved R2.2 reply/tool transcripts found all12 other request sequences identical. Only177's second request changes. Final text is naturally identical under frozen replies; this is a control-flow test, not a new accuracy result.

Live validation is prospectively limited to177 and two proof controls289,33. The revised candidate must not be credited with a synthetic12/13 by combining older and newer outputs. Original inputs, references and submissions unchanged; no official evaluation is triggered.
