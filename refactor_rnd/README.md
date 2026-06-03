# Refactor R&D: TRIT -> CONTEXT -> ABSTRACTION

This subtree is quarantined R&D. It does not change the main trainable TSM model.

Terminology used here:

```text
TRIT  = ternary vote unit: Approve / Deny / Abstain
CONTEXT = bounded relation field with order/lag-sensitive structure
ABSTRACTION = reusable classified structure
```

Vocabulary lock used in this bench:

```text
ABSTRACTOR = mechanism/process
ABSTRACTION = product/unit
KNOWLEDGE = what survives recursion
CLASSIFY = the step that cuts Knowledge into an Abstraction
```

TSM converts XP into TRITs, organizes TRITs into Contexts, and classifies stable Knowledge into Abstractions.

A Context is not a histogram and not a bag of units. A Context is a bounded,
order-sensitive relation field over TRITs or Abstractions. It must preserve
enough transition, lag, order, and frame structure for the next recursion
level to recover what the prior level meant.

The bench tests one narrow claim: higher recursion should receive structured CONTEXTS,
not histograms of lower-unit firings. A CONTEXT must preserve ordered relation,
not just transition counts.

Working stack:

```text
0 Potential
1 TRIT
2 Relation
3 Appraise
4 Evaluate
5 Divide
6 Pattern
7 Law
8 Recurse
9 Knowledge
10 Classify -> Abstraction
```

Core loop:

```text
TRITs vote.
Contexts preserve ordered relation.
Stable Contexts become Knowledge.
Knowledge is classified into Abstractions.
Abstractions re-enter new ordered Contexts.
```

Run:

```powershell
python -m pytest refactor_rnd/tests -q
python -m refactor_rnd.run_context_recursion --seeds 31,37,43,47,53 --out refactor_rnd/runs
```
