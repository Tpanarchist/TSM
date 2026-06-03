# Refactor R&D: TRIT -> CONTEXT -> ABSTRACTION

This subtree is quarantined R&D. It does not change the main trainable TSM model.

Terminology used here:

```text
TRIT  = primitive ternary vote: Approve / Deny / Abstain
CONTEXT = variable-size relation context / context dimension
ABSTRACTION = stable classified knowledge unit
```

An ABSTRACTION is the stable classified unit produced from contexts.

The bench tests one narrow claim: higher recursion should receive structured CONTEXTS,
not histograms of lower-unit firings. A CONTEXT relation-space encoder preserves directed transitions.

Readable cycle:

```text
XP -> TRITs -> relations -> CONTEXTS -> Appraise/Evaluate/Divide/Pattern/Law/Recurse/Knowledge/Classify -> ABSTRACTIONS
```

Run:

```powershell
python -m pytest refactor_rnd/tests -q
python -m refactor_rnd.run_context_recursion --seeds 31,37,43,47,53 --out refactor_rnd/runs
```
