# Refactor R&D: TRIT -> TRYTE -> TRION

This subtree is quarantined R&D. It does not change the main trainable TSM model.

Terminology used here:

```text
TRIT  = primitive ternary vote: Approve / Deny / Abstain
TRYTE = variable-size relation packet / pocket dimension
TRION = stable classified knowledge unit
```

A TRION is TSM's Cognit in ternary-native vocabulary.

The bench tests one narrow claim: higher recursion should receive structured TRYTES,
not histograms of lower-unit firings. A histogram keeps marginal counts and discards
off-diagonal relations. A TRYTe relation-space encoder preserves directed transitions.

Readable cycle:

```text
XP -> TRITs -> relations -> TRYTES -> Appraise/Evaluate/Divide/Pattern/Law/Recurse/Knowledge/Classify -> TRIONs
```

Run:

```powershell
python -m pytest refactor_rnd/tests -q
python -m refactor_rnd.run_pocket_recursion --seeds 31,37,43,47,53 --out refactor_rnd/runs
```
