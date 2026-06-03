# Pocket-Recursion Ledger

## Purpose

This bench tests whether the failed recursion toy was testing the wrong cross-level encoding.

The rejected mechanism:

```text
lower TRION firings -> histogram -> next level
```

The proposed pocket mechanism:

```text
lower TRIT/TRION sequence -> TRYTe relation-space -> next level
```

## Claim Boundary

Passing this bench does not prove TSM recursion works at scale. It only shows that relation-space TRYTES can carry structure that histograms erase in small synthetic worlds.

## Kill Conditions

- If same-marginal histograms recover regimes, the task is not isolating relations cleanly.
- If TRYTe relation-space fails on same-marginal or shared-anchor tasks, the pocket-dimension encoding is not sufficient even in the toy setting.
- If novelty does not re-spike Abstain, the developmental thermometer claim is not reproduced here.
