# Context-Recursion Ledger

## Purpose

This bench tests whether the failed recursion toy was testing the wrong cross-level encoding.

## Vocabulary Lock

```text
Abstractor = mechanism/process
Abstraction = product/unit
Knowledge = what survives recursion
Classify = the step that cuts surviving Knowledge into an Abstraction
```

## Context Definition

```text
Context = bounded relation field with order/lag-sensitive structure
```

A Context is not a histogram and not a bag of units. It must preserve what
happened, after what, at what lag/order position, inside what bounded frame.

The rejected mechanism:

```text
lower abstraction firings -> histogram -> next level
```

The proposed context mechanism:

```text
lower TRIT/abstraction sequence -> CONTEXT relation-space -> recurse
what survives -> Knowledge
Classify -> Abstraction
```

## Claim Boundary

Passing this bench does not prove TSM recursion works at scale. It only shows that relation-space CONTEXTS can carry structure that histograms erase in small synthetic worlds, so stable Knowledge can survive long enough to be cut into reusable Abstractions.

## Probe C Result

Run artifact:

```text
refactor_rnd/runs/20260603_093618_context_recursion
```

Level-3 collapse is currently diagnosed as information loss in the transition-only encoder, not as a hard recursion-depth wall.

Key result:

```text
transition level3:            abstraction_nmi 0.000, supervised_probe_accuracy 0.500
transition_lag_order level3:  abstraction_nmi 1.000, supervised_probe_accuracy 1.000
```

Interpretation:

```text
signal gone under transition-only context = representational collapse
signal restored by lag/order context = order-sensitive context still carries the regime
```

The classify-before-recursion ablation did not isolate the failure. Both paths reached perfect level-3 separation in this toy when the next level consumed richer vector contexts:

```text
raw_context level3:              abstraction_nmi 1.000
classified_abstraction level3:   abstraction_nmi 1.000
```

Current reading:

```text
The next fragile joint is not "depth itself."
It is the definition of Context.
Transition-only Context is too weak at level 3.
Order/lag-aware Context preserves the signal.
```

Operational consequence:

```text
TRITs vote.
Contexts preserve ordered relation.
Stable Contexts become Knowledge.
Knowledge is classified into Abstractions.
Abstractions re-enter new ordered Contexts.
```

## Next Stress Cases

The next bench should hold the architecture fixed and stress the Context definition harder:

- variable lag
- noisy order
- missing events
- repeated symbols
- ambiguous shared anchors

The active question is not whether Context works in the toy. It is how much order/lag structure Context can lose before recursion collapses.

## Probe D Result

Run artifact:

```text
refactor_rnd/runs/20260603_184401_context_recursion
```

Probe D says the tightened Context definition is directionally correct.

Key result:

```text
transition_lag_order beats transition in 6/6 stress cases
```

That means order/lag-aware Context is not just helping on one toy corner case. It survives every stress family better than transition-only Context in this bench:

```text
variable lag
noisy order
missing events
repeated symbols
ambiguous shared anchors
frame shift
```

Current reading:

```text
Context is robust relative to transition-only encoding.
Context is not free or unlimited.
The current toy still shows a narrow degradation margin: the first coarse threshold appears around 0.25 for the order-aware encoder.
```

Important nuance:

```text
The summary now matches the rows: controlled loss hurts level-3 signal in 6/6 stress cases.
That is a row-level damage claim, not a threshold-gap claim.
The coarse threshold is still tied at about 0.25 across the order-aware encoders in this toy,
so threshold and row-level semantic damage must be read separately.
```

Severity ranking:

```text
mean level-3 damage = level3_nmi_damage + level3_probe_damage

noisy_order:              1.039
ambiguous_shared_anchors: 0.750
frame_shift:              0.577
variable_lag:             0.551
repeated_symbols:         0.500
missing_events:           0.250
```

Current ranking read:

```text
noisy_order is the most damaging stressor in the current toy.
missing_events is the least damaging of the six in the current toy.
The four-object wall should therefore be read as ordered Context degradation under load,
with ordering noise and anchor ambiguity currently ranking above literal missing events.
```

Stress-transform note:

```text
variable_lag is now literal insertion of delay steps.
missing_events is now literal deletion of steps.
That makes the bench closer to delayed reappearance and occlusion, rather than a fixed-length approximation.
```

## Four-Object Wall Mapping

This is a bridge note for main TSM diagnostics, not a main-architecture change.

Current intervention priority should follow mean level-3 damage, not the coarse threshold crossing alone.

| priority | toy stressor | likely object-permanence analogue | possible main-TSM diagnostic |
|---:|---|---|---|
| 1 | noisy_order | curved motion, slot crossing or swapping, timing jitter, wrong phase alignment, same-class reappearance confusion | context_order_damage, candidate_order_flip_rate, prediction_observation_phase_error, slot_rank_instability, trajectory_order_consistency, reappearance_order_confusion |
| 2 | ambiguous_shared_anchors | shared local features, similar trajectories, anchor aliasing under same-class load | reappearance_order_confusion, slot_rank_instability, candidate_order_flip_rate |
| 3 | frame_shift | window-boundary mismatch, wrong temporal chunk, delayed prediction-observation alignment | prediction_observation_phase_error, context_order_damage, trajectory_order_consistency |
| 4 | variable_lag | delayed reappearance, timing drift, uneven latency between prediction and observation | prediction_observation_phase_error, trajectory_order_consistency, reappearance_order_confusion |
| 5 | repeated_symbols | same-class objects, identity aliasing with rough order still partly intact | slot_rank_instability, reappearance_order_confusion |
| 6 | missing_events | occlusion, unobserved frames, brief disappearance with some order still recoverable | context_order_damage, trajectory_order_consistency |

Current bridge claim:

```text
The four-object wall is likely ordered Context degradation under load, not mere absence under load.
The first intervention target in main TSM should therefore be order, phase, and rank stability diagnostics rather than generic memory expansion.
```

## Probe E Result

Run artifact:

```text
refactor_rnd/runs/20260603_192040_context_recursion
```

Probe E splits level-3 damage into semantic confusion and order loss.

Current decomposition:

```text
semantic_confusion_score = full_ordered_abstraction_nmi - degraded_ordered_abstraction_nmi
order_loss_score = full_order_recovery_accuracy - degraded_order_recovery_accuracy
total_damage_score = semantic_confusion_score + order_loss_score
```

Key result:

```text
most_total_damage_case: noisy_order
most_order_loss_case: variable_lag
most_semantic_confusion_case: noisy_order
least_total_damage_case: missing_events
```

Case readout:

```text
noisy_order:              semantic 0.789, order 0.250, total 1.039 -> Context-to-Abstraction bridge failure
variable_lag:             semantic 0.176, order 0.500, total 0.676 -> bridge failure with strongest order-loss component
frame_shift:              semantic 0.577, order 0.000, total 0.577 -> Abstraction/classification problem
ambiguous_shared_anchors: semantic 0.000, order 0.250, total 0.250 -> Context ordering problem
repeated_symbols:         semantic 0.000, order 0.250, total 0.250 -> Context ordering problem
missing_events:           semantic 0.000, order 0.000, total 0.000 -> low damage in the current toy
```

Current interpretation:

```text
The worst overall stressor is still noisy_order, and it damages both semantics and ordered relation.
The strongest pure order-loss signal is variable_lag.
Frame shift currently looks more like a classification/abstraction failure than an order-only failure.
Ambiguous anchors and repeated symbols look like order/integrity failures with less semantic collapse.
Missing events remain the lightest stressor in this toy.
```

Operational consequence:

```text
If main TSM shows noisy order and delayed reappearance failures first, improve ordered Context stability.
If frame-shift-like failures dominate, inspect the Context -> Abstraction handoff and classification boundary.
If both move together, the bridge is too brittle under load.
```

## Probe F Result

Run artifact:

```text
refactor_rnd/runs/20260603_200602_context_recursion
```

Probe F splits Probe E ordered implication damage into phase drift versus residual rank instability.

Current decomposition:

```text
phase_error_score = ordered-implication damage share attributable to extra aligned-slot relief under translation-tolerant matching
rank_instability_score = remaining ordered-implication damage not rescued by alignment
ordered_implication_damage_score = phase_error_score + rank_instability_score
```

Key result:

```text
most_total_order_loss_case: variable_lag
most_ordered_implication_damage_case: variable_lag
most_phase_error_case: ambiguous_shared_anchors
most_rank_instability_case: variable_lag
least_total_order_loss_case: missing_events
phase_control_validated: True
```

Case readout:

```text
variable_lag:             phase 0.018, rank 0.482, total 0.500 -> rank instability problem
noisy_order:              phase 0.000, rank 0.250, total 0.250 -> rank instability problem
repeated_symbols:         phase 0.034, rank 0.216, total 0.250 -> rank instability problem
ambiguous_shared_anchors: phase 0.044, rank 0.206, total 0.250 -> rank instability problem
pure_phase_offset:        phase 0.020, rank 0.042, total 0.062 -> synthetic phase control with low rank instability
frame_shift:              phase 0.000, rank 0.000, total 0.000 -> low order damage
missing_events:           phase 0.000, rank 0.000, total 0.000 -> low order damage
```

Current interpretation:

```text
The synthetic pure_phase_offset control now produces measurable phase error, so Probe F's phase side is not blind.
variable_lag still does not behave like a recoverable global phase offset; it remains overwhelmingly rank instability.
pure_phase_offset carries much less rank instability than variable_lag and noisy_order, which is the intended phase-control behavior.
ambiguous_shared_anchors now carries the largest absolute phase share among the natural stressors in this toy, while still remaining rank-instability dominated overall.
Frame shift continues to look semantic/classificatory rather than ordered-implication-damage-dominant.
```

POV Interpretation:

```text
phase drift = same thread, shifted in time
rank instability = candidate threads lost lawful ordering
current toy result = variable_lag maps to rank instability, not clean phase drift
```

Operational consequence:

```text
Probe F can now detect phase drift when a synthetic offset-only control is present, which makes the variable_lag result stronger rather than weaker.
For main TSM, delayed-reappearance failures should still first inspect slot_rank_instability, candidate_order_flip_rate,
reappearance_order_confusion, and relation_identity_flip behavior before assuming pure prediction-observation phase drift.
prediction_observation_phase_error remains a valid diagnostic family, but it is no longer the default explanation for variable_lag-style failure.
```

Refined Context definition:

```text
Context = bounded relation field preserving ranked ordered implications.
```

So the active question narrows again:

```text
Which main TSM failures look like the validated pure_phase_offset control,
and which ones still look like variable_lag-style rank instability under higher load?
```

## Kill Conditions

- If same-marginal histograms recover regimes, the task is not isolating relations cleanly.
- If CONTEXT relation-space fails on same-marginal or shared-anchor tasks, the context-dimension encoding is not sufficient even in the toy setting.
- If novelty does not re-spike Abstain, the developmental thermometer claim is not reproduced here.
