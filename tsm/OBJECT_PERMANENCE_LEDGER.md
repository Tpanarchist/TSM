# TSM Object-Permanence Ledger

This ledger records the current experimental status of the temporal object-continuity work. It is intentionally scoped to the synthetic temporal-object stream and should not be read as full object permanence.

## Current Status

1. Memory carries hidden object identity: yes.
2. Memory helps Reality prediction during occlusion: yes.
3. Memory shapes Definition state during occlusion: yes.
4. Object-file carries same-instance signal: yes.
5. Query path helps hard same-class file discrimination: yes.
6. Active candidate gating preserves the correct file in the lookup set: yes, scaffolded.
7. Active lookup improves constrained reappearance matching: yes.
8. Occlusion bridge survives active lookup: yes.
9. Visible feature binding remains weak: unresolved.
10. Candidate gating is learned from Definition/file geometry rather than position metadata: partial.
11. Full exact object permanence: not yet.

## Current Claim

TSM now has object-file continuity signal that survives occlusion and distinguishes same-instance identity above chance, especially against hard same-class negatives. The active candidate scaffold can preserve the correct file in the live lookup set and improve constrained reappearance lookup. A learned active gate can recover part of that live set from Definition/file geometry without position input, but it is still weaker than the scaffold. Full object permanence is still not solved because visible reappeared state does not bind cleanly back to the exact object file under global lookup, and the active gate is not yet learned from a strong perception/trajectory expectation model.

## Next Target

The next mechanism should strengthen the learned active candidate gate without making the ternary Definition layer dense. A reappearing visible state should eventually query a live candidate set shaped by learned context, recency, trajectory, and expected phase before it competes against the full object-file field.

## Active Candidate-Gating Result

Run: `runs/20260601_155155_temporal_objects_active_file_query`

The active-file query path uses recent object-file position, hit state, age, and an optional wrapped position metric for the synthetic temporal stream. The wrapped metric is needed because the held-out stream uses cyclic positions; the raw non-wrapped gate found the correct file only about half the time.

Best held-out checkpoint:

- active candidate target present fraction: `1.000`
- active candidate mean count: `7.143`
- active query exact candidate match accuracy: `0.469`
- active query hard same-class candidate match accuracy: `0.693`
- occluded memory-definition object probe delta: `+0.329`

This is a useful partial win. The active candidate set can keep the correct file live and improve constrained reappearance lookup without destroying the occluded Definition bridge. It is not full object permanence: visible feature matching remains weak, and the mechanism still depends on synthetic position metadata rather than a learned general object-file attention policy.

## Learned Gate Target

The next validation target is a learned active-file gate trained against the scaffolded gate as a teacher while withholding position metadata from the learned scorer. The learned scorer should use Definition/file geometry, memory confidence, and age first. The win condition is:

- learned active target present fraction stays high
- learned candidate count stays low/moderate
- learned hard candidate match stays above the broad query path
- occluded memory-definition object probe delta remains near the cycle/active scaffold result
- scaffold dependence can be reduced without losing the hidden-object bridge

## Learned Gate Initial Result

Run: `runs/20260601_160635_temporal_objects_active_file_query`

Best held-out checkpoint with `learned_active_file_gate_weight: 0.02`:

- learned active target present fraction: `0.876`
- learned candidate mean count: `8.000`
- learned active exact candidate match accuracy: `0.235`
- learned active hard same-class candidate match accuracy: `0.425`
- learned/scaffold recall: `0.746`
- learned/scaffold precision: `0.666`
- scaffolded active exact candidate match accuracy: `0.391`
- scaffolded active hard same-class candidate match accuracy: `0.582`
- occluded memory-definition object probe delta: `+0.305`
- ternary nonzero fraction: `0.362`

This is a useful but mixed result. The learned gate can now recover the correct file in its top-k candidate set most of the time without position input, and it preserves the occluded bridge. It is not ready to replace the scaffold: learned matching still trails scaffolded matching, and the stronger learned-gate pressure makes the ternary code much denser than the prior scaffold-only active-gate run. The next target is learned gating with better sparsity/selection pressure, not removal of the scaffold yet.

## Detached Learned Gate Result

Run: `runs/20260601_161705_temporal_objects_learned_active_file_gate`

This run stops the learned-gate teacher loss from backpropagating into the Definition/query scores. The gate still learns from Definition/file geometry, memory confidence, and age, but it cannot make the DefinitionBank denser just to satisfy the scaffold imitation target.

Best held-out checkpoint with `learned_active_file_gate_detach_inputs: true` and top-k 8 selection:

- learned active target present fraction: `0.794`
- learned candidate row coverage fraction: `1.000`
- learned target recall fraction: `0.794`
- learned candidate mean count: `8.000`
- learned active exact candidate match accuracy: `0.235`
- learned active hard same-class candidate match accuracy: `0.496`
- learned/scaffold recall: `0.603`
- learned/scaffold precision: `0.538`
- scaffolded active exact candidate match accuracy: `0.313`
- scaffolded active hard same-class candidate match accuracy: `0.598`
- occluded memory-definition object probe delta: `+0.314`
- ternary nonzero fraction: `0.186`

Evaluation-only selection sweep on the same checkpoint showed that top-k 10 is the better all-row setting:

- learned active target recall fraction: `0.834`
- learned candidate row coverage fraction: `1.000`
- learned candidate mean count: `10.000`
- learned active exact candidate match accuracy: `0.235`
- learned active hard same-class candidate match accuracy: `0.521`
- learned/scaffold recall: `0.758`
- learned/scaffold precision: `0.541`
- occluded memory-definition object probe delta: `+0.314`
- ternary nonzero fraction: `0.186`

Threshold selection at `0.6` is high-precision but incomplete:

- learned active target present fraction on covered rows: `1.000`
- learned candidate row coverage fraction: `0.500`
- learned target recall fraction across all rows: `0.500`
- learned active exact candidate match accuracy: `0.471`
- learned active hard same-class candidate match accuracy: `0.564`

The detached gate fixes the ternary densification problem: nonzero fraction drops from `0.362` to `0.186`, while the occluded bridge stays near `+0.31`. The tradeoff is recall. Top-k 10 improves all-row target recall and hard same-class matching without densifying the DefinitionBank, but it still does not solve exact global rebinding. The next target is a better learned gate scorer, not stronger gradient pressure into Definitions.

A fresh CLI run with top-k 10 saved in `runs/20260601_163012_temporal_objects_learned_active_file_gate` confirms the sparsity fix but not a stable learned-lookup win:

- best checkpoint learned target recall: `0.782`
- best checkpoint learned exact / hard match: `0.156` / `0.490`
- best checkpoint occluded bridge: `+0.263`
- best checkpoint ternary nonzero fraction: `0.079`
- latest checkpoint learned target recall: `0.772`
- latest checkpoint learned exact / hard match: `0.313` / `0.443`
- latest checkpoint occluded bridge: `+0.160`
- latest checkpoint ternary nonzero fraction: `0.078`

This makes the current result mixed. Detaching the learned-gate teacher loss prevents the DefinitionBank from becoming a dense feature bus. Top-k 10 gives full row coverage. But the learned gate still does not reliably improve exact lookup while preserving the occluded Definition bridge. The next mechanism needs better learned gate inputs or scoring structure, not more pressure on ternary activations.

## Context-Aware Learned Gate Result

Run: `runs/20260601_164308_temporal_objects_learned_active_file_gate`

This run adds model-internal context features to the learned active-file gate. The learned scorer receives the visible reappearance context, the hidden/source context attached to each candidate file, and their difference. It still does not receive raw position metadata, and the learned-gate teacher loss remains detached from Definition/query scores.

Best checkpoint:

- learned target recall: `0.844`
- learned row coverage: `1.000`
- learned exact / hard match: `0.313` / `0.556`
- learned/scaffold recall: `0.818`
- learned/scaffold precision: `0.584`
- occluded bridge: `+0.211`
- ternary nonzero fraction: `0.040`

Latest checkpoint:

- learned target recall: `0.834`
- learned row coverage: `1.000`
- learned exact / hard match: `0.313` / `0.527`
- learned/scaffold recall: `0.803`
- learned/scaffold precision: `0.573`
- occluded bridge: `+0.287`
- ternary nonzero fraction: `0.064`

This is a partial positive result. Compared with the prior fresh detached top-k 10 run, the context-aware gate improves learned target recall, hard same-class matching, scaffold agreement, and occluded bridge preservation while keeping the ternary code sparse. It still does not beat the scaffolded active gate, and the best-vs-latest split shows the same underlying tension: the checkpoint with strongest candidate discrimination is not the checkpoint with the strongest hidden Definition bridge. The next target is still a learned expectation/gating mechanism that preserves the bridge while improving exact reappearance binding.
