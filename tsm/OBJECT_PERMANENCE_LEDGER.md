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
11. Context-aware learned candidate gating improves lookup without densifying Definitions: yes.
12. Object files carry explicit phase/trajectory state: yes, scaffolded.
13. Learned phase/trajectory dynamics predicts reappearance position better than naive velocity: yes.
14. Object-file expectation predicts its own future Definition state: partial, still weak.
15. Full exact object permanence: not yet.

## Current Claim

TSM now has object-file continuity signal that survives occlusion and distinguishes same-instance identity above chance, especially against hard same-class negatives. The active candidate scaffold can preserve the correct file in the live lookup set and improve constrained reappearance lookup. A learned active gate can recover part of that live set from Definition/file geometry and context without position input, but it is still weaker than the scaffold. Object files now carry scaffolded phase/trajectory state, and a learned dynamics head can predict reappearance position much better than the naive velocity projection. This position-level win does not yet produce a strong future Definition/query prediction under held-out evaluation. Full object permanence is still not solved because visible reappeared state does not bind cleanly back to the exact object file under global lookup.

## Next Target

The next mechanism should turn the learned position dynamics into representation-level expectation without making the ternary Definition layer dense. A reappearing visible state should eventually query a live candidate set shaped by learned context, recency, trajectory, expected phase, predicted reappearance region, and predicted Definition state before it competes against the full object-file field.

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

## Residual Object-File Expectation Result

Run: `runs/20260601_165442_temporal_objects_learned_active_file_gate`

Committed config for this experiment: `configs/temporal_objects_expected_active_file_gate.yaml`

This run adds a residual object-file expectation head. The head receives the active object-file score, hidden/source context, file confidence, and age, then predicts an expected reappearance query. The learned active gate can also receive the expected query, its difference from the visible query, and their product. Inputs to the expectation head are detached by default so the expectation objective cannot make the DefinitionBank dense.

Best held-out checkpoint:

- learned target recall: `0.834`
- learned row coverage: `1.000`
- learned exact / hard match: `0.235` / `0.521`
- expected-file exact / hard match: `0.000` / `0.238`
- scaffolded active exact / hard match: `0.391` / `0.688`
- occluded bridge: `+0.381`
- ternary nonzero fraction: `0.089`

Latest held-out checkpoint:

- learned target recall: `0.834`
- learned row coverage: `1.000`
- learned exact / hard match: `0.156` / `0.521`
- expected-file exact / hard match: `0.000` / `0.156`
- scaffolded active exact / hard match: `0.235` / `0.582`
- occluded bridge: `+0.206`
- ternary nonzero fraction: `0.085`

This is a useful mixed result, not an object-permanence win. The expectation path is wired, trainable, and does not densify the DefinitionBank. It also does not destroy the learned candidate recall. But the expected query itself fails exact held-out reappearance matching, and learned candidate matching is weaker than the prior context-aware run. The positive signal is bridge preservation, especially in the best checkpoint. The negative signal is that a residual file-plus-context expectation is not enough to predict "this file expected this reappearance." The next version needs richer phase/trajectory expectation or a more direct expected-state target, not stronger pressure on the current head.

## Direct Future-State Expectation Result

Run: `runs/20260601_170813_temporal_objects_expected_future_state`

Config: `configs/temporal_objects_expected_future_state.yaml`

This run changes the expectation objective from "predict the reappeared query projection" to "predict the future raw Definition state," then derives an expected query projection from that predicted state for the learned gate. The loss is split into a paired future-state term and a hard same-class/different-instance term. Gate inputs remain detached, so the learned-gate teacher cannot update the expectation head.

Best held-out checkpoint:

- expectation pair / hard loss: `3.275` / `1.373`
- expected-state exact / hard match: `0.000` / `0.156`
- expected-file exact / hard match: `0.000` / `0.235`
- learned target recall: `0.844`
- learned exact / hard match: `0.313` / `0.515`
- scaffolded active exact / hard match: `0.469` / `0.693`
- occluded bridge: `+0.241`
- ternary nonzero fraction: `0.112`

Latest held-out checkpoint:

- expectation pair / hard loss: `3.364` / `1.362`
- expected-state exact / hard match: `0.000` / `0.289`
- expected-file exact / hard match: `0.000` / `0.238`
- learned target recall: `0.860`
- learned exact / hard match: `0.313` / `0.515`
- scaffolded active exact / hard match: `0.469` / `0.684`
- occluded bridge: `+0.237`
- ternary nonzero fraction: `0.108`

This is another useful negative result. The future-state loss is active and decreases compared with the earlier query-target run, but the expected state still fails exact held-out matching. The final train-batch summary shows nonzero expected-state matching, while held-out exact remains `0.000`, so the current head is not learning a reusable reappearance expectation. It also does not improve the context-aware learned gate: exact matching is flat, hard matching is slightly weaker, the occluded bridge is weaker, and ternary density is higher than the prior context-aware run. The next target should add a real phase/trajectory state to the object file rather than increasing pressure on the same file-plus-context expectation head.

## Phase/Trajectory Object-File State Result

Run: `runs/20260601_172419_temporal_objects_expected_trajectory_state`

Config: `configs/temporal_objects_expected_trajectory_state.yaml`

This run adds explicit object-file phase/trajectory state. Memory now stores last visible position, simple visible-to-visible velocity, and last visible phase. The expectation head can use normalized last position, velocity, projected position, current/next phase one-hot state, and visibility flags. This is still scaffolded: phase and position metadata come from the synthetic stream.

Best held-out checkpoint:

- expectation pair / hard loss: `2.827` / `1.493`
- expected-state exact / hard match: `0.078` / `0.229`
- expected-file exact / hard match: `0.127` / `0.301`
- learned target recall: `0.866`
- learned exact / hard match: `0.240` / `0.518`
- scaffolded active exact / hard match: `0.319` / `0.693`
- occluded bridge: `+0.145`
- ternary nonzero fraction: `0.083`
- trajectory position error: `0.364`

Latest held-out checkpoint:

- expectation pair / hard loss: `2.794` / `1.385`
- expected-state exact / hard match: `0.078` / `0.336`
- expected-file exact / hard match: `0.000` / `0.336`
- learned target recall: `0.814`
- learned exact / hard match: `0.235` / `0.560`
- scaffolded active exact / hard match: `0.313` / `0.701`
- occluded bridge: `+0.291`
- ternary nonzero fraction: `0.060`
- trajectory position error: `0.364`

This is a partial result. Adding phase/trajectory state makes the expectation losses lower and improves hard expected-state/file matching compared with the direct future-state run, while ternary remains sparse. But exact expected-state matching is still only at the paired chance floor, learned exact lookup does not improve, and the simple velocity projection is a bad predictor of the wrapped reappearance position. The useful lesson is narrower: object files now carry the missing state substrate, but the current projection is too crude. The next target is a learned phase-transition/trajectory model for reappearance position or phase-conditioned object-file prediction, not more pressure on the expectation head.

## Learned Phase/Trajectory Dynamics Result

Run: `runs/20260601_174437_temporal_objects_learned_trajectory_state`

Config: `configs/temporal_objects_learned_trajectory_state.yaml`

This run adds a learned object-file dynamics head. The head receives trajectory features, hidden/source context, file confidence, and age, then predicts a residual correction over the naive projected reappearance position. The corrected position can replace the naive projection inside the expectation features. The head is zero-initialized, so it starts equivalent to naive velocity projection and must earn any improvement through training.

Best held-out checkpoint:

- dynamics loss: `0.007424`
- trajectory position error: `0.364`
- learned dynamics position error: `0.142`
- learned dynamics position improvement: `+0.222`
- expected-state exact / hard match: `0.133` / `0.211`
- expected-file exact / hard match: `0.156` / `0.235`
- learned target recall: `0.866`
- learned exact / hard match: `0.235` / `0.419`
- scaffolded active exact / hard match: `0.313` / `0.577`
- occluded bridge: `+0.485`
- ternary nonzero fraction: `0.106`

Latest held-out checkpoint:

- dynamics loss: `0.007096`
- trajectory position error: `0.364`
- learned dynamics position error: `0.138`
- learned dynamics position improvement: `+0.226`
- expected-state exact / hard match: `0.127` / `0.127`
- expected-file exact / hard match: `0.049` / `0.127`
- learned target recall: `0.866`
- learned exact / hard match: `0.235` / `0.387`
- scaffolded active exact / hard match: `0.313` / `0.479`
- occluded bridge: `+0.341`
- ternary nonzero fraction: `0.107`

This is a position-level success and a representation-level mixed result. The learned dynamics head clearly beats the naive velocity projection on held-out wrapped reappearance position, cutting normalized position error from about `0.364` to about `0.14`. It also keeps the occluded bridge alive and does not cause ternary densification beyond the recent expectation runs. But the better position estimate does not yet translate into stronger expected Definition-state matching or learned global file rebinding. The next target is not more dynamics loss; it is making the predicted region/phase condition a better future Definition-state expectation.
