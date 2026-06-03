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
14. Predicted reappearance position is load-bearing for local file-to-slot binding: yes, on curved contested motion.
15. Position channels are load-bearing inside feature-only binding: yes, under ablation control.
16. Binding representations preserve recoverable position: yes for object-local slots; partial for scene/global binding states.
17. Same-class contested local file-to-slot continuity: passed for 2 objects across seeds and initial-pass for 3 objects.
18. Four-object same-class local binding: partial; learned dynamics beats ballistic but p90 endpoint error exceeds spacing.
19. Global same-class file retrieval: not yet; likely related to the same density/candidate-count curve.
20. Object-file expectation predicts its own future Definition state: partial, still weak.
21. Runtime confidence can expose endpoint error: partial; works on easier 2/3-object conditions but fails at the 4-object load wall.
22. Full exact object permanence: not yet.

## Current Claim

TSM now has object-file continuity signal that survives occlusion and distinguishes same-instance identity above chance in the original single-target stream. The active candidate scaffold can preserve the correct file in the live lookup set and improve constrained reappearance lookup. Object-local slots solve the visible same-class scene-mush problem in the contested stream. Oracle endpoint binding proves the slot assignment logic is correct, and the curved contested stream proves learned trajectory dynamics can beat hand ballistic motion and recover perfect local file-to-slot binding for two objects across seeds and for an initial three-object count sweep. Four-object local binding is partial rather than solved; spacing-vs-error diagnostics show learned p90 endpoint error exceeds the four-object spacing budget. Full object permanence is still not solved because global same-class file retrieval and live candidate-set control remain weak.

## Next Target

The next mechanism should focus on binding under load after separating density from logic. Local slots, local assignment, and learned trajectory dynamics work under the curved contested gate through three objects, but four-object local set binding degrades once endpoint error approaches or exceeds inter-object spacing. First test a wider-frame or wider-spacing four-object variant; if that passes, the current failure is density/precision. If it still fails, the candidate/load mechanism itself needs repair. Do not add governance, action, or broad similarity heads before fixing this candidate/load problem.

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

## Predicted-Position Candidate Probe

Evaluated checkpoint: `runs/20260601_174437_temporal_objects_learned_trajectory_state`

This probe removes the oracle from active candidate masking. The old true-future-position candidate set is kept only as an `oracle_position` diagnostic. The active candidate path now uses the learned dynamics predicted position when available; a feature-only/no-position candidate set includes all valid live files and serves as the baseline.

Best held-out checkpoint:

- dynamics position error: `0.142`
- dynamics position improvement over naive: `+0.222`
- oracle-position exact / hard match: `0.313` / `0.577`
- predicted-position exact / hard match: `0.156` / `0.317`
- predicted-position target recall: `0.584`
- feature-only exact / hard match: `0.235` / `0.381`
- feature-only target recall: `1.000`
- occluded bridge: `+0.485`
- ternary nonzero fraction: `0.106`
- Definition position linear improvement: `-0.032`
- file-query position linear improvement: `-0.049`
- memory-conditioned Definition position linear improvement: `-0.329`

Latest held-out checkpoint:

- dynamics position error: `0.138`
- dynamics position improvement over naive: `+0.226`
- oracle-position exact / hard match: `0.313` / `0.479`
- predicted-position exact / hard match: `0.156` / `0.317`
- predicted-position target recall: `0.584`
- feature-only exact / hard match: `0.235` / `0.319`
- feature-only target recall: `1.000`
- occluded bridge: `+0.341`
- ternary nonzero fraction: `0.107`
- Definition position linear improvement: `-0.026`
- file-query position linear improvement: `-0.035`
- memory-conditioned Definition position linear improvement: `-0.326`

This triggers the Probe 1 kill condition. The learned dynamics head predicts position well, but using that predicted position as the candidate key performs worse than the feature-only/no-position baseline and far worse than the oracle-position candidate set. The position-recoverability diagnostic is also negative: the current pooled Definition/raw-score, file-query, and memory-conditioned Definition representations do not linearly recover reappearance position better than a centroid baseline. The next target is representation/interface repair: position-aware or slot-aware binding, or a local prediction-error competition where object files own reappearance by lowest prediction error against the percept.

## Position-Aware Binding Representation Result

Run: `runs/20260601_203942_temporal_objects_position_aware_binding`

Config: `configs/temporal_objects_position_aware_binding.yaml`

This patch adds a narrow position-aware binding representation without adding new losses or expectation heads. Visible reappearance features append a salience-derived percept position. File-side binding features append the learned predicted reappearance position when available. The raw DefinitionBank no longer injects coordinates into ternary axes, so geometry is exposed to binding diagnostics without turning Definitions into a coordinate bus.

Best held-out checkpoint:

- Definition position linear improvement / R2: `+0.146` / `0.763`
- file-query position linear improvement / R2: `+0.151` / `0.779`
- memory-conditioned Definition position linear improvement / R2: `-0.011` / `0.062`
- dynamics position error / improvement: `0.144` / `+0.220`
- oracle-position exact / hard match: `0.391` / `0.368`
- predicted-position exact / hard match: `0.235` / `0.317`
- feature-only exact / hard match: `0.235` / `0.319`
- occluded bridge: `+0.274`
- ternary nonzero fraction: `0.235`

Latest held-out checkpoint:

- Definition position linear improvement / R2: `+0.152` / `0.781`
- file-query position linear improvement / R2: `+0.157` / `0.803`
- memory-conditioned Definition position linear improvement / R2: `-0.012` / `0.057`
- dynamics position error / improvement: `0.137` / `+0.227`
- oracle-position exact / hard match: `0.391` / `0.463`
- predicted-position exact / hard match: `0.235` / `0.317`
- feature-only exact / hard match: `0.235` / `0.458`
- occluded bridge: `+0.129`
- ternary nonzero fraction: `0.257`

This clears the recoverability part of the interface repair for visible reappeared Definition/file-query features, but not for memory-conditioned source features. It also triggers the next kill gate: once position is recoverable, predicted-position candidate matching still fails to beat the feature-only candidate baseline. A brief variant that pushed the appended coordinate representation directly through the active query loss made ternary dense and weakened the bridge, so that path was rejected. The next target is prediction-error binding, not stronger similarity pressure.

## Position-Ablated Feature-Only Control

Evaluated checkpoint: `runs/20260601_203942_temporal_objects_position_aware_binding`

This control compares the feature-only candidate lookup with the full position-aware binding vector against the same lookup with the appended coordinate channels removed. The candidate mask is unchanged, so the only difference is whether the feature distance can see the binding-position channels.

Best held-out checkpoint:

- full feature-only exact / hard match: `0.235` / `0.319`
- position-ablated feature-only exact / hard match: `0.078` / `0.240`
- full Definition position improvement / R2: `+0.146` / `0.763`
- ablated Definition position improvement / R2: `-0.084` / `-0.619`
- full file-query position improvement / R2: `+0.151` / `0.779`
- ablated file-query position improvement / R2: `-0.110` / `-0.892`
- row coverage / target recall: `1.000` / `1.000` for both paths
- occluded bridge: `+0.274`
- ternary nonzero fraction: `0.235`

Latest held-out checkpoint:

- full feature-only exact / hard match: `0.235` / `0.458`
- position-ablated feature-only exact / hard match: `0.156` / `0.385`
- full Definition position improvement / R2: `+0.152` / `0.781`
- ablated Definition position improvement / R2: `-0.077` / `-0.596`
- full file-query position improvement / R2: `+0.157` / `0.803`
- ablated file-query position improvement / R2: `-0.107` / `-0.895`
- row coverage / target recall: `1.000` / `1.000` for both paths
- occluded bridge: `+0.129`
- ternary nonzero fraction: `0.257`

This resolves the ambiguity in the prior feature-only column. The position-aware feature-only path is not a pure non-geometric baseline: it is already using the coordinate channels, and removing them weakens both exact and hard same-class matching. The explicit predicted-position mask still fails, but geometry is now load-bearing through feature distance. The next step is to re-baseline and stabilize this partial win before building prediction-error competition.

## Contested Two-Object Result

Run: `runs/20260601_210546_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

This run adds a target-centric same-class contested stream. Each scene has two same-shape object tracks. Each row selects one track as the target, occludes that target in the occlusion phases, and leaves the other track visible as a distractor. Memory uses `object_file_id`, so both tracks in the same scene maintain separate object files. Held-out evaluation confirms all rows are contested:

- temporal same-class contested fraction: `1.000`

Best held-out checkpoint:

- full feature-only exact / hard match: `0.157` / `0.157`
- position-ablated feature-only exact / hard match: `0.079` / `0.079`
- predicted-position exact / hard match: `0.236` / `0.236`
- oracle-position exact / hard match: `0.157` / `0.157`
- row coverage / target recall: `1.000` / `1.000`
- Definition position improvement / R2: `-0.065` / `-0.273`
- file-query position improvement / R2: `-0.065` / `-0.271`
- dynamics position error / improvement: `0.230` / `+0.169`
- occluded bridge: `0.000`
- ternary nonzero fraction: `0.110`

Latest held-out checkpoint:

- full feature-only exact / hard match: `0.157` / `0.157`
- position-ablated feature-only exact / hard match: `0.079` / `0.079`
- predicted-position exact / hard match: `0.236` / `0.236`
- oracle-position exact / hard match: `0.157` / `0.157`
- row coverage / target recall: `1.000` / `1.000`
- Definition position improvement / R2: `-0.061` / `-0.252`
- file-query position improvement / R2: `-0.062` / `-0.255`
- dynamics position error / improvement: `0.227` / `+0.171`
- occluded bridge: `0.000`
- ternary nonzero fraction: `0.110`

This is the intended discriminating failure. The single-target geometry-through-feature win does not survive same-class contested identity. Position channels still help slightly over ablated features, but position recoverability collapses because the visible binding position is a whole-image salience read over both objects, not a target-specific object slot. Explicit predicted-position masking is slightly better than full feature distance in this run, but still weak and not enough to solve contested same-instance rebinding. The next step is Probe 2: active object files must compete by prediction error against the percept, rather than relying on global feature distance or a whole-image position channel.

## Local Prediction-Error Binding Probe

Run: `runs/20260601_212900_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

This patch first audits the `object_file_id` path. `object_file_id` is used as the object-memory storage key and as an auxiliary/evaluation instance label, but it is not used to filter or select bind-time candidate masks. The run reports:

- `object_file_id_bind_time_leakage_audit_pass`: `1.000`
- `object_file_id_bind_time_candidate_filter_usage`: `0.000`
- `object_file_id_auxiliary_label_usage`: `1.000`

The patch then adds two diagnostic Probe 2 paths without adding a new training loss:

- full-state prediction-error binding: compare the actual position-aware reappeared query state to each active file's expected position-aware state, then pick the file with lowest error.
- local prediction-error binding: suppress the rest of the reappeared image around each candidate file's predicted region, re-run the Definition/file-query read on that local percept, then pick the file with lowest error against that file's expected state.

Best held-out checkpoint:

- dynamics position error / improvement: `0.234` / `+0.162`
- predicted-position exact / hard match: `0.158` / `0.158`
- feature-only exact / hard match: `0.079` / `0.079`
- feature-only position-ablated exact / hard match: `0.079` / `0.079`
- active state-prediction-error exact / hard match: `0.079` / `0.079`
- active local-prediction-error exact / hard match: `0.237` / `0.237`
- feature-only local-prediction-error exact / hard match: `0.088` / `0.088`
- learned active target recall: `0.882`
- Definition position R2: `-0.395`
- file-query position R2: `-0.397`
- occluded bridge: `0.000`
- ternary nonzero fraction: `0.248`

Latest held-out checkpoint:

- dynamics position error / improvement: `0.232` / `+0.165`
- predicted-position exact / hard match: `0.158` / `0.158`
- feature-only exact / hard match: `0.079` / `0.079`
- feature-only position-ablated exact / hard match: `0.079` / `0.079`
- active state-prediction-error exact / hard match: `0.079` / `0.079`
- active local-prediction-error exact / hard match: `0.158` / `0.158`
- feature-only local-prediction-error exact / hard match: `0.158` / `0.158`
- learned active target recall: `0.878`
- Definition position R2: `-0.390`
- file-query position R2: `-0.398`
- occluded bridge: `0.000`
- ternary nonzero fraction: `0.227`

This is a scoped partial result. The audit clears the immediate `object_file_id` bind-time leakage concern. Learned dynamics still improves reappearance position over naive projection, and the best checkpoint shows local prediction-error binding can beat feature-only in the contested stream. But full-state prediction error collapses to the feature-only floor, local prediction error is unstable across checkpoints, position recoverability remains negative in the contested two-object representation, and the occluded Definition bridge is absent. The current mechanism does not solve same-instance reappearance binding. The next target is a true slot-aware/local object representation that preserves object-specific geometry before the prediction-error competition, not more similarity pressure or governance.

## Object-Local Slot Recoverability Gate

Run: `runs/20260601_214941_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

This patch adds object-local slots as continuous percept carriers, then reads sparse ternary Definition state from each slot-local image. Slot assignment is salience/locality based and uses no `object_id` or `object_file_id`. Ground-truth target/distractor positions are used only after slot discovery to score recoverability.

Best held-out checkpoint:

- object_file_id bind-time audit pass / candidate filter usage: `1.000` / `0.000`
- slot count / valid fraction / used count: `2.000` / `1.000` / `2.000`
- slot occupancy entropy: `0.999`
- slot separation / collapse fraction: `0.457` / `0.000`
- target position error / recall: `0.009` / `1.000`
- distractor position error / recall: `0.009` / `1.000`
- pair position error: `0.009`
- slot position R2 / improvement: `1.000` / `+0.252`
- slot assignment object_file_id/object_id usage: `0.000` / `0.000`
- slot ternary nonzero fraction / axis usage: `0.135` / `5.000`
- scene-level Definition position R2: `-0.318`
- scene-level file-query position R2: `-0.317`
- predicted-position exact match: `0.315`
- feature-only exact match: `0.079`
- active local prediction-error exact match: `0.000`
- occluded bridge: `0.000`

Latest held-out checkpoint:

- object_file_id bind-time audit pass / candidate filter usage: `1.000` / `0.000`
- slot count / valid fraction / used count: `2.000` / `1.000` / `2.000`
- slot occupancy entropy: `0.999`
- slot separation / collapse fraction: `0.457` / `0.000`
- target position error / recall: `0.009` / `1.000`
- distractor position error / recall: `0.009` / `1.000`
- pair position error: `0.009`
- slot position R2 / improvement: `1.000` / `+0.252`
- slot assignment object_file_id/object_id usage: `0.000` / `0.000`
- slot ternary nonzero fraction / axis usage: `0.119` / `6.000`
- scene-level Definition position R2: `-0.311`
- scene-level file-query position R2: `-0.308`
- predicted-position exact match: `0.298`
- feature-only exact match: `0.079`
- active local prediction-error exact match: `0.000`
- occluded bridge: `0.000`

This clears the slot recoverability gate. The contested frame can now be decomposed into two object-local percept carriers without label assignment, and the slot carriers recover both target and distractor geometry with essentially perfect held-out R2. The sparse ternary slot readout is live but not dense. This is not yet an object-permanence win: object-file binding and the occluded Definition bridge are still unsolved. The next step is to make object files compete against slots, not whole-scene query states: each live file predicts an expected slot state/position, each visible slot supplies local evidence, and binding is decided by lowest joint file-to-slot prediction error.

## File-to-Slot Trajectory Binding Probe

Run: `runs/20260602_064540_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

Before binding, the contested generator was audited for trajectory separability at the reappearance frame. Across train/test/held-out variants, the two true reappearance positions stay `12.806` px apart. The naive wrapped velocity projection remains inaccurate, with mean projection error around `8.289` to `10.470` px, so the learned dynamics path is still the relevant predictor.

This patch adds a file-to-slot assignment probe. Object slots still recover the two visible objects without identity labels. Each object file supplies a learned-dynamics predicted reappearance position. Assignment solves the minimum joint position-error match between candidate files and recovered slots. `object_file_id`, `object_id`, and `sequence_id` are used only after assignment for scoring/audit; assignment usage remains `0.000 / 0.000 / 0.000`.

Best held-out checkpoint:

- slot pair position error / slot R2: `0.009` / `1.000`
- slot ternary nonzero fraction: `0.247`
- dynamics position error / improvement / valid fraction: `0.234` / `+0.162` / `1.000`
- global file-slot target / hard / distractor / pair: `0.071` / `0.071` / `0.071` / `0.000`
- global candidates / target recall / assignment error: `12.750` / `0.150` / `0.199`
- predicted-position file-slot target / hard / distractor / pair: `0.119` / `0.119` / `0.109` / `0.030`
- predicted-position candidates / target recall / assignment error: `7.632` / `0.247` / `0.222`
- feature-only file-slot target / hard / distractor / pair: `0.071` / `0.071` / `0.071` / `0.000`
- learned-active file-slot target / hard / distractor / pair: `0.110` / `0.110` / `0.110` / `0.000`
- occluded bridge: `0.000`

Latest held-out checkpoint:

- slot pair position error / slot R2: `0.009` / `1.000`
- slot ternary nonzero fraction: `0.263`
- dynamics position error / improvement / valid fraction: `0.232` / `+0.165` / `1.000`
- global file-slot target / hard / distractor / pair: `0.071` / `0.071` / `0.071` / `0.000`
- global candidates / target recall / assignment error: `12.750` / `0.150` / `0.198`
- predicted-position file-slot target / hard / distractor / pair: `0.119` / `0.119` / `0.109` / `0.030`
- predicted-position candidates / target recall / assignment error: `7.632` / `0.247` / `0.220`
- feature-only file-slot target / hard / distractor / pair: `0.071` / `0.071` / `0.071` / `0.000`
- learned-active file-slot target / hard / distractor / pair: `0.119` / `0.119` / `0.119` / `0.000`
- occluded bridge: `0.000`

This is a clean negative/partial result. The generator is separable and the slot interface is no longer the bottleneck. Predicted-position gating improves over the global/feature-only floor, but only weakly: target recall reaches about `0.247`, exact/hard binding only `0.119`, pair binding only `0.030`, and the occluded Definition bridge remains absent. The failure has moved from visible object localization to active-file identity continuity. The position signal is present, but current dynamics precision is too low relative to same-class object spacing. The next discriminator is an oracle-position ceiling test: if perfect endpoints bind cleanly, sharpen dynamics; if they do not, fix assignment logic.

## Oracle-Position File-to-Slot Ceiling

Run evaluated: `runs/20260602_064540_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

This diagnostic uses the true reappearance endpoint as if it were the file's predicted endpoint, then runs the same position-error assignment against recovered slots. Identity labels are still not used by the assignment. Three variants are reported:

- oracle local ceiling: per row, only the target endpoint and same-class distractor endpoint are provided as the live file set.
- oracle global: all batch file endpoints are available, exposing duplicate-position/global-candidate ambiguity.
- oracle mask: true endpoints plus the existing oracle-shaped candidate mask.

Best held-out checkpoint:

- slot pair position error / slot R2: `0.009` / `1.000`
- dynamics position error / improvement / valid fraction: `0.234` / `+0.162` / `1.000`
- oracle local ceiling target / hard / distractor / pair: `1.000` / `1.000` / `1.000` / `1.000`
- oracle local candidates / target recall / distractor recall / assignment error: `2.000` / `1.000` / `1.000` / `0.009`
- oracle global target / hard / distractor / pair: `0.446` / `0.446` / `0.446` / `0.262`
- oracle mask target / hard / distractor / pair: `0.446` / `0.446` / `0.315` / `0.131`
- predicted-position target / hard / distractor / pair: `0.119` / `0.119` / `0.109` / `0.030`
- assignment audit object_file_id/object_id/sequence usage: `0.000` / `0.000` / `0.000`

Latest held-out checkpoint:

- slot pair position error / slot R2: `0.009` / `1.000`
- dynamics position error / improvement / valid fraction: `0.232` / `+0.165` / `1.000`
- oracle local ceiling target / hard / distractor / pair: `1.000` / `1.000` / `1.000` / `1.000`
- oracle local candidates / target recall / distractor recall / assignment error: `2.000` / `1.000` / `1.000` / `0.009`
- oracle global target / hard / distractor / pair: `0.446` / `0.446` / `0.446` / `0.262`
- oracle mask target / hard / distractor / pair: `0.446` / `0.446` / `0.315` / `0.131`
- predicted-position target / hard / distractor / pair: `0.119` / `0.119` / `0.109` / `0.030`
- assignment audit object_file_id/object_id/sequence usage: `0.000` / `0.000` / `0.000`

This separates mechanism failure from dynamics-precision failure. With perfect local endpoints, file-to-slot assignment works completely. With learned dynamics endpoints, it fails because the endpoint estimate is still too blurry: normalized error around `0.232-0.234` is about `6.5` px on a `28x28` frame, roughly half the true `12.806` px inter-object spacing. The next target is not richer appearance or context; it is a sharper trajectory endpoint predictor and/or a live-file candidate state that preserves the local two-file set without identity-label help.

## Oracle Noise Sweep And Ballistic Baseline

Run evaluated: `runs/20260602_064540_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

This diagnostic adds two measurements without retraining:

- oracle endpoint noise sweep: start from true target/distractor endpoints, inject deterministic pixel noise, then bind those noised endpoints to recovered slots by the same label-free file-to-slot position assignment.
- hand ballistic baseline: predict reappearance as `last_visible_position + velocity * phase_elapsed`, using the same wrap geometry as the active-file projector.

Best held-out checkpoint:

- learned dynamics error / improvement / valid fraction: `0.234` / `+0.162` / `1.000`
- ballistic error / dynamics-over-ballistic improvement / valid fraction: `0.248` / `+0.014` / `1.000`
- predicted-position slot target / hard / pair: `0.119` / `0.119` / `0.030`
- ballistic slot target / hard / pair / assignment error: `0.158` / `0.158` / `0.000` / `0.209`
- oracle ceiling target / hard / pair: `1.000` / `1.000` / `1.000`
- oracle noise 0 px target / pair / assignment error: `1.000` / `1.000` / `0.009`
- oracle noise 1 px target / pair / assignment error: `1.000` / `1.000` / `0.036`
- oracle noise 2 px target / pair / assignment error: `1.000` / `1.000` / `0.072`
- oracle noise 3 px target / pair / assignment error: `1.000` / `1.000` / `0.107`
- oracle noise 4 px target / pair / assignment error: `1.000` / `1.000` / `0.143`
- oracle noise 6 px target / pair / assignment error: `1.000` / `1.000` / `0.214`
- oracle noise 7 px target / pair / assignment error: `0.817` / `0.817` / `0.243`
- oracle noise 8 px target / pair / assignment error: `0.725` / `0.725` / `0.264`

Latest held-out checkpoint:

- learned dynamics error / improvement / valid fraction: `0.232` / `+0.165` / `1.000`
- ballistic error / dynamics-over-ballistic improvement / valid fraction: `0.248` / `+0.016` / `1.000`
- predicted-position slot target / hard / pair: `0.119` / `0.119` / `0.030`
- ballistic slot target / hard / pair / assignment error: `0.158` / `0.158` / `0.000` / `0.209`
- oracle ceiling target / hard / pair: `1.000` / `1.000` / `1.000`
- oracle noise 0 px target / pair / assignment error: `1.000` / `1.000` / `0.009`
- oracle noise 1 px target / pair / assignment error: `1.000` / `1.000` / `0.036`
- oracle noise 2 px target / pair / assignment error: `1.000` / `1.000` / `0.072`
- oracle noise 3 px target / pair / assignment error: `1.000` / `1.000` / `0.107`
- oracle noise 4 px target / pair / assignment error: `1.000` / `1.000` / `0.143`
- oracle noise 6 px target / pair / assignment error: `1.000` / `1.000` / `0.214`
- oracle noise 7 px target / pair / assignment error: `0.817` / `0.817` / `0.243`
- oracle noise 8 px target / pair / assignment error: `0.725` / `0.725` / `0.264`

The local two-file assignment mechanism tolerates endpoint noise through about `6 px` and starts degrading at `7 px`. The learned dynamics head sits at the edge of that budget: `0.232-0.234` normalized is about `6.5 px`. The hand ballistic baseline is worse at about `6.95 px`, and learned dynamics only improves it by `0.014-0.016` normalized, roughly `0.4 px`. The immediate target is therefore precise endpoint dynamics below the `6 px` cliff, plus preserving the relevant local live-file set. Appearance/context/gain routing remain unnecessary for this specific controlled failure.

## Dynamics Error Shape And Local Pair Binding

Run evaluated: `runs/20260602_064540_temporal_objects_contested_position`

Config: `configs/temporal_objects_contested_position.yaml`

This diagnostic separates endpoint error shape from global candidate-set failure. It reports:

- local learned-dynamics file-to-slot assignment using only the target file and same-class distractor file for that row.
- paired endpoint structure: true pair distance, predicted pair distance, midpoint pull, bias, error percentiles, and paired error correlation.
- oracle error-shape injections at `6.5 px`: center-biased compression, correlated shared shift, and heavy-tail adversarial rows.

Best held-out checkpoint:

- global predicted-position target / hard / pair: `0.119` / `0.119` / `0.030`
- local learned-dynamics target / hard / distractor / pair: `0.747` / `0.747` / `0.747` / `0.747`
- local ballistic target / hard / pair: `1.000` / `1.000` / `1.000`
- true pair distance: `0.457` normalized, `12.806 px`
- predicted pair distance / ratio / compression: `0.440` / `0.962` / `0.017`
- midpoint error / midpoint pull: `0.123` / `-0.007`
- learned endpoint error mean / median / p75 / p90 / p95 / max: `0.234` / `0.243` / `0.292` / `0.429` / `0.491` / `0.501`
- endpoint bias norm: `0.024`
- paired error cosine / x-corr / y-corr: `-0.456` / `-0.555` / `-0.502`
- center-biased oracle target / pair / ratio: `0.000` / `0.000` / `0.015`
- correlated oracle target / pair / ratio: `1.000` / `1.000` / `1.000`
- heavy-tail oracle target / pair / ratio: `0.737` / `0.737` / `1.543`

Latest held-out checkpoint:

- global predicted-position target / hard / pair: `0.119` / `0.119` / `0.030`
- local learned-dynamics target / hard / distractor / pair: `0.747` / `0.747` / `0.747` / `0.747`
- local ballistic target / hard / pair: `1.000` / `1.000` / `1.000`
- true pair distance: `0.457` normalized, `12.806 px`
- predicted pair distance / ratio / compression: `0.438` / `0.957` / `0.020`
- midpoint error / midpoint pull: `0.123` / `-0.007`
- learned endpoint error mean / median / p75 / p90 / p95 / max: `0.232` / `0.242` / `0.289` / `0.427` / `0.490` / `0.501`
- endpoint bias norm: `0.027`
- paired error cosine / x-corr / y-corr: `-0.448` / `-0.558` / `-0.506`
- center-biased oracle target / pair / ratio: `0.000` / `0.000` / `0.015`
- correlated oracle target / pair / ratio: `1.000` / `1.000` / `1.000`
- heavy-tail oracle target / pair / ratio: `0.737` / `0.737` / `1.543`

This corrects the prior interpretation. The learned head is not mainly compressing both tracks into the midpoint: predicted pair distance is only mildly compressed, midpoint pull is slightly negative, and correlated shared-shift noise does not hurt assignment. The local learned-dynamics binding result is also much better than the global predicted-position result, so `0.119` is mostly a live-candidate/global-selection failure, not purely an endpoint metrology failure. The learned endpoint error is heavy-tailed and anti-correlated across paired files; the synthetic heavy-tail injection reproduces the local learned-dynamics result closely (`0.737` vs `0.747`). The hand ballistic endpoint has worse mean error but perfect local assignment, which means the learned residual is bending some cases in identity-damaging ways.

The narrower claim is now: local contested file-to-slot binding works when the endpoint trajectory model is correct. On the current contested-position stream, that endpoint model is partly hand-supplied by the simple motion rule. The generator's X coordinate is linear in phase, while Y has only a small stepped drift, so constant-velocity extrapolation is too close to a closed-form solution for held-out local binding. This is a valid architecture milestone for slots and assignment, but not proof that TSM has learned a nontrivial trajectory model.

## Curved Contested Motion Gate

Config added: `configs/temporal_objects_contested_curved.yaml`

Dataset added: `temporal_objects_contested_curved`

This stream preserves the earned substrate:

- two same-class object tracks per scene.
- two object-local visible slots.
- no object identity labels in file-to-slot assignment.
- oracle endpoint, ballistic endpoint, learned dynamics endpoint, local file-to-slot, and global retrieval diagnostics remain side by side.

The motion rule changes. Instead of phase-linear X with stepped Y, each track follows deterministic phase offsets that make the last visible `phase 1 -> 2` velocity a poor predictor of the hidden `phase 2 -> 0` continuation. The reappearance endpoints remain separable, but the hand ballistic endpoint is no longer an oracle.

Unit-level generator gate:

- curved true reappearance tracks remain separated by more than `8 px`.
- curved local ballistic pair assignment is below `0.75` across train/test/held-out splits.
- oracle endpoint assignment should still be `1.000` once the model exposes clean slots.

Next experiment:

- train `configs/temporal_objects_contested_curved.yaml`.
- compare oracle endpoint vs ballistic endpoint vs learned dynamics endpoint on local file-to-slot binding.
- call the next milestone only if learned/residual dynamics beats ballistic on the curved stream while oracle remains perfect and slot recovery remains clean.

## Curved Contested Motion Result

Run: `runs/20260602_115937_temporal_objects_contested_curved`

Config: `configs/temporal_objects_contested_curved.yaml`

Held-out/test evaluation confirms the intended discriminator:

- nonlinear contested motion fraction: `1.000`
- slot pair position error: `0.009`
- slot position R2: `0.999995`
- oracle local target / pair: `1.000` / `1.000`
- ballistic local target / pair: `0.505` / `0.505`
- learned dynamics local target / pair: `1.000` / `1.000`
- ballistic endpoint error: `0.416`
- learned dynamics endpoint error: `0.163`
- learned-over-ballistic endpoint improvement: `+0.253`
- learned endpoint p95 / max: `0.296` / `0.298`
- global learned file-slot target / pair: `0.281` / `0.000`

This is the missing trajectory discriminator. The curved generator breaks the hand ballistic endpoint as a local assignment key while preserving clean slots and perfect oracle assignment. After training, the learned dynamics endpoint recovers perfect local same-class file-to-slot binding and sharply improves endpoint error over ballistic.

Narrow claim:

TSM can learn a nontrivial deterministic object-file trajectory model that supports contested local file-to-slot binding across occlusion.

Boundary:

This still does not solve global file retrieval. The global learned file-slot path remains weak (`0.281` target, `0.000` pair), so the remaining problem is live candidate-set / memory-index control, not local slots, local assignment, or trajectory dynamics.

## Curved Two-Object Seed Sweep

Run: `runs/20260602_124456_ternary_seed_sweep`

Config: `configs/temporal_objects_contested_curved.yaml`

Seeds: `31, 37, 43`

Held-out/test summary:

| seed | oracle all-set | ballistic all-set | learned all-set | learned object | slot R2 | slot pair err | ballistic err | learned err | global pair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | 1.000 | 0.505 | 1.000 | 1.000 | 0.999995 | 0.009 | 0.416 | 0.155 | 0.000 |
| 37 | 1.000 | 0.505 | 1.000 | 1.000 | 0.999995 | 0.009 | 0.416 | 0.165 | 0.000 |
| 43 | 1.000 | 0.505 | 1.000 | 1.000 | 0.999995 | 0.009 | 0.416 | 0.153 | 0.000 |

This validates the two-object curved result across seeds. The learned dynamics endpoint repeatedly beats the ballistic prior and solves the local same-class file-to-slot assignment. The global pair metric remains `0.000` across seeds, so the local permanence substrate and the global memory-index problem are now clearly separated.

## Curved Object-Count Sweep

Run summary: `runs/curved_object_count_sweep_summary.json`

Configs:

- `configs/temporal_objects_contested_curved.yaml`
- `configs/temporal_objects_contested_curved_3.yaml`
- `configs/temporal_objects_contested_curved_4.yaml`

Held-out/test summary:

| objects | oracle all-set | ballistic all-set | ballistic object | learned all-set | learned object | slot R2 | slot pair err | ballistic err | learned err | global pair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.505 | 0.505 | 1.000 | 1.000 | 0.999995 | 0.009 | 0.416 | 0.163 | 0.000 |
| 3 | 1.000 | 0.273 | 0.424 | 1.000 | 1.000 | 0.998054 | 0.015 | 0.424 | 0.131 | 0.091 |
| 4 | 1.000 | 0.000 | 0.186 | 0.258 | 0.629 | 0.998663 | 0.016 | 0.432 | 0.207 | 0.000 |

This answers the first scaling question. Clean slots and oracle assignment survive through four same-class objects, so the perceptual slot layer and assignment audit are not the bottleneck. Learned trajectory binding scales from two to three objects under the curved generator. At four objects it becomes load-fragile: learned dynamics still beats ballistic in endpoint error and per-object assignment, but full local set binding falls to `0.258`.

Current interpretation:

- Two-object curved local permanence: repeatably validated.
- Three-object curved local permanence: initial pass.
- Four-object curved local permanence: partial; candidate-count/load fragility appears before global retrieval is fixed.
- Global memory retrieval remains unsolved at every count.

Next target:

Improve binding under load before claiming general local permanence. The likely next mechanism is not broader cognition; it is a local live-set/assignment improvement that preserves the validated slot and trajectory dynamics while handling four competing same-class files. Candidate-set control and assignment should be evaluated together because the four-object failure resembles a scaled version of the global retrieval problem.

## Curved Spacing-Vs-Error Diagnostic

Runs evaluated:

- `runs/20260602_115937_temporal_objects_contested_curved`
- `runs/20260602_125952_temporal_objects_contested_curved_3`
- `runs/20260602_130402_temporal_objects_contested_curved_4`

This diagnostic adds all-track endpoint spacing metrics:

- minimum inter-object reappearance spacing.
- learned endpoint error mean / median / p90.
- ballistic endpoint error mean / median / p90.
- slot localization error.
- oracle, ballistic, and learned all-set binding.

Held-out/test summary:

| objects | min spacing px | learned mean | learned median | learned p90 | learned p90 / spacing | ballistic mean | ballistic p90 | slot err | oracle all-set | learned all-set |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 14.142 | 0.163 | 0.179 | 0.291 | 0.576 | 0.416 | 0.662 | 0.009 | 1.000 | 1.000 |
| 3 | 8.694 | 0.131 | 0.112 | 0.231 | 0.745 | 0.424 | 0.602 | 0.015 | 1.000 | 1.000 |
| 4 | 8.623 | 0.207 | 0.191 | 0.388 | 1.259 | 0.432 | 0.668 | 0.016 | 1.000 | 0.258 |

Interpretation:

The four-object failure is not a slot failure and not an assignment bug: slot localization remains tight, slot R2 remains near `0.999`, and oracle all-set assignment stays `1.000`. It is also not just "global retrieval" as a separate subsystem. The four-object local problem enters the same candidate-density regime: minimum spacing is about `8.6 px`, while learned p90 endpoint error is about `0.388` normalized, or `10.9 px`, which exceeds the local spacing budget. In the two- and three-object cases, learned p90 error remains below the spacing budget and local binding succeeds.

Current fork:

- If the frame is kept at `28x28`, the next target is endpoint precision / load robustness for four same-class tracks.
- If the goal is to test candidate-count logic independently of density, add a wider-frame or wider-spacing four-object variant first.
- Global `0.000` is now plausibly the far end of this same density/candidate-count curve, not necessarily a separate failure mode.

## Wide Four-Object Curved Control

Config added: `configs/temporal_objects_contested_curved_4_wide.yaml`

Dataset added: `temporal_objects_contested_curved_4_wide`

Diagnostic added: all-track endpoint errors split by slot cleanliness:

- slot-clean object fraction.
- slot-dirty object fraction.
- clean-slot endpoint error mean / p90.
- dirty-slot endpoint error mean / p90.
- high-error object fraction.
- high-error clean fraction.

Pre-registered fork:

- If wider spacing recovers learned four-object binding, the previous failure was mostly arena density / spacing budget.
- If wider spacing does not recover binding while slots are clean, the failure is load-induced trajectory degradation in the file/dynamics path.
- If wider spacing corrupts slots, the control is confounded and the slot geometry must be repaired before interpreting binding.

Calibration note:

An initial wide run with `object_slot_nms_radius: 5.0` was confounded: visible slots duplicated around objects and slot recoverability dropped. Direct slot auditing showed that `6.0` restores distinct object-local slots for the `40x40` stream. The diagnostic result below uses the repaired `6.0` NMS radius.

Run: `runs/20260602_135616_temporal_objects_contested_curved_4_wide`

Config: `configs/temporal_objects_contested_curved_4_wide.yaml`

Held-out/test summary:

| condition | min spacing px | learned mean | learned median | learned p90 | learned p90 / spacing | ballistic mean | ballistic p90 | slot err | slot R2 | slot clean | oracle all-set | ballistic all-set | learned all-set | learned object |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-object 28px | 8.623 | 0.207 | 0.191 | 0.388 | 1.259 | 0.432 | 0.668 | 0.016 | 0.999 | n/a | 1.000 | 0.000 | 0.258 | 0.629 |
| 4-object wide 40px | 13.831 | 0.263 | 0.248 | 0.406 | 1.174 | 0.485 | 0.721 | 0.015 | 0.9999 | 1.000 | 1.000 | 0.000 | 0.167 | 0.500 |

Slot-clean endpoint split for the corrected wide run:

- slot clean object fraction: `1.000`
- slot dirty object fraction: `0.000`
- slot error mean / p90: `0.0146` / `0.0181`
- clean-slot endpoint mean / p90: `0.263` / `0.406`
- dirty-slot endpoint mean / p90: `0.000` / `0.000`
- high-error object fraction: `0.250`
- high-error clean fraction: `1.000`

Interpretation:

Wider spacing did not recover four-object learned binding. The control is clean: slot localization is strong, slot collapse is `0.000`, oracle all-set assignment remains `1.000`, and all high endpoint-error cases occur on clean slots. That rules out arena spacing alone and slot contamination for the corrected wide condition.

The four-object wall is now better described as load-induced trajectory degradation under four same-class tracks. The learned dynamics endpoint stays better than ballistic, but its p90 error remains above the spacing budget even after the arena is widened, and full local set binding remains low.

Current claim:

TSM has validated curved local contested binding for two and three same-class objects. Four-object binding fails because the per-object dynamics/file trajectory state degrades under load, not because the slot layer or assignment logic cannot represent the scene.

Next target:

Clean the slot-to-file trajectory pipeline under four-object load. Do not add broader cognition, governance, or appearance/context routing until the dynamics state can keep four clean object trajectories separated.

## Probe A: Neutral Decline-To-Bind

Diagnostic added:

- `reappeared_dynamics_neutral_all_file_slot_*`
- `reappeared_ballistic_neutral_all_file_slot_*`

This is not a new binding mechanism and it is not Definition splitting. Forced-choice file-to-slot metrics remain unchanged. Probe A asks whether a file's nearest slot is genuinely separable from its second-nearest slot under the current endpoint error band.

Rule:

```text
margin = second_nearest_slot_distance - nearest_slot_distance
uncertainty_proxy = measured endpoint error

if margin <= uncertainty_proxy:
    decline / neutral
else:
    confident bind
```

The uncertainty proxy is measured endpoint error, so this is an evaluation probe, not a deployable runtime confidence head. It tests whether the failures are inside an honest deadband.

Buckets:

- confident correct bind
- confident wrong bind
- correct decline: neutral prevented a forced wrong bind
- wrong decline: neutral withheld a forced correct bind

Audits:

- assignment object-file-id usage: `0.000`
- assignment object-id usage: `0.000`
- old forced-choice metrics remain side by side

Held-out/test summary:

| condition | forced correct | forced wrong | neutral decline | confident correct | confident wrong | correct decline | wrong decline | decline precision | learned all-set |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-object curved | 1.000 | 0.000 | 0.374 | 0.626 | 0.000 | 0.000 | 0.374 | 0.000 | 1.000 |
| 3-object curved | 0.833 | 0.167 | 0.333 | 0.667 | 0.000 | 0.167 | 0.167 | 0.500 | 1.000 |
| 4-object curved | 0.500 | 0.500 | 0.625 | 0.375 | 0.000 | 0.500 | 0.125 | 0.792 | 0.250 |
| 4-object wide | 0.396 | 0.604 | 0.813 | 0.188 | 0.000 | 0.604 | 0.208 | 0.749 | 0.167 |

Interpretation:

The four-object failures are mostly inside a real neutral band. In the 4-object curved condition, all forced wrong decisions are converted into correct declines under the endpoint-error deadband, with no confident-wrong bucket. The wide 4-object condition shows the same pattern: forced wrong binding is high, but those wrong binds mostly become correct declines when the system is allowed to say "not enough resolution."

This supports the narrow TSM claim:

```text
Neutral is not null.
Neutral can act as an appraisal state meaning:
current measurement cannot resolve this distinction.
```

Caveat:

The probe is conservative. It also declines some forced-correct decisions:

- 2-object curved wrong-decline: `0.374`
- 3-object curved wrong-decline: `0.167`
- 4-object curved wrong-decline: `0.125`
- 4-object wide wrong-decline: `0.208`

So the neutral band is real, but its current threshold is not calibrated. It is useful as an appraisal diagnostic before it is useful as a runtime binding policy.

## Shared-Trajectory Load Diagnostic

Endpoint metrics now also report:

- shared first-two-track endpoint mean / p90.
- extra-track endpoint mean / p90.
- per-track endpoint means for tracks 0-3.

Tracks 0 and 1 follow the same curved trajectory family across the 2-, 3-, and 4-object generators. This checks whether the same trajectory gets worse as simultaneous object count rises.

Held-out/test summary:

| condition | shared mean | shared p90 | extra mean | extra p90 | track0 mean | track1 mean | track2 mean | track3 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-object curved | 0.163 | 0.291 | 0.000 | 0.000 | 0.188 | 0.138 | 0.000 | 0.000 |
| 3-object curved | 0.125 | 0.234 | 0.146 | 0.231 | 0.116 | 0.134 | 0.146 | 0.000 |
| 4-object curved | 0.179 | 0.354 | 0.235 | 0.410 | 0.186 | 0.172 | 0.225 | 0.246 |
| 4-object wide | 0.293 | 0.462 | 0.232 | 0.406 | 0.265 | 0.321 | 0.227 | 0.238 |

Interpretation:

The same shared trajectories do get worse under four-object load. The 3-object case does not degrade the shared tracks, but the 4-object case does, and the wide 4-object run still degrades despite clean slots and wider spacing. That supports the load-induced trajectory degradation diagnosis over a pure spacing or slot-contamination diagnosis.

Current fork:

- Probe A passes as an appraisal diagnostic: four-object wrong binds mostly become correct declines.
- The next runtime step should be calibrated uncertainty / neutral binding, not immediate hard Definition splitting.
- Definition splitting or new distinction creation becomes appropriate after the model can expose an internal uncertainty estimate rather than using measured endpoint error as the probe's oracle-side proxy.

## Probe A-prime Stage 1: Runtime Confidence Calibration

Diagnostic added:

- `reappeared_dynamics_runtime_confidence_*`

This is still not a runtime decline policy. It tests whether runtime-available signals can predict actual endpoint error before using a neutral bind decision. The confidence path uses:

- predicted dynamics endpoint.
- visible slot geometry and slot salience.
- object-file confidence and age.
- optional learned-vs-ballistic endpoint disagreement.

Scoring uses true future positions and file labels only after the runtime signals are computed. The confidence-path audit reports zero usage for:

- true future position.
- measured endpoint error.
- object id.
- object-file id.
- sequence id.

Held-out/test summary:

| condition | set bind | Probe A correct decline | runtime uncertainty/error Pearson | runtime uncertainty/error Spearman | naive margin/error Pearson | runtime confidence mean | confidence drop on correct declines |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-object curved | 1.000 | 0.000 | 0.933 | 0.911 | 0.925 | 0.735 | n/a |
| 3-object curved | 1.000 | 0.167 | 0.749 | 0.757 | 0.845 | 0.729 | 0.057 |
| 4-object curved | 0.250 | 0.500 | 0.087 | 0.130 | 0.142 | 0.704 | 0.019 |
| 4-object wide | 0.167 | 0.604 | -0.091 | -0.163 | 0.007 | 0.695 | -0.010 |

All confidence-path leakage audit metrics are `0.000` in the four evaluated conditions.

Interpretation:

Stage 1 is a partial negative result. Runtime geometry/confidence signals correlate with actual endpoint error in the easier two- and three-object conditions, and mean confidence drops mildly as object count rises. But the signal fails exactly where it is needed: under four-object load, the internal uncertainty estimate barely correlates with actual endpoint error, and in the wide four-object condition it is slightly anticorrelated. The naive margin-only baseline fails there too.

The key failure is not that neutral appraisal is wrong. Probe A still shows that many four-object forced errors are correct declines under an oracle-side endpoint-error band. The failure is that the current runtime signals do not know when they are inside that band.

Current fork:

- Do not implement Stage 2 runtime decline from this confidence score.
- Do not implement Definition splitting yet.
- The next useful patch should create a better internal uncertainty source for the trajectory endpoint itself, such as a learned variance/error head, ensemble/dropout disagreement, or per-slot trajectory residual calibration, then rerun Probe A-prime before using neutral as a policy.

## Probe A-prime Stage 1b: Learned Calibration Head

Diagnostic/configs added:

- `active_file_calibration` loss.
- `active_file_calibration_weight`.
- `active_file_calibration_detach_inputs`.
- `reappeared_dynamics_runtime_confidence_calibrated_*` metrics.
- `configs/temporal_objects_calibration_curved*.yaml`.

The calibration head is a separate runtime judgment head over the trajectory measurement. It receives detached runtime-only features by default:

- object-file trajectory/dynamics features.
- predicted reappearance endpoint.
- slot candidate geometry.
- slot salience/entropy.
- object load.
- learned-vs-ballistic endpoint disagreement.

It is trained against normalized actual endpoint error as supervision. The actual endpoint error is not used as a runtime input. Leakage audits for the confidence path remain `0.000`.

Runs:

- `runs/20260602_195742_temporal_objects_calibration_curved`
- `runs/20260602_195831_temporal_objects_calibration_curved_3`
- `runs/20260602_195635_temporal_objects_calibration_curved_4`
- `runs/20260602_195929_temporal_objects_calibration_curved_4_wide`

Held-out/test summary:

| condition | set bind | runtime Pearson | naive Pearson | calibrated Pearson | calibrated Spearman | calibrated confidence mean | calibrated confidence drop on correct declines | calibrated high-error lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-object curved | 1.000 | 0.933 | 0.947 | 0.874 | 0.845 | 0.821 | n/a | 0.015 |
| 3-object curved | 1.000 | 0.802 | 0.911 | 0.721 | 0.778 | 0.862 | 0.008 | 0.009 |
| 4-object curved | 0.250 | 0.118 | 0.166 | 0.338 | 0.477 | 0.831 | 0.004 | 0.003 |
| 4-object wide | 0.333 | 0.105 | 0.164 | 0.252 | 0.305 | 0.815 | 0.004 | 0.003 |

Interpretation:

This is a partial result, not a runtime-neutral win. The learned calibration head does recover some signal at the four-object wall: Pearson improves from about `0.118` to `0.338` on 4-object curved and from about `0.105` to `0.252` on wide-4. That means a separate judgment head is more informative than the prior handcrafted runtime confidence under load.

But the head is not yet calibrated enough to license Stage 2. Confidence stays high at the wall, the confidence drop on Probe-A correct-decline cases is only about `0.004`, and calibrated uncertainty barely separates high-error from low-error cases. The head can weakly rank failure, but it does not yet "feel maybe" strongly enough to drive a decline-to-bind policy.

Current fork:

- Do not implement runtime decline-to-bind from this head yet.
- Do not implement Definition splitting yet.
- The next calibration patch should make uncertainty sharper, not broader: train a residual/variance target with stronger high-error weighting, a pairwise ranking loss over endpoint errors, or an ensemble/dropout disagreement diagnostic. The success condition is not just correlation; it is visible confidence drop and positive high-error lift at 4/wide-4.

## Probe A-prime Stage 1c: Tail Danger Calibration

Diagnostic/code added:

- `endpoint_error_to_spacing_ratio_*` runtime-confidence metrics.
- `unsafe_endpoint_error_fraction`.
- AUROC/AUPRC for unsafe endpoint-error cases.
- Unsafe mean/lift metrics for runtime, calibrated, naive-margin, and absolute candidate-margin scores.
- Error-bucket means for low/mid/high actual endpoint-error quantiles.
- Optional `active_file_calibration_tail_weight` / `active_file_calibration_tail_ratio_threshold` objective.
- Tail configs:
  - `configs/temporal_objects_tail_calibration_curved_4.yaml`
  - `configs/temporal_objects_tail_calibration_curved_4_wide.yaml`

Unsafe is defined as:

```text
normalized_endpoint_error / nearest_interobject_spacing >= 0.5
```

That is the nearest-position flip-risk boundary. This is a scoring/training target only. True positions and endpoint errors are not runtime inputs to the confidence head, and confidence leakage audits remain `0.000`.

Evaluation outputs:

- `runs/20260602_195742_temporal_objects_calibration_curved/tail_eval_best.json`
- `runs/20260602_195831_temporal_objects_calibration_curved_3/tail_eval_best.json`
- `runs/20260602_195635_temporal_objects_calibration_curved_4/tail_eval_best.json`
- `runs/20260602_195929_temporal_objects_calibration_curved_4_wide/tail_eval_best.json`
- `runs/20260602_204436_temporal_objects_tail_calibration_curved_4/tail_eval_latest.json`
- `runs/20260602_204529_temporal_objects_tail_calibration_curved_4_wide/tail_eval_latest.json`

Baseline calibration held-out/test danger summary:

| condition | unsafe frac | p90 error/spacing | runtime AUROC/AUPRC | calibrated AUROC/AUPRC | naive-margin AUROC/AUPRC | candidate-margin AUROC/AUPRC | calibrated unsafe lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-object curved | 0.250 | 0.546 | 1.000 / 1.000 | 0.923 / 0.821 | 1.000 / 1.000 | 0.923 / 0.821 | 0.0159 |
| 3-object curved | 0.333 | 0.653 | 0.938 / 0.878 | 0.969 / 0.946 | 0.969 / 0.946 | 0.938 / 0.912 | 0.0119 |
| 4-object curved | 0.500 | 1.134 | 0.695 / 0.789 | 0.746 / 0.768 | 0.659 / 0.763 | 0.627 / 0.750 | 0.0072 |
| 4-object wide | 0.729 | 1.196 | 0.847 / 0.938 | 0.812 / 0.882 | 0.875 / 0.940 | 0.881 / 0.943 | 0.0111 |

Tail-weighted continuation summary:

| condition | checkpoint | unsafe frac | p90 error/spacing | calibrated Pearson/Spearman | calibrated AUROC/AUPRC | calibrated unsafe lift | calibrated low/mid/high bucket |
|---|---|---:|---:|---:|---:|---:|---:|
| 4-object curved | baseline best | 0.500 | 1.134 | 0.338 / 0.477 | 0.746 / 0.768 | 0.0072 | 0.197 / 0.206 / 0.206 |
| 4-object curved | tail latest | 0.375 | 1.022 | 0.500 / 0.606 | 0.835 / 0.802 | 0.0205 | 0.370 / 0.389 / 0.397 |
| 4-object wide | baseline best | 0.729 | 1.196 | 0.252 / 0.305 | 0.812 / 0.882 | 0.0111 | 0.224 / 0.230 / 0.230 |
| 4-object wide | tail latest | 0.583 | 1.237 | 0.381 / 0.436 | 0.786 / 0.817 | 0.0007 | 0.503 / 0.504 / 0.504 |

Interpretation:

This is a sharper yellow light, not a green light for runtime neutral. The baseline calibration head is better judged by unsafe ranking than by Pearson alone. On 4-object curved it beats the handcrafted runtime score on AUROC, but only weakly separates unsafe from safe cases by magnitude. On wide-4, the handcrafted and candidate-margin baselines still beat the calibrated head on danger ranking.

The tail-risk objective improves the regular 4-object wall: calibrated AUROC rises from `0.746` to `0.835`, Spearman from `0.477` to `0.606`, and unsafe lift from `0.007` to `0.020`. But the wide-4 continuation is a negative control: the head becomes broadly high-risk and nearly flat across error buckets, with unsafe lift dropping to `0.001`. That means the tail objective can increase danger ranking in one wall case, but it is not a reliable runtime appraisal mechanism yet.

Current fork:

- Do not implement Stage 2 runtime decline-to-bind from this head.
- Do not implement Definition splitting yet.
- Keep the tail-risk objective as an optional diagnostic/training control, not the default claim.
- Next useful calibration source is not a bigger decline policy. It is richer uncertainty evidence in the file state: residual history, distributional/variance dynamics, ensemble/dropout disagreement, or per-slot trajectory residual memory.

## Probe A-prime Stage 1d: Relation-Local Calibration Audit

Diagnostic/code added:

- file-level unsafe ranking remains `endpoint_error / nearest_interobject_spacing >= 0.5`.
- slot-level unsafe ranking reports slot localization danger with the same spacing threshold.
- pair-level unsafe ranking reports whether the file-slot decision is below its nearest-competitor margin.
- within-scene AUROC/AUPRC for file-level and pair-level unsafe detection.
- per-scene calibration variance.
- safe-vs-unsafe uncertainty gap inside each scene.
- scene-adjusted calibrated pair risk:

```text
pair_risk_adjusted = pair_risk - scene_mean(pair_risk)
```

Relation eval outputs:

- `runs/20260602_195635_temporal_objects_calibration_curved_4/relation_eval_best.json`
- `runs/20260602_204436_temporal_objects_tail_calibration_curved_4/relation_eval_latest.json`
- `runs/20260602_195929_temporal_objects_calibration_curved_4_wide/relation_eval_best.json`
- `runs/20260602_204529_temporal_objects_tail_calibration_curved_4_wide/relation_eval_latest.json`

Held-out/test summary:

| condition | file unsafe | pair unsafe | slot unsafe | pair-valid scenes | calibrated pair AUROC/AUPRC | calibrated within-scene pair AUROC/AUPRC | scene-adjusted pair AUROC/AUPRC | calibrated within-scene variance | calibrated within-scene pair gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4-object baseline | 0.500 | 0.688 | 0.000 | 0.750 | 0.850 / 0.924 | 0.806 / 0.917 | 0.779 / 0.907 | 0.000058 | 0.0089 |
| 4-object tail | 0.375 | 0.563 | 0.000 | 0.750 | 0.862 / 0.904 | 0.889 / 0.833 | 0.775 / 0.844 | 0.000287 | 0.0222 |
| 4-wide baseline | 0.729 | 0.771 | 0.000 | 0.667 | 0.836 / 0.895 | 0.917 / 0.938 | 0.820 / 0.875 | 0.000071 | 0.0106 |
| 4-wide tail | 0.583 | 0.729 | 0.000 | 0.750 | 0.913 / 0.957 | 0.917 / 0.955 | 0.905 / 0.951 | 0.000000 | 0.0010 |

Baselines for within-scene pair ranking:

| condition | runtime AUROC/AUPRC | naive-margin AUROC/AUPRC | candidate-margin AUROC/AUPRC |
|---|---:|---:|---:|
| 4-object baseline | 0.681 / 0.861 | 0.806 / 0.917 | 0.806 / 0.917 |
| 4-object tail | 0.500 / 0.667 | 0.583 / 0.694 | 0.583 / 0.694 |
| 4-wide baseline | 0.771 / 0.920 | 0.833 / 0.924 | 0.854 / 0.938 |
| 4-wide tail | 0.833 / 0.934 | 0.938 / 0.969 | 0.938 / 0.969 |

Interpretation:

This audit blocks the simplest "scene-global danger mood" interpretation, but it still does not clear Stage 2. The calibrated head can often rank unsafe pair relations inside the same scene. In wide-4 tail, pair-level AUROC/AUPRC is high globally and within-scene (`0.913 / 0.957` global, `0.917 / 0.955` within-scene). That means the head is not merely saying "this whole world is hard." It contains relation-local ordering signal.

The problem is magnitude. Wide-4 tail has almost zero calibrated within-scene variance and only `0.001` unsafe gap. The head can preserve a tiny ordering, but it does not separate unsafe from safe pairs with enough amplitude to drive a reliable decline-to-bind policy. In regular 4-object tail, the amplitude improves (`0.022` gap), but the scene-adjusted AUROC drops relative to raw pair ranking, so the signal is still fragile.

Slot unsafe stays `0.000` in all four audits. That confirms the wall is not slot localization. The failure remains calibration of file-slot relation danger under load.

Current fork:

- Runtime neutral remains blocked.
- Definition splitting remains blocked.
- The next useful target is not stronger binary tail loss by itself. The system needs object-local uncertainty state with amplitude, not just rank:
  - residual history per object file,
  - predicted endpoint variance/distribution,
  - ensemble or dropout disagreement,
  - per-slot trajectory residual memory,
  - calibration normalized within active candidate sets.
