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
14. Predicted reappearance position is load-bearing for explicit candidate masking: no.
15. Position channels are load-bearing inside feature-only binding: yes, under ablation control.
16. Binding representations preserve recoverable position: partial. Visible reappeared Definition/file-query binding features now preserve position; memory-conditioned source features remain weak.
17. Same-class contested two-object continuity: failed initial test.
18. Object-file expectation predicts its own future Definition state: partial, still weak.
19. Full exact object permanence: not yet.

## Current Claim

TSM now has object-file continuity signal that survives occlusion and distinguishes same-instance identity above chance in the original single-target stream. The active candidate scaffold can preserve the correct file in the live lookup set and improve constrained reappearance lookup. A learned active gate can recover part of that live set from Definition/file geometry and context without position input, but it is still weaker than the scaffold. Object files now carry scaffolded phase/trajectory state, and a learned dynamics head can predict reappearance position much better than the naive velocity projection. A position-aware binding interface exposes reappeared position in the visible Definition/file-query representations for the single-target stream. Position ablation shows that feature-only binding uses those coordinate channels there. The contested two-object stream breaks that partial win: visible binding position is no longer target-specific enough when two same-class objects are present. Full object permanence is still not solved because visible reappeared state does not bind cleanly back to the exact object file under contested global lookup.

## Next Target

The next mechanism should be Probe 2: prediction-error competition over full expected reappearance state. Each active object file should predict an expected reappearance state, the actual percept should arrive, and the file with the lowest local prediction error should own the reappearance. Do not add another expectation loss, governance layer, or broad similarity head before testing this ownership mechanism.

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
