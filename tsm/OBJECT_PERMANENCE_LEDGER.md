# TSM Object-Permanence Ledger

This ledger records the current experimental status of the temporal object-continuity work. It is intentionally scoped to the synthetic temporal-object stream and should not be read as full object permanence.

## Current Status

1. Memory carries hidden object identity: yes.
2. Memory helps Reality prediction during occlusion: yes.
3. Memory shapes Definition state during occlusion: yes.
4. Object-file carries same-instance signal: yes.
5. Query path helps hard same-class file discrimination: yes.
6. Visible reappearance globally binds to exact file: no.
7. Reappearance binding preserves occlusion bridge: not yet.
8. Active candidate gating can constrain reappearance lookup: initial pass.

## Current Claim

TSM now has object-file continuity signal that survives occlusion and distinguishes same-instance identity above chance, especially against hard same-class negatives. Full object permanence is still not solved because the visible reappeared Definition state does not bind cleanly back to the exact object file under global lookup.

## Next Target

The next mechanism should restrict reappearance lookup to active object-file candidates rather than increasing broad contrastive pressure. A reappearing visible state should query a live candidate set shaped by context, recency, position, and expected phase before it competes against the full object-file field.

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
