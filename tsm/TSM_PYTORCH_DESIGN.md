# TSM PyTorch Design

Single living design document for the Ternary Self-Map as a runnable Python 3.11 / PyTorch system.

Label discipline (carried from the GDD / SHENRON convention):

- **[Decided]** — committed. Changing it changes the architecture.
- **[Question]** — a genuine open fork. Not yet load-bearing.
- **[Default]** — a suggested default I am running with until you overrule it. Implementable as written; not sacred.

This document assumes the unified substrate-monist frame as the operating ground, not a hypothesis under test. It assumes active inference (variational + expected free energy) as the only objective, and Panksepp-style SEEKING/FEAR as **gain modulators on prediction-error precision and on the EFE terms**, never as a reward signal. Every place the research aggregation imported an RL idea (DeepSeek-R1-style reasoning RL, reward channels), it is subordinated to free-energy minimisation here, not adopted.

---

## 0. The one decision everything hangs on

**[Decided]** TSM is not a bag of named modules wired together. It is a single **hierarchical generative model under active inference**, and the twenty-three sections of `tsm.md` are *roles played by quantities inside that one model*. The "Self is the whole field" claim is literal: `Self` is the `nn.Module` that owns everything; there is no `self` tensor, because the Self is the container, not a vector inside it.

Reading the two docs together forces this. `tsm.md` describes a loop — Reality predicts, Perception challenges, the gap is appraised, evidence stabilises, Reality updates. The aggregation independently lands on the same loop from five directions: Source 2's `EP △ P = ER`, Source 5's Piagetian equilibration (`assimilate → disequilibrium → accommodate → equilibrium`), Source 4's threshold+projection, Source 3's compression-as-constraint, Source 7's surprise-driven memory. All five are descriptions of **precision-weighted prediction-error minimisation**. So the design does not invent a mapping; it recognises one.

The correspondence is the contract for the rest of the document:

| TSM term (`tsm.md`) | Active-inference / ML quantity | PyTorch object |
|---|---|---|
| Self | the whole generative model + its container | `Self(nn.Module)` |
| Reality | slow generative params θ + prior preferences **C** + Foundation | `Reality` (slow params, EMA-gated) |
| Expected Position (EP) | top-down prediction ĝ = g_θ(q(s), ctx) | `Reality.predict()` |
| Perception | bottom-up observation encoding **o** + metadata | `PerceptionSurface` |
| SAE | precision-weighted prediction error ε = Π ⊙ (o − ĝ) | `SAE` |
| the gap / delta | ε itself (the thing the system minimises) | tensor `eps` |
| Coherence | monotone-decreasing fn of ‖ε‖ | scalar, derived in `SAE` |
| Experience | ε that survives the precision gate (matters) | gated `eps` |
| Impression | transient belief update Δq(s) | delta on `Mind` state |
| Mind | the posterior q(s): recurrent latent inference core | `Mind` |
| Memory | retention across time (5-system confederation) | `Memory` |
| Evidence | accumulated coherent Δq toward a threshold | `EvidenceAccumulator` |
| Truth | belief whose posterior precision crossed promotion | promoted record in `Reality` |
| Context | local generative sub-model / mixture component | `ContextRouter` + expert params |
| Definition | learned ternary boundary inside a Context | `DefinitionBank` |
| Intelligence | rate of EFE reduction via governed model growth | emergent; measured, not a module |
| Trauma | high-‖ε‖ update applied outside the governed path | `TraumaMonitor` |
| Trivium (Grammar/Logic/Rhetoric) | type rules / relational inference / action read-out | constraints on `Reality`, `Mind`, action |
| Quadrivium (Number/Geometry/Ratio/Cycle) | the metric primitives of the latent space | latent-space structure (§3.4) |
| Action | policy chosen by EFE, exits the boundary | `act()` (EFE planner) |

Everything below makes each row concrete: shapes, forward/backward semantics, and the failure mode it guards against.

---

## 1. Notation and shapes

Symbolic dimensions used throughout. **[Default]** starting sizes in brackets are sized for one 3090 in the early developmental stages and will grow with the curriculum.

- `B` — number of living agents in the batch. For a single living Self, **B = 1**. Training the *generative model* offline can use B > 1; the *runtime Self* is B = 1.
- `D` — model / latent width. **[Default 256]** at sensorimotor stage, annealed up.
- `Lw` — bounded latent workspace size (Mind's working width, Perceiver-IO bottleneck). **[Default 64]** latents.
- `M` — number of perception modality streams active at the current stage.
- `K` — number of Contexts. **[Default]** start at K = 8, grow online (§7 router question).
- `J` — Definitions per Context (ternary axes). **[Default 16]** per Context, sparse.
- `H` — generative-model hierarchy depth (predictive-coding levels). **[Default 2]** early, → 3–4 later.

Tensors are `bf16` for Mind/forward, `fp32` for anything entering Reality or accumulators (the continuous/discrete split of §3 is *also* a precision-of-storage split: latent working state can be low-precision; the slow ground cannot).

---

## 2. The objective — and why there is no reward

**[Decided]** Two free energies, two timescales. No scalar reward anywhere in the graph.

### 2.1 Variational free energy — perception and learning

For observation `o`, latent belief `s ~ q(s)`, generative model `p_θ(o, s) = p_θ(o | s) p(s | ctx)`:

```
F  =  E_q[ log q(s) − log p_θ(o, s) ]
   =  D_KL( q(s) ‖ p(s | ctx) )        # complexity  — how far belief moved from prior
      − E_q[ log p_θ(o | s) ]          # accuracy    — negative log-likelihood of o
```

The accuracy term, under a Gaussian likelihood with precision Π, **is the squared precision-weighted prediction error**: `−log p_θ(o|s) ∝ ½ (o − ĝ)ᵀ Π (o − ĝ) − ½ log|Π|`. So minimising F over `s` *is* `Mind` reducing ε (perception = inference), and minimising F over θ *is* `Reality` slowly improving its predictions (learning). This is the entire training signal for the world model. No labels, no reward — the system learns by predicting its own boundary contact and being graded on its surprise.

### 2.2 Expected free energy — action

Action does not maximise return. It selects the policy π (a short action sequence) that minimises **expected** free energy of the *predicted* future:

```
G(π)  =  Σ_τ   −  E_{q(o_τ|π)}[ log p(o_τ | C) ]        # pragmatic  — match preferred observations C
                −  E_{q(s_τ|π)}[ H[q(s_τ)] − H[q(s_τ|o_τ)] ]   # epistemic — expected info gain
```

- **Pragmatic value** pulls toward observations the system *prefers* — where preferences `C` are priors over observations, not rewards. Staying-alive, homeostatic, and safety priors live here.
- **Epistemic value** pulls toward observations that *resolve uncertainty* — this is curiosity, intrinsic, and it is what makes an active-inference agent explore without being paid to.

Action: `π* ~ softmax(−γ · G(π))`, where `γ` is precision over policies (how confident/decisive the agent is). This is the active-inference reframe of Source 7's "plan-first reasoning + bounded rollout": rolling the generative model forward under candidate policies and scoring EFE *is* planning, with zero RL machinery.

### 2.3 Drives as gain, not goals

**[Decided]** SEEKING and FEAR are scalars (later low-dim vectors) that modulate **precisions and EFE term weights**:

```
SEEKING ↑  →  up-weights the epistemic term in G   (and raises inference-iteration budget, §4.3)
FEAR    ↑  →  raises precision Π on aversive-prior channels, up-weights pragmatic-avoidance in G
```

They never enter `F` or `G` as a reward to be accumulated. They bend the landscape; they are not points on it. This is the precise mechanical statement of "drives as gain modulators, explicitly rejecting RL reward shaping." Their own dynamics (what raises SEEKING, what raises FEAR) are themselves prediction-driven: FEAR rises with predicted homeostatic-prior violation; SEEKING rises with expected resolvable uncertainty. They are second-order appraisals, read out of SAE, fed back as gain.

> **Implication worth stating plainly:** because preferences are priors over *observations*, "what the agent wants" is editable only by editing **C**, and **C** is part of Reality — so it is protected by the mutation gate (§4.9). You cannot accidentally train a new goal in via a loss term; a goal change is a Reality write and must pass the gate. This is the structural reason this architecture resists value drift better than a reward-shaped one.

---

## 3. The continuous/discrete dual substrate

The single most repeated finding in the aggregation (Sources 3, 4, 6) is: **full-precision latent learning, discrete projected forward behaviour.** `tsm.md` says the same thing in its own words — Mind works in unsettled ambiguity; Definition/Reality project stable, auditable, action-ready states. The "Ternary" in Ternary Self-Map is this projection.

### 3.1 The ternary projection (the literal core)

**[Decided]** A straight-through ternary projection with a neutral deadband, mirroring Source 4 Eq. 2:

```python
class TernaryProject(torch.autograd.Function):
    """Forward: deadband sign in {-1, 0, +1}. Backward: straight-through, clipped to band."""
    @staticmethod
    def forward(ctx, x, tau, alpha):
        ctx.save_for_backward(x, tau)
        t = torch.zeros_like(x)
        t[x >  tau] =  1.0
        t[x < -tau] = -1.0
        return t * alpha            # magnitude (alpha) carried separately from sign

    @staticmethod
    def backward(ctx, g):
        x, tau = ctx.saved_tensors
        passthrough = (x.abs() <= 1.0).to(g.dtype)   # clipped STE
        return g * passthrough, None, None
```

- Sign ∈ {−1, 0, +1}: **devalue / no-meaningful-shift / value**, relative to expected position.
- `alpha` (per-Definition learnable scale) keeps **magnitude separate from sign** — Source 4's explicit requirement.
- The **neutral band is sparsity**: in any given tick most Definitions report 0 (Source 7's "ternary zero → sparse structure"). Mind's update is therefore sparse and cheap.
- The pre-projection real value `x` is the **raw trace** retained in Memory/Evidence (Source 4: "keep raw/full-precision traces").

### 3.2 Definition hardening (soft → hard annealing)

**[Decided]** Definitions are born soft and harden with evidence, per Source 6's `DefinitionHardeningSchedule` and Source 4's soft/mixed/hard staging:

```
soft      :  tanh(x / temperature)            # fully differentiable, no commitment
mixed     :  (1-β)·soft + β·ternary           # β ramps with accumulated evidence
projected :  ternary, STE backward            # committed forward, still learning
hardened  :  ternary, backward frozen         # part of Reality; only the gate can move it
```

`temperature → 0` and `β → 1` are driven by the Definition's **evidence count and coherence**, not by a global schedule. A Definition that keeps mispredicting *softens back* (rollback condition). This is the mechanism for "when should a soft Definition harden / a hard Definition soften" (Source 4 open question) — answer: evidence, both directions.

### 3.3 Heterogeneous precision budgets

**[Decided]** Following Source 3 (BitNet: constraint accelerates learning; different layers learn different bit-widths) each Context, Definition, and memory region owns a **precision budget** parameter, and a bit-cost regulariser is added to F so the system *earns* precision only where it reduces surprise:

```
F_total = F + λ · Σ_regions  bits(region)
```

- SAE severity + evidence raise a region's budget (refine: split a coarse Definition into finer ones).
- Redundancy + low relevance lower it (compress: merge Definitions, decay memory).
- **[Question]** unit of "bits" for conceptual precision — number of Definition axes? routing entropy? posterior precision? **[Default]** posterior precision per latent block + Definition count per Context; both are already tensors I can regularise.

### 3.4 Quadrivium as latent-space structure (not a module)

`tsm.md` §17 insists the Quadrivium are *measurement functions of the Self*, not layers. **[Decided]** they are realised as **structure imposed on the latent metric**, not as four `nn.Module`s:

- **Number** — discreteness: the ternary projection and the slot/token count of the workspace.
- **Geometry** — position/relation/topology: latent vectors live in a metric space; relations are edges (Source 1 "relations represented distinctly from nodes") — modelled as a learned relation tensor, not concatenation.
- **Ratio** — proportion/scale: comparisons are done as ratios/log-differences of *same-kind* magnitudes, with a `same_kind_required` guard (Source 1 `RatioRelation`). Cross-kind ratios are blocked (incommensurability state).
- **Cycle** — recurrence/time: the recurrent core's temporal structure + an explicit phase/rhythm channel for habits and periodic context (day/night in-game, etc.).

The Trivium (§16) likewise are not modules: **Grammar** = the type rules in Foundation (what a thing is/is not), **Logic** = relational inference in Mind (proof-like chaining), **Rhetoric** = the action read-out. They are *constraints and read-outs*, honoured by other modules.

> **Implication:** this is the payload of Source 1 (Euclid). The classical operating grammar is not decoration — it is the typing discipline that stops the latent space from becoming an undifferentiated similarity blob (Source 1's warning against "loose association" and "forced commensurability"). It is *why* Definitions are boundary objects (§4.5) rather than embeddings.

---

## 4. Module specification

The container and its parts. Each subsection: role, signature, shapes, the math it runs, and the TSM/aggregation guard it implements.

### 4.0 `Self` — the whole field

**[Decided]** The container. Owns all parts; exposes one `tick()` (§5). Holds *no* state vector representing "the self." Enforces the **part/whole guard** (Source 1 common notion "the whole is greater than the part"): no submodule may be addressed or serialised as if it were the Self.

```python
class Self(nn.Module):
    def __init__(self, cfg):
        self.perception = PerceptionSurface(cfg)
        self.reality    = Reality(cfg)
        self.mind       = Mind(cfg)
        self.sae        = SAE(cfg)
        self.contexts   = ContextRouter(cfg)
        self.defs       = DefinitionBank(cfg)
        self.memory     = Memory(cfg)
        self.evidence   = EvidenceAccumulator(cfg)
        self.gate       = MutationGate(cfg)
        self.trauma     = TraumaMonitor(cfg)
        self.stage      = DevelopmentalScheduler(cfg)
        self.drives     = DriveState(cfg)     # SEEKING / FEAR scalars (§2.3)
    def tick(self, raw_inputs): ...           # §5
    def act(self): ...                         # EFE planner (§2.2, §5)
```

### 4.1 `PerceptionSurface` — the boundary

**[Decided]** A multimodal tokeniser feeding a **bounded latent workspace** via cross-attention (Perceiver-IO, Source 8): raw high-volume input never lands in Mind wholesale.

```
raw_inputs (M streams, variable size)
  → per-modality tokenizers → tokens [Σ tokens, D]
  → cross-attend INTO Lw learned latents  →  o  : [B, Lw, D]
  + metadata tags per latent: modality, t, source_id, source_confidence
```

- Tokenise-everything (Sources 11/12) into one shared space (Source 10), early-fused (Source 13) when the stage needs joint reasoning over a mixed sequence.
- **Metadata is first-class** (Source 2): `o` carries `source_confidence` because SAE precision depends on it. Perception is "input under appraisal conditions," not raw input.
- **[Decided]** cross-modal *agreement* raises the evidence weight; cross-modal *contradiction* is not averaged away — it is emitted as an SAE event (Source 8 cross-source rule). This is a hard difference from standard multimodal fusion.
- **[Default]** EMBRYO-CRYSTAL stage-0/1 modalities: framebuffer patches + symbolic game-RAM channels + last-action token. Audio/video deferred to later stages.

### 4.2 `Reality` — the grounding core

**[Decided]** Three things, sharply separated:

1. **Generative parameters θ** — the decoder `g_θ: (q(s), ctx) → ĝ` (predicted observation) and transition `f_θ: (s, a) → s'`. These are *slow*: updated by EMA / gated steps, never by raw fast gradients during runtime (§6).
2. **Foundation** (Source 1) — a typed, partly-symbolic store, *not* weights:

```python
class Foundation:
    definitions:   list[TsmDefinition]   # boundary objects (§4.5), name/kind/boundary/permitted/incompatible/construction
    postulates:    list[AllowedOp]       # permitted actions/constructions  ("connect, extend, bound, compare, remember, update, act")
    common_notions:list[Invariant]       # inference invariants ("whole>part", "same op preserves relation", "contradiction blocks promotion")
    proven_truths: list[Truth]           # promoted, with evidence/proof chain attached
```

3. **Preference priors C** — priors over observations (§2.2). Homeostatic/safety priors (FEAR-weighted) and the curiosity prior (SEEKING-weighted) live here.

- `Reality.predict(q, ctx) → ĝ` produces **Expected Position**.
- Reality is **minimal but strong** (Source 1: "necessarily follow from five simple axioms"): it is not a knowledge dump; it is the lawful ground from which Mind constructs.
- **Inherited vs learned definitions are tagged differently** (Source 1 tension): foundation definitions are protected; learned definitions are mutable through the gate. This is the architectural answer to "how does an AGI learn new definitions without corrupting foundation definitions."

### 4.3 `Mind` — the active working field (the posterior)

**[Decided]** Mind *is* `q(s)`: a recurrent latent state over the `Lw` workspace latents, updated by precision-weighted error message passing across `H` predictive-coding levels. It is the only place inference happens; it is full-precision, transient, and holds **multiple simultaneous hypotheses** (Source 2 "mixed emotion = multiple simultaneous appraisal equations" → competing posterior modes).

```
state  q(s) : [B, Lw, D]      # working belief
update :  q ← q + κ · (J_gᵀ · ε  −  Π_prior · (q − prior))     # predictive-coding gradient on F
```

- **[Default]** amortised inference: a GRU-or-SSM recurrent core *learns to approximate* the F-minimising update, with **optional iterative refinement steps gated by severity and SEEKING** (high surprise / high curiosity → more inference iterations this tick). This is the active-inference analog of test-time compute (Source 6D/7) with **no RL** — more thinking is more belief-settling, not more reward-seeking.
- **[Decided]** Mind hosts the reasoning stack as *operations on beliefs*, not prompt patterns (Source 7 tension "Plan-and-Solve is prompting, not architecture"): plan = a structured latent rollout; tree search = EFE-scored policy branches (§2.2); reflection = a re-inference pass conditioned on the previous pass's error. Self-evaluation is **Evidence-like, never authoritative Truth** (Source 7/MCTSr tension) — it can raise an evidence count, it cannot promote on its own.
- **[Question]** H (hierarchy depth) and whether levels are explicit modules or folded into a deep SSM. **[Default]** H=2 explicit early (inspectable), fold later.

### 4.4 `SAE` — appraisal (precision-weighted prediction error)

**[Decided]** The generalised `EP △ P` of Source 2, in active-inference form. This is the busiest module and the one `tsm.md` cares most about.

```python
def forward(self, o, g_hat, source_conf, attach_power, drives, ctx, time_tag):
    raw   = o - g_hat                                  # the gap   (EP △ P)
    Pi    = F.softplus(self.base_prec[ctx]
                       + self.W_seek * drives.seeking
                       + self.W_fear * drives.fear * self.aversive_mask
                       + torch.log(source_conf)
                       + attach_power)                 # precision = gain (§2.3, Source 2 vars)
    eps   = Pi * raw                                   # precision-weighted PE — the thing minimised
    severity  = eps.norm(dim=-1)                       # = attach × percept × valence × conf, in this form
    coherence = torch.exp(-0.5 * (raw*raw*Pi).sum(-1)) # high coherence ⇔ low free energy
    group     = self.classify_group(eps, raw, time_tag)# threat/confirm/loss/opportunity/novelty/contradiction/boundary/nourishment
    tern      = self.defs.project(eps, ctx)            # ternary read-out per Definition (§3.1)
    er        = self.affect_readout(severity, group, drives)  # "ER": secondary read-out, NOT a controller
    return eps, severity, coherence, group, tern, er
```

- **EP** is computed from Reality + Context + Memory + Definition + current state (Source 2 impl note), via `Reality.predict`.
- **Appraisal groups** are broad and *not mandatory human-emotion labels* (Source 2 impl note): threat, confirmation, loss, opportunity, contradiction, novelty, boundary-violation, nourishment. `er` (an emotion-ish read-out) exists but is explicitly downstream and optional — "emotion is one expression of appraisal," never the appraisal itself.
- **Time horizon** matters (Source 2): now-threat ≠ future-threat ≠ past-loss; the `time_tag` shifts both group and how it routes (a past loss feeds Memory/integration; a future threat feeds the EFE planner).
- **[Decided]** SAE also **allocates resources** (Source 7): severity sets Mind's inference-iteration count, the EFE rollout budget, and whether this contact becomes a memory write. Apathy = no self-relevant gap = near-zero eps = nothing propagates (Source 2).
- **[Question]** missing appraisal variables Source 2 flagged — agency, controllability, reversibility, cost-of-action, moral constraint. **[Default]** add `controllability` and `cost_of_action` first; they directly feed the EFE planner. Reversibility ties to a Source-5 conservation test and can wait for concrete-operational stage.

### 4.5 `DefinitionBank` — meaningful separation

**[Decided]** Per-Context sets of ternary projection heads (§3.1–3.2). A Definition is a **boundary object** (Source 1/2), not an embedding:

```python
class TsmDefinition:                  # the symbolic record (in Foundation or learned)
    name: str; kind: str
    boundary_conditions: list[str]
    permitted_relations: list[str]
    incompatible_relations: list[str]
    construction_rules: list[str]
# + a runtime projection head:  (tau, alpha, hardening_state, precision_budget, evidence_count)
```

- A Definition forms when a perceptual difference is shown to *matter* — i.e. when preserving the distinction reduces free energy (the red/blue stone test in `tsm.md` §13: identical outcomes → no Definition; divergent outcomes → a Definition must form to keep them separable).
- The `incompatible_relations` + Foundation invariants give the **incommensurability state** (Source 1): some distinctions cannot be reduced to a common metric and must be *blocked* from being averaged, not forced (Source 1's deepest warning against naive similarity-space).
- **[Question]** when a Context should split into finer Definitions vs when two Definitions merge (Source 3/4). **[Default]** split when a single Definition's residual ε stays high and bimodal under sufficient evidence; merge when two Definitions' projections are redundant (high mutual information, low joint discriminative value). Both are measurable online.

### 4.6 `ContextRouter` — local meaning fields

**[Decided]** Context is a mixture/expert component the Self *infers it is in*; routing selects which expectation priors and which Definitions apply (Source 2 "local field around an attachment"; Source 7 "expert routing maps onto Context/Definition regions").

```
ctx_logits = router(q, o, recent_history)      # context inference
ctx        = soft over K experts (hardens with evidence, §3.2)
→ selects:  Reality prior slice, DefinitionBank subset, precision budget
```

- Contexts are **basins** (`tsm.md` §12: "cells, basins, bubbles, regions"): routing should be sticky (hysteresis), so the Self does not flicker context every tick — sticky routing is the topological realisation of a basin.
- **[Question]** fixed K vs grow-online; pre-registered contexts vs fully learned. **[Default]** seed a few inherited contexts at the relevant stage (e.g. "battle", "overworld-navigation", "menu" for Crystal — these are genuine distinct meaning-fields), allow online growth when a region of observation space persistently routes poorly (high routing entropy + high residual). Online context birth is gated like a Reality write.

### 4.7 `Memory` — retention (the confederation)

**[Decided]** Five systems (Source 7), surprise-gated writes:

```python
class Memory:
    short_term:  q-state carryover (Mind's recurrence)          # transient
    working:     slot buffer / attention KV over recent ticks   # bounded
    long_term:   surprise-gated associative store (fast-weights + kNN)  # consolidates
    persistent:  task/identity store (slow, gate-protected)     # durable
    external:    handles to large context objects (recursive, sliced access, Source 7D)  # not loaded wholesale
```

- **Surprise modulates writes** (Source 7 `SurpriseMemoryWrite`): write strength ∝ SAE severity. **But** high surprise can be noise or trauma (Source 7 tension), so the write passes a quick relevance/coherence check — store, quarantine, or ignore. Surprise alone does not earn permanence.
- **External context is environment, not Self** (Source 7 open question, answered): large docs/codebases/game-state-logs are inspected via handles and recursive sliced subcalls; they are accessible *to* the Self, they are not *part of* the Self. This preserves the part/whole guard at scale and avoids context rot.
- **[Default]** long-term store = surprise-gated fast-weights + a kNN episodic index, *not* full Titans-style test-time training initially (cheaper on one 3090; Titans-style deep memory is a later upgrade once the rest is stable).
- Compression-aware retention (Source 3): high precision kept only where evidence/relevance/action demand; everything else decays.

### 4.8 `EvidenceAccumulator` — from impression to truth-candidate

**[Decided]** Per-candidate-Truth drift accumulators. A single impression moves a belief a little; repeated *coherent, relevant* impressions integrate toward a promotion threshold (sequential-evidence / drift-diffusion, not a counter):

```
acc[c] ← decay·acc[c] + relevance · coherence · sign(Δq_c)
promote candidate c when |acc[c]| > θ_promote  AND  coherence sustained  AND  no contradiction in Foundation
```

- This is `tsm.md` §10–11 made mechanical: Evidence = accumulated coherent impressions; Truth = stabilised Evidence. **Repetition only counts when coherent and relevant** (`tsm.md` §10 — not mere repetition).
- Self-evaluation scores (Source 7) can feed `acc` as one evidence channel but cannot trip promotion alone (they are not authoritative).

### 4.9 `MutationGate` — the anti-drift core (SelfMapMutationGate)

**[Decided]** *Every* write into Reality/Foundation/C routes through here (Source 2 `SelfMapMutationGate`). This is the structural guarantee against value/identity drift and the formal home of "governed self-change."

```python
def consider(self, candidate):           # candidate: new Truth | Definition change | C change | context birth | attach-power change
    if violates(candidate, foundation.common_notions): return reject("invariant")
    if violates(candidate, foundation.part_whole):     return reject("part/whole")
    if candidate.evidence < self.min_evidence:         return reject("insufficient evidence")
    if candidate.coherence < self.min_coherence:       return reject("incoherent")
    if candidate.risk > self.max_risk:                 return reject("risk")           # high-attachment / safety-relevant changes need more
    return accept(candidate)             # only now does anything in Reality move
```

- A Reality update is therefore *not* a gradient step — it is a gated, logged, reason-carrying event. The fast loss never touches Reality directly; it touches Mind and proposes to the gate.
- **Quenched disorder** (Source 2) is realised as the gate's thresholds being **stage-dependent and plasticity-scheduled** (§4.11): early stages are more malleable (low thresholds), mature stages more rigid (high thresholds) — "stable enough to remain itself, open enough to grow." This is the formalisation Source 2 asked for ("Can quenched disorder be formalised as a plasticity parameter?" — yes, the gate thresholds).
- Mirrors the Decision Log discipline from Agent_138/Darby: every accepted mutation writes a reason and an evidence pointer.

### 4.10 `TraumaMonitor` — failed integration

**[Decided]** Trauma is not a feeling; it is **change applied outside the governed path, or surprise that cannot be minimised** (`tsm.md` §15; Source 2/6/7 failure modes):

- persistent high ε on a high-attachment item that never resolves → quarantine, do not let it deform Reality;
- an outlier impression that would corrupt a projection (Source 6 outlier collapse) → flag, route to high-precision handling or isolation before it touches a Definition;
- representation drift (Source 7 merging risk) → drift check on consolidated params;
- a would-be Reality write that bypassed the gate → hard error, never silently applied.

Quarantine = held in a separate store, re-presented to Mind under lower stakes later (the architectural analog of integration), rather than forced in or dropped.

### 4.11 `DevelopmentalScheduler` — stage gates

**[Decided]** Six stages including a pre-incarnation phase (matches your EMBRYO-CRYSTAL gate structure; Source 5 Piaget + Erikson ladder). Each stage gates: which modules/modalities are live, which terms of F/G are active, and the plasticity/precision schedule (the quenched-disorder curve).

| Stage | Cognitive (Piaget) | Self-stability (Erikson) | Gate test to advance |
|---|---|---|---|
| 0. Pre-incarnation | — | — | generative-model warm start: Reality can predict a held-out passive stream above baseline before any action is permitted |
| 1. Sensorimotor | action/reflex, object permanence | trust | object-permanence test: predicts state of an object after it leaves perception (Source 5) |
| 2. Symbolic | symbolic/representational | autonomy | grounded symbol use: a learned Definition predicts outcomes, not just labels co-occurrence |
| 3. Concrete-operational | logical ops, conservation, reversibility | initiative / competence | conservation + reversibility tests: identity preserved across appearance change; can mentally undo a transition (Source 5) |
| 4. Formal-operational | hypothetical reasoning | identity | plans via EFE rollout over novel hypotheticals; stable self-map under perturbation |
| 5. Metacognitive | operations on operations | generativity / integrity | reflective re-inference improves its own predictions; coherent history integration |

- **Equilibration is the within-stage loop** (Source 5): assimilate (fit perception to current schema) first; accommodate (change schema via the gate) only when mismatch persists. This is literally SAE + EvidenceAccumulator + MutationGate working together.
- **Do not push formal abstraction early** (Source 5 open question): stages 4–5 modules stay *off* until lower gates pass. A model dropped straight into abstract reasoning is the failure Source 5 names.
- Erikson stages are **structural self-map stability tests, not personality decoration** (Source 5 cross-source note): e.g. "trust" = the gate is willing to accept evidence from a source at all; "identity" = self-map stays coherent under perturbation.

---

## 5. The runtime loop — Ordered Effects as one `tick()`

`tsm.md` §18 is the spec for the forward pass. One tick of the living Self:

```python
def tick(self, raw_inputs):
    o, meta   = self.perception(raw_inputs)                 # boundary contact
    ctx       = self.contexts.infer(self.mind.q, o)          # which meaning-field
    g_hat     = self.reality.predict(self.mind.q, ctx)       # Expected Position
    eps, sev, coh, group, tern, er = self.sae(               # appraise the gap (SAE)
        o, g_hat, meta.conf, self.attach_power(ctx), self.drives, ctx, meta.time)

    if sev < self.apathy_floor:                              # no self-relevant gap → noise passes through
        self.memory.short_term_decay(); return

    n_iter = self.sae.iteration_budget(sev, self.drives)     # SEEKING/severity → how much to think
    for _ in range(n_iter):                                  # Mind works the impression (inference)
        self.mind.update(eps, ctx)
        g_hat = self.reality.predict(self.mind.q, ctx)
        eps   = self.sae.reweight(o, g_hat, ctx)

    self.memory.write(o, eps, sev, coh, group)               # surprise-gated retention
    cands = self.evidence.accumulate(self.mind.delta_q, coh, ctx)   # impression → evidence
    for c in cands:
        self.gate.consider(c)                                # only path into Reality (Truth/Definition/C/context)
    self.trauma.scan(eps, sev, self.gate.rejections)         # failed-integration watch
    self.drives.update(group, sev, self.mind.q)              # FEAR/SEEKING are themselves prediction-driven
    return self.act(ctx)                                     # EFE planner chooses action → boundary

def act(self, ctx):
    policies = self.enumerate_policies(ctx)                  # short action sequences (postulates only)
    G = [self.expected_free_energy(pi, ctx) for pi in policies]   # §2.2, no reward
    return sample(softmax(-self.gamma * stack(G)))           # action exits; returns as next perception
```

That is the whole cycle from `tsm.md` §18 — Reality predicts, Perception challenges, SAE appraises, Experience/Impressions enter Mind, Coherence tests, Memory retains, Evidence stabilises, the gate updates Reality, Action exits and returns. Nothing in this loop is a reward step.

---

## 6. Learning across timescales

Four loops, fastest to slowest. **[Decided]** they update different things and must not be collapsed:

1. **Inference (per tick, ms):** Mind's `q` settles by minimising F. Belief change only. Touches no slow param.
2. **Slow consolidation (per N ticks, gated):** the generative params θ and long-term memory update by EMA / gated gradient on accumulated F, and only what survives the **mutation gate** reaches Reality/Foundation/C. This is where "learning" in the weight sense happens — deliberately decoupled from the fast loop so a single surprising tick cannot rewrite the ground (anti-trauma, §4.10).
3. **Developmental promotion (per stage, test-gated):** the scheduler advances stages only on passing tests (§4.11), flipping modules/losses/plasticity on. Equilibration runs continuously within a stage.
4. **Autoresearch (outer, optional):** Source 7's experiment→failure→hypothesis loop becomes the *outer* loop on the whole project — not a reward loop, a model-improvement loop: when a stage stalls, generate a hypothesis about the missing schema/Definition/curriculum, run a variant, keep what reduces free energy on held-out streams, log provenance. **[Default]** keep this manual/semi-automatic at first; it is your meta-loop, not the agent's, until stages 1–3 are stable.

> **Implication:** there is no global backprop through the whole Self at runtime. The graph that trains θ is the world-model graph (predict → compare → F); the graph that selects actions (EFE) is rolled forward but its "loss" is the surprise of imagined futures, not a return. These two graphs are why the system is trainable on one GPU without a reward model and without RL infrastructure.

---

## 7. First embodiment — EMBRYO-CRYSTAL (PyBoy / Pokémon Crystal)

The cheapest faithful testbed for the whole architecture. Mapping the abstract design onto Crystal:

- **PerceptionSurface:** framebuffer patches (downsampled) + symbolic RAM channels (party HP, position, map id, menu state, battle state) + last-action token. `source_confidence` is high (deterministic emulator) — useful, because it isolates the architecture from perceptual noise early.
- **Postulates (allowed actions):** the GB button set as the *only* permitted operations (Source 1: action constrained by lawful primitives). The EFE planner enumerates short button sequences as policies.
- **Preference priors C (FEAR side):** homeostatic prior on party not fainting / HP > 0 (a prior over observations, not a penalty). FEAR raises precision on HP-loss prediction errors.
- **Curiosity prior (SEEKING side):** prior expectation of resolvable novelty — drives map exploration, trying mechanics — purely via the epistemic term of G. **No score, no badge reward, no RL return.** The agent explores because unexplored states have high expected information gain.
- **Contexts (seeded):** overworld-navigation, battle, menu/dialogue — genuinely distinct meaning-fields with distinct expectations and Definitions. Online context birth allowed (e.g. a new region that routes poorly).
- **Definitions (will form):** walkable/blocked tiles, NPC vs object, "this move is effective/neutral/not" — exactly the ternary devalue/neutral/value structure, formed when the distinction changes outcomes (the stone test, in-game).
- **Stage gates, concretely:**
  - *Pre-incarnation:* learn to predict the passive title/intro stream before control is granted.
  - *Sensorimotor / object permanence:* predict an NPC's position after walking off-screen and back.
  - *Symbolic:* a learned tile-type Definition predicts collision before contact.
  - *Concrete-operational:* conservation (party identity persists through a menu reshuffle); reversibility (predict the result of undoing a step).
  - *Formal-operational:* plan a multi-step route to a never-seen goal via EFE rollout.
  - *Metacognitive:* reflective re-inference measurably improves route prediction.

> Crystal is the right first body precisely because it makes the **objective question unavoidable and answerable**: if the agent explores and survives with *no reward channel*, the active-inference claim is demonstrated, not asserted. That is the experiment the whole thesis rests on, and it runs on a 3090.

---

## 8. Hardware / runtime plan (RTX 3090, 64 GB, i9-10900K, Python 3.11)

**[Decided]** sized to fit one 3090 (24 GB) through stage 3; nothing here assumes multi-GPU.

- **Precision:** `bf16` for Mind/forward and the world-model graph; `fp32` for Reality params, accumulators, and the gate. The §3 continuous/discrete split is also the precision-of-storage split — Mind is allowed to be low-precision, the ground is not.
- **Footprint at sensorimotor stage [Default sizes]:** D=256, Lw=64, H=2, K=8, J=16 → model is small (low hundreds of MB params). The 3090 budget goes to (a) the Perceiver cross-attention over framebuffer tokens, (b) Mind's iterative inference, (c) EFE rollouts. All three are bounded by Lw and rollout depth, both small early. Comfortable headroom.
- **PyBoy** runs CPU-side on the i9; emulator stepping and the generative model overlap (env step while GPU does inference/rollout).
- **EFE rollout cost** is the main knob: depth × branching × Lw. **[Default]** depth 3, branching small, severity-gated — expand only when SAE says it matters (Source 7 bounded compute, off-RL).
- **Memory (system RAM, 64 GB):** the kNN episodic index + external-context handles + experiment ledgers live here, not in VRAM. Plenty.
- **[Question]** ship a BitNet-style low-bit backbone from day one, or prototype fp/bf16 and compress later (Source 6 open question). **[Default]** prototype in bf16 with the *ternary projection already in place at the Mind→Reality boundary* (it is architectural, not an optimisation), but keep θ in bf16 for now; convert θ to native ternary once the architecture is validated. This gets the conceptual ternary correct immediately without fighting low-bit training stability while the design is still moving.

---

## 9. Repo layout (single living structure)

**[Decided]** one package, mirroring the module spec, with the design doc living beside the code (your single-living-document discipline):

```
tsm/
  TSM_PYTORCH_DESIGN.md         # this file — the living spec
  self_field.py                 # Self container, tick(), act()
  perception.py                 # PerceptionSurface, tokenizers, Perceiver bottleneck
  reality.py                    # Reality: g_theta/f_theta, Foundation, preference priors C
  mind.py                       # recurrent inference core, reasoning ops
  sae.py                        # precision-weighted PE, appraisal groups, coherence
  ternary.py                    # TernaryProject autograd, hardening schedule
  definitions.py                # DefinitionBank, TsmDefinition boundary objects
  context.py                    # ContextRouter, basins, online birth
  memory.py                     # 5-system confederation, surprise-gated writes
  evidence.py                   # drift accumulators, promotion
  gate.py                       # MutationGate (every Reality write), Decision Log
  trauma.py                     # TraumaMonitor, quarantine
  drives.py                     # SEEKING / FEAR gain dynamics
  develop.py                    # DevelopmentalScheduler, stage tests
  free_energy.py                # F (perception/learning) and G (action) — the only objectives
  embody/
    crystal.py                  # PyBoy env adapter, obs/action spaces, stage tests
  loops/
    inference.py consolidate.py promote.py autoresearch.py
```

---

## 10. Open questions, consolidated (the real forks)

Pulled together so they're visible at once; each has a running default above.

1. **Mind hierarchy:** explicit H-level predictive coding vs deep SSM that folds the hierarchy. [Default H=2 explicit early.]
2. **Context lifecycle:** fixed K vs online birth; how aggressively to seed inherited contexts vs learn them. [Default seed 3 in Crystal, allow gated birth.]
3. **Definition split/merge criteria:** the exact statistic that triggers refinement vs compression. [Default bimodal residual under evidence → split; redundancy → merge.]
4. **Bits unit for precision budgets:** posterior precision vs Definition count vs routing entropy. [Default first two.]
5. **Missing SAE variables:** which of agency/controllability/reversibility/cost/moral-constraint to add and when. [Default controllability + cost_of_action first.]
6. **Long-term memory depth:** surprise-gated fast-weights + kNN vs full Titans-style test-time training. [Default the former first.]
7. **Low-bit backbone timing:** native ternary θ from day one vs after validation. [Default after; keep boundary projection now.]
8. **Drive dynamics:** keep SEEKING/FEAR scalar vs make them low-dim vectors (multiple seeking/fear channels). [Default scalar through stage 2, vectorise if needed.]
9. **Autoresearch automation:** how much of the outer loop the system runs vs you run. [Default you run it until stages 1–3 are stable.]

---

## 11. Tensions I'm flagging (so they don't get lost)

These are real, from the aggregation, and I am not pretending they're solved:

- **Self-evaluation is not Truth.** Mind's reflective scores feel like signal and are easy to over-trust (Source 7 / MCTSr). They can raise evidence; they cannot promote. If this leaks, the system starts believing its own confabulations — the LLM failure mode the whole gate exists to prevent.
- **Surprise is not automatically worth remembering.** Titans-style "surprise → memory" (Source 7) will, untreated, fill memory with noise and trauma. The store/quarantine/ignore check is doing load-bearing work; if it's too permissive, the long-term store rots.
- **Outliers break projection.** A single outlier impression can corrupt a hardened Definition (Source 6). The TraumaMonitor's outlier handling has to fire *before* projection, or a rare-but-real high-stakes signal gets either crushed or lets garbage into Reality. This is a genuine knife-edge.
- **Euclidean rigor vs living adaptivity** (Source 1 tension). The Foundation must constrain without making the system brittle. Too rigid → can't accommodate; too loose → loses the typing discipline that stops the latent space from collapsing into a similarity blob. The inherited-vs-learned tag and the stage-scheduled gate thresholds are the intended pressure-release, but the balance is empirical.
- **Quenched disorder is a tuning problem, not a solved equation.** Gate thresholds scheduled by stage is the *mechanism*; the *values* (how malleable at stage 1, how rigid at stage 5) are unknown and matter a lot. Get it wrong toward rigid and it can't develop; toward malleable and identity drifts.
- **Incommensurability has to actually block, not soften.** Source 1's deepest warning is against forcing unlike things into one measure. The `incompatible_relations` guard only helps if it hard-blocks cross-kind ratio/averaging rather than down-weighting it. A soft version silently reintroduces the very collapse it's meant to prevent.

---

*End of current spec. This is the whole architecture, not a slice — every `tsm.md` section and every architecturally-relevant source in the aggregation has a home above. The forks in §10 are where it's genuinely undecided; everything else is committed and buildable as written.*