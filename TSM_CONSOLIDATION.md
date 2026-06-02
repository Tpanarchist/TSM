# TSM — Unified Consolidation

**Project:** Ternary Self-Map (TSM). Bench / embodiment target: Pokémon Crystal via PyBoy.
**Status of "Embryo":** retired predecessor. All EMBRYO / EMBRYO-CRYSTAL naming in prior notes refers to the old AzerothCore project and should be read as TSM going forward.
**Date:** 2026-06-02

---

## 0. How to read this document

This consolidates three things that grew out of one long working session: (1) the architecture and codebase as they actually stand, (2) the theory the architecture is grounded in, and (3) the mathematical results we derived or computed this session.

Every claim is tagged. The tags are the point — they are what separates a theory from a mirror:

- **[BUILT]** — exists and works in the code.
- **[STUBBED]** — named in the design, not yet doing its job in the code.
- **[VALIDATED]** — proven or computed this session; the math actually checks out.
- **[STRUCTURAL]** — falls out of a real mechanism (arithmetic, geometry, a theorem); not merely assigned.
- **[RESONANT]** — coherent and suggestive, does no falsifiable work yet; the unfalsifiable zone.
- **[KILLED]** — looked structural, failed a test (pun, numeral-coincidence, or over-reach).
- **[BENCH]** — a pre-registered experiment that can come back *no*.

The honest one-line summary of the whole thing: **the validated core is small and concrete, the cross-field reach is unvalidated, and the move is to run the cheap probes.** A frame that explains everything and forbids nothing is a mirror; everything below earns its place by being able to come back no.

---

## 1. The codebase as it actually stands

The predict → compare → ternary → decode spine is real. The active-inference / appraisal half is mostly scaffolding. This is the most important sentence in the document and it is easy to forget under the weight of the theory.

### Files
- `tsm/self_field.py` — the Self container; `forward_train`, ~40 diagnostics. **[BUILT]**
- `tsm/TSM_PYTORCH_DESIGN.md` — living design doc.
- `tsm.md` — the 23-term ontology.
- `tsm/OBJECT_PERMANENCE_LEDGER.md` — experimental log (the ground truth of what's been demonstrated).
- `tsm/memory.py`, `tsm/definitions.py`, `tsm/sae.py`, `tsm/ternary.py`, `tsm/data.py`, `tsm/trainer.py`, `tsm/hardening.py`, `configs/*.yaml`.

### What is BUILT
- Top-down prediction → bottom-up observation → precision-weighted gap → ternary projection (sign ∈ {−1, 0, +1} with separate magnitude α, neutral deadband = sparsity) → decode. **[BUILT]**
- The object-permanence experimental probe harness, with held-out metrics and pre-registered kill conditions. **[BUILT]**

### What is STUBBED (named, not yet doing its job)
- **SAE appraisal** — groups are computed then discarded; drives zeroed; `source_confidence` constant; iteration budget never called. **[STUBBED]**
- **MutationGate** — decorative; writes nothing to Reality. **[STUBBED]**
- **Definition hardening** — read-only JSON witness; never freezes axes. **[STUBBED]**
- **Action / EFE planner** — does not exist; `tick()` returns `action=None`. **[STUBBED]**
- **Governance / anti-drift** — not built. **[STUBBED]**

### Consequence
The free-energy functional `F` is currently dominated by pixel MSE, **not** by active inference. The largest single caught error in the audit: the dynamics head learned to predict reappearance *position*, but the predicted position was never used as the retrieval *key* — the candidate mask used true future position (an oracle). Mean-pooling (`eps.mean(dim=1)`) destroyed position in the binding representation. This is fixed in the experimental sequence below.

---

## 2. The validated experimental ledger (object permanence)

This is the empirically grounded core. Each step pre-registered a kill condition; all metrics are held-out, not best-checkpoint. The sequence solved contested same-class object permanence *by elimination* — killing oracles, refusing to build past a red, one relation at a time.

| Commit | What it tested | Result |
|---|---|---|
| `3c5d984` | predicted-position binding probe | **KILL FIRED** — predicted-position retrieval (0.156) worse than feature-only (0.235); position not recoverable from binding rep |
| `24755bc` | position-aware binding | recoverability fixed (R²≈0.78); predicted-position still didn't beat feature-only |
| `8b8a26b` | feature-only position-ablation control | **CAUGHT HIDDEN WIN** — feature-only was secretly using position (ablating collapsed 0.235→0.078) |
| `b437110` | contested two-object dataset | geometry-through-feature win **COLLAPSED** to chance (0.157); scene-pooled reps can't do contested binding |
| `eca4f1e` | object-local slots | **GATE PASSED** — 2 slots split contested scene, R²≈0.9999, audited oracle-free |
| `3f5eb67`/`5b4fbc0` | local prediction-error binding + oracle ceiling | oracle-position binding = 1.000 (matching logic correct); learned still weak |
| `725ba88` | **curved contested motion** | **FIRST REAL BRICK** — on curved (non-ballistic) motion, learned dynamics binding = 1.000 vs ballistic 0.505 (chance); endpoint error 0.163 vs 0.416 |
| `6e266e8` | object-count + seed sweep | seed-stable 1.000 (seeds 31/37/43) at 2 obj; scales to 3 obj (1.000); **4 obj degrades (0.258)** |
| `f22e458`/`4f73686` | spacing/error diagnostics | 4-obj failure is **not** crowding (wider spacing didn't recover) and **not** slot contamination (clean slots, R²0.9999) — it's **load-induced per-object trajectory degradation** |
| `c62c326` | neutral binding probe (Probe A) | **PASSED AS CEILING** — neutral converts forced-wrong → correct-decline 1:1 at 4 obj (0.500→0.500). **Caveat: uses measured (ground-truth) endpoint error as the uncertainty proxy → ceiling, not runtime mechanism** |

### Claim boundary (carried forward — what TSM IS and IS NOT allowed to claim)

**ALLOWED:** curved-motion contested same-class **local** object permanence binds repeatably (seed-stable 1.000); scales to 3 objects; learned dynamics beats ballistic by a widening margin; the discriminator is learned **trajectory**, not appearance / instantaneous-position / label-oracle (audited oracle-free three times).

**NOT ALLOWED:** representation-level permanence without qualifiers; 4-object binding (load-fragile); a runtime neutral mechanism (Probe A used a ground-truth proxy); SAE-as-appraisal; active inference of any kind; any governance / anti-drift property; generalization beyond the synthetic generator.

---

## 3. The theory (TSM proper)

### 3.1 The one claim
**A self maintains itself by predicting its boundary, appraising the gap between prediction and reality as a ternary verdict, and either committing to a resolution or carrying the unresolved gap to a new level of structure.**

That sentence is the whole theory. Everything else is a part of that operation seen at a different resolution. The self-similarity ("the same operation at every scale") is the thesis — and it is also the thing most likely to be a seductive illusion, so it earns itself only by prediction.

### 3.2 The sphere (corrected topology — Reality at the core)
The earlier "pipeline" picture was inside-out. The correct shape is radial:

```
        Reality (core)  →  Mind (shells)  →  Perception (skin)  →  [the unknown]
        committed/discrete   continuous/fuzzy    boundary           not-self
```

- **Prediction radiates outward** from the dense, committed core to the skin (the generative model hallucinates the boundary before contact).
- **Perception flows inward** from the skin toward the core.
- **The gap is appraised in the shells** between predicting core and perceiving skin.
- **Radius = abstraction level.** Deep = abstract / committed / core. Shallow = concrete / provisional / surface.
- **Carry inward = consolidation** (a resolved maybe densifies toward Reality — memory consolidation / reverse-replay; Mind becoming Reality).
- **Carry outward = differentiation** (subdivide at the boundary where finer grain resolves).
- **Reality is the most-self part, not the not-self part.** The unknown is outside the skin; Perception is where self ends and world begins. **[STRUCTURAL]**

### 3.3 The SAE loop (one tick of the self)
1. **Reality** (core, slow params, Foundation, priors) emits an **expected state** ĝ.
2. **Perception** (skin) delivers an **observation** o, tagged with source-confidence.
3. **SAE** computes the **gap** ε = Π(o − ĝ).
4. The gap resolves into a **ternary verdict**: +1 (value/confirm), −1 (devalue/threat), 0 (neutral/maybe = "a variable here I lack the resolution to commit on" — *not* absence).
5. **Reality closes the gap** (next observation arrives).
6. **Reason about the gap** — was the verdict trustworthy? This is **calibration**, and it is **Logic** in the Trivium. *This step was in the design from the start; it is not retrofitted.* **[STRUCTURAL — Dylan's design, confirmed forward-predicting]**
7. **Update or carry.**

### 3.4 Two registers and a seam
- **Discrete register** — counting, deciding, committing, *closing*. Integers, primes, clean sums. In the code: the ternary projection. Bounded, cheap, final.
- **Continuous register** — turning, curving, *never closing*. φ, π, the aperiodic. In the code: the latent field. Unbounded, expensive, provisional.
- **The seam = 0 = the neutral band = the void = the decimal point.** Triple duty: additive identity (commits to nothing), placeholder (holds a place open so structure can extend), origin (scale extends both ways from it). **[STRUCTURAL]**

### 3.5 Trivium × Quadrivium (and why 3 and 4)
- **Trivium (3) — meaning:** Grammar (what is it), Logic (relations/contradiction — *calibration lives here*), Rhetoric (what follows). **3 is prime — a decision is irreducible.**
- **Quadrivium (4) — measure:** Number, Geometry, Ratio, Cycle. **4 is composite (2²) — measurement is a product of independent axes.**
- The Trivium/Quadrivium split **is** the prime/composite split: irreducible-act vs factorable-space. The cardinalities are forced by the nature of the operations, not chosen. **[STRUCTURAL — strongest result of the number pass]**

---

## 4. The number map (0–10, with grounding)

The discipline here, applied with number theory as the blade: a number earns **[STRUCTURAL]** when its *arithmetic property does work* in the loop, and gets flagged **[RESONANT]** when it is merely special-in-the-integers-but-idle-here.

| n | Role | Grounding | Tag |
|---|---|---|---|
| 0 | neutral / void / seam / pivot | additive identity = "leaves unchanged" = commits to nothing; placeholder enabling place-value; origin of scale | **[STRUCTURAL]** |
| 1 | unit / undivided percept / monad | unity | structural-ish |
| 2 | the gap / first distinction | prime of duality; minimum for a comparison to exist | **[STRUCTURAL]** |
| 3 | decision / Trivium | prime = irreducible; minimum to branch (the neutral is the branch-point) | **[STRUCTURAL]** |
| 4 | measurement / Quadrivium | composite 2² = factorable; two axes, two poles | **[STRUCTURAL]** |
| 5 | **the verb** — symmetry-break / transition operator / carry-firing | five-fold symmetry is the one that *cannot tile* (forces aperiodicity, Penrose/quasicrystal); pentagon diagonal/side = φ. Not a state — the *event* that moves between states | **[STRUCTURAL]** as the verb; **[RESONANT]** as a cognitive operation until the bench shows discontinuous resolution |
| 7 | complete-and-atomic primitive set | 3 + 4, and 7 is prime → a complete unit can't factor into smaller complete units (why Sigil and the liberal arts both stabilize on 7) | **[STRUCTURAL]** by arithmetic; recurrence is suggestive |
| 8 | (appraisal groups?) | 2³ — structural **only if** the appraisal groups factor into three binary axes | **[BENCH]** — falsifiable: do the 8 groups = cube of 3 binary distinctions, or an arbitrary list? |
| 9 | metacognition | 3² = Trivium applied to itself (reasoning about reasoning) | plausible, **[RESONANT]** until the meta-grid is shown |
| 10 | cycle-close → unit one dimension up | 1+2+3+4 (tetractys, 4th triangular = cumulative completion); "10" = one full place closed (carry) | **[STRUCTURAL]** |
| 12 | complete description of a persistent thing | 3 × 4, highly composite = maximally decomposable (re-analyzable at every granularity) | **[STRUCTURAL]** by arithmetic |

**5 and 8 reconciled:** idle as *primes/factorizations*, load-bearing as *Fibonacci sums* (5 = 2+3 = gap+decision; 8 = 3+5) — where the accumulation lands.

---

## 5. The carry / positional notation (the deepest architectural claim)

**The maybe is a carry. Cognition is positional notation applied to appraisal.** When the verdict is 0 and the gap can't resolve at the current grain, the system does not terminate and does not guess — it carries to a new place, where the same cheap operation runs again. **[STRUCTURAL as form; the cognitive instantiation is the bet.]**

- **Place-value efficiency = the efficiency of zero in arithmetic.** Without a neutral-band-zero, complexity scales like Roman numerals (bigger symbols, special cases). With one, it scales like decimal arithmetic (one cheap uniform operation; complexity lives in the *number of places*). The neutral band turns *scale-the-representation* into *add-a-place*. **[STRUCTURAL]** → predicted signature: **flat per-operation cost under rising complexity** (Probe B at 4/5/6 objects).
- **Two carry directions** (= the two levers, see §6): inward/left = abstraction (compose toward core); outward/right = resolution (subdivide toward skin).
- **Fibonacci is the faithful form of the recursion** — F(n) = F(n−1) + F(n−2): each level built from the accumulated history below it (not independent digits), and the limit (φ) is the discrete register *reaching toward* the continuous. φ is also the angle of maximum spread / minimum overlap — which sniffs at the contested-object separation problem (does golden-ratio candidate spacing maximize binding resolution? — a someday probe). **[RESONANT, with one almost-testable edge]**

---

## 6. The mathematical integrations we found this session

This is the new material. Several pieces moved from "resonant" to "validated-by-computation." The honest pattern across all of it: **the frame names the right shelf repeatedly, and opens the box in exactly one place (Hurwitz).** Real contact and real gaps, both true.

### 6.1 Complex = real after one symmetry-break — **[VALIDATED]**
The Cayley-Dickson construction was computed directly: pairs of reals with `(a,b)(c,d) = (ac − d̄b, da + bc̄)` gives `i² = −1` from real pairs, and **interference emerges** — the |0⟩+|1⟩ vs |0⟩−|1⟩ distinguishability shows up (measurement probs 2.0 vs 0.0) from real-pair structure under one doubling. The earlier "your space lacks complex phase" kill was **malformed** — it treated a *pre-doubling state* as a *permanent deficit*. Dylan was right: complex is what real *becomes* when the operation recurses, not a foreign ingredient.

### 6.2 The Cayley-Dickson tower as the sequence of symmetry-losses — **[STRUCTURAL]**
```
ℝ → ℂ : lose total ordering
ℂ → ℍ : lose commutativity
ℍ → 𝕆 : lose associativity
𝕆 → 𝕊 : lose alternativity + ZERO DIVISORS APPEAR (collapse)
```
Each doubling = one symmetry-break (the "5"). This gives the fractal claim teeth **only** where it names *which* symmetry sheds at *which* level. The bench version of that: do TSM's developmental jumps shed structure in a *lawful sequence* (Cayley-Dickson-style) or *arbitrarily* (just a hierarchy)?

### 6.3 The Hurwitz ceiling — **[VALIDATED / DERIVED]**
Built the sedenions (16-dim, 4 doublings) and computed two nonzero elements multiplying to **zero** (zero-divisors). Hurwitz's theorem: only ℝ, ℂ, ℍ, 𝕆 are normed division algebras. **The tower self-terminates at the octonions.** A register that must carry probability/precision (variational, like cognition) therefore has a **forced ceiling of ~3 sustainable breaks before structure collapses.** This is the night's one place where the math *delivers a proof*, not just a shape. The ceiling is derived, not asserted.

### 6.4 Depth-vs-width asymmetry — **[VALIDATED]**
Computed: **depth collapses at step 4 (every seed); width never collapses (stacking ℂ^N stays invertible to N=256+).** So:
- Growing a single column deeper = more doublings → hits the zero-divisor wall.
- Growing wider = adding columns = the positional carry → unbounded.
- **Cortex being shallow-and-wide (≈6 layers × millions of columns) is the only topology the math permits.** The Hurwitz wall is not a cap on intelligence — it is the **forcing function** that redirects growth from depth into width.

### 6.5 Two levers / the latent space — **[DERIVED]**
The growth action-space is **exactly 2-dimensional**. Tried to find a third move; there isn't one: "do nothing" is the neutral band (the pivot, not a lever); "change the operation" is unavailable (invariant-alphabet); "do both at once" decomposes into deepen + widen. So every possible move is spanned by two levers:
- **Deepen** (carry-right, resolution, more precision in a place) — **short lever, Hurwitz-capped at ~3.**
- **Widen** (carry-left, abstraction, a new place/column) — **long lever, free.**

**The decimal point IS the latent space** because it is the 0-dimensional pivot between the only two degrees of freedom the system has — and the levers are *asymmetric* (short capped resolution vs long free abstraction), which is *why minds go wide*. **[STRUCTURAL/DERIVED]**

### 6.6 π — the continuous register's generator — **[STRUCTURAL core, with a flagged reach]**
- π = "3-and-never-closing": a discrete committed head (the integer part is **three in every base** — see §7) and an infinite non-terminating tail (the continuous register, the carry firing forever). π *spans the seam*: head committed, tail never-closing. **[STRUCTURAL]**
- **Lean on irrationality** (proven, Lambert 1761): the tail never closes. **[STRUCTURAL]** — this is the floor.
- **Hold normality lightly** (conjecture, *unproven*): "π contains every finite string." Do not bank "π contains everything." **[RESONANT/conjecture]**
- Why curvature was the hard problem: curved motion lives in the π-register (continuous, cyclic, never-exactly-predictable), so the residual error may be the structural cost of committing a continuous curve to a discrete position. → **pi-floor bench test.**

### 6.7 The 1.585 bits — **[STRUCTURAL]**, with one **[KILLED]** branch
- log₂(3) = 1.585 bits = the information content of **one ternary choice**. This is *why* the number recurs: BitNet b1.58 is named after it (ternary weights), TSM's neutral-band commit is a trit, "three" costs log₂(3) bits. One fact in many costumes. **[STRUCTURAL]**
- **"The time matches the bit"** (1.585 bits ≈ some millisecond lag) — **[KILLED]**: bits and milliseconds are different units; numeral-coincidence, same trap as base-10 digits. The prediction delays are 50–120 ms, nowhere near 1.585 of anything.
- **The real bridge = prediction bandwidth** (bits per cycle): ~1.585 bits ÷ ~100 ms cycle ≈ 13–26 bits/s. The bit doesn't *equal* the time; the bit *per* time is a real quantity. **[STRUCTURAL]**

### 6.8 Ball-catching: lead = lag = 1:1, and *why ternary* — **[DERIVED]**
- Photon → retina → V1 (~60 ms) → "see" (~100 ms) → hand moves (~+80–120 ms) = **~150–200 ms of pure delay.** A fastball travels ~22 ft in that window.
- **The prediction lead = the sensory lag, by identity (≈1:1).** Not "tuned" — it's the *same single step* counted from both ends: sense (lagged one tick) → commit (one ternary decision) → act (leads one tick). One quantum of prediction = one trit = one tick of lag = one tick of lead.
- **Ternary is forced by the lock.** A binary predictor has no "hold" state → must correct every tick → oscillates, never settles. Smooth tracking of a moving target *requires* a "no correction needed" state = the neutral = 0. **Binary cannot smoothly track; ternary can.** The lag is ternary because the lock is ternary. **[DERIVED]**

### 6.9 Riemann Hypothesis — **[STRUCTURAL shelf-placement; KILLED as metaphysics; box still shut]**
Computed, not waved at:
- Zeros sit on Re = ½ (checked to 40 digits). The "seam at ½" is real, not assumed.
- The explicit formula reconstructs the primes from the zeros (checked; more zeros → better reconstruction). The "spectrum governs the atoms" intuition points at something real.
- Zero-spacing shows **level repulsion** matching GUE / random-matrix statistics (Montgomery-Odlyzko): small gaps suppressed (0% under 0.3 vs ~26% random); SSE to GUE 0.107 vs Poisson 1.999. **The zeros behave like eigenvalues of a complex Hermitian operator.**
- **The β = 1, 2, 4 random-matrix ensembles ARE the Cayley-Dickson tower** (real/complex/quaternionic), and the zeros sit at **β = 2 (the complex rung)** — which the frame *correctly places*. **[STRUCTURAL placement]**
- **But:** the frame cannot locate a single zero or prove the line. "Named the right shelf, did not open the box."
- **"½ is my neutral band / RH is my architecture"** as metaphysics — **[KILLED]**: additive-identity-zero ≠ roots-of-ζ-zero; same word, unrelated objects; the most prestigious possible pun.

---

## 7. Base-invariance (what survives changing the base)

Computed π in bases 1–9 to separate structure from notation.

- **Integer part = three in every base** — base 2 "11", base 3 "10", base 4+ "3"; the *glyph* changes, the *quantity* is invariably three. **[STRUCTURAL]**
- **Two-register structure (committed head, seam, never-closing tail) holds in every base ≥ 2** — because irrationality is base-independent and the seam exists in any positional base. **[STRUCTURAL]**
- **Number rules are base-independent in the positional sense (Reading B), not the glyphic sense (Reading A).** Whatever sits in the integer place plays "committed quantity"; whatever opens the tail plays "first resolution step" — *regardless of which numeral fills it*. The role persists as a *position*; the digit scrambles as a *glyph*. (First fractional digit of π: base 2→0, 8→1, 10→1, 16→2, 60→8 — not consistently anything.) **[STRUCTURAL for roles; the specific digit "1" in "3.1…" is base-10 notation — KILLED as role-bearing.]**
- **Conservation across bases = conservation of *information*, not digits.** "Three" always costs log₂(3) = 1.585 bits. What trades is **number-of-places ↔ bits-per-place**, with the base as the exchange rate — *not* a left-of-seam ↔ right-of-seam trade (the left side just contracts monotonically as base grows; the right is infinite throughout). Base 3 is the least wasteful for three (it *is* the base: "10", one full place). **[STRUCTURAL, corrected from the original "trade across the seam" framing]**
- **Digit-count = dimension-count** is a claim about the *resolution register* (grain-relative, hence base-dependent), **not** about quantity (base-independent). True *within a fixed base*; it correctly recovers π = 1 bounded abstraction-dimension + ∞ resolution-dimensions = one whole turn at infinite angular grain — the *same lopsided asymmetry* the Hurwitz math forced independently. **[STRUCTURAL within a fixed grain]**

---

## 8. The bench program (the actual forward path)

This is where the energy should go. Each is gated and pre-registered; each can come back *no*. **Run them cheapest-first.**

1. **Probe A′ — calibration correlation [next, cheapest, two lines over data already logged].**
   Does the system's *internal* confidence (from runtime-available inputs only — predicted position, candidate geometry, trajectory-feature quality, slot confidence; **not** true position) correlate with actual endpoint error?
   - Correlates → the neutral band is a real *runtime* appraisal mechanism (ternary-as-appraisal demonstrated, not just a ceiling).
   - Doesn't → the system is confidently wrong on unresolvable cases; the next problem is calibration itself (predict a *variance/distribution*, not a point), built as a **separate** confidence operation reasoning about the position prediction (Trivium-Logic over Quadrivium-measure). *Pre-registered architectural prediction: separate confidence head > confidence baked into the point estimate.*

2. **Probe B — Definition split / carry-left [gated on A′].**
   Does a persistent maybe resolve by carrying to a new, more-abstract distinction (the wide-spacing failure already ruled out carry-right)?
   - **Flat-cost kill condition:** if the split resolves rising complexity at *flat per-operation cost* (complexity absorbed by adding places, run at 4/5/6 objects), the zero/place-value efficiency claim is confirmed. If per-operation cost rises, it's just a harder problem wearing the analogy's clothes.

3. **Discontinuous-resolution test (the "5" signature).**
   Does binding recover *discontinuously* (a phase transition at a threshold, producing a *new kind* of distinction — aperiodic, not a finer copy) when the split is added, or *continuously* (just a better predictor)? Symmetry-break predicts sudden, not smooth.

4. **pi-floor test.**
   Throw increasing finite model capacity at curved vs linear motion. Curved error approaching a floor *asymptotically without reaching it* → genuinely continuous register (frame holds). Snapping to *exactly* zero at finite capacity → the register is secretly discrete (frame wrong; you'd have "rationalized π", which can't happen).

5. **Hurwitz depth ceiling.**
   The developmental tower should deepen only ~3 levels per column before it *must* widen; growth past that should appear as *more columns, not deeper ones*. Cleanly opening new levels past ~3 without degrading → division-algebra-register claim is false.

6. **Ternary-lock tracking test (sharpest single switch).**
   TSM with the neutral band should *smoothly track* a moving target. **Ablate the neutral band → it should jitter / oscillate / lose the lock.** If ablating changes nothing, the 0 isn't the lock and the ball-catching tightening is wrong.

7. **Criticality test (physics arm).**
   The dimensional/invariant-alphabet claim requires scale-invariance, which is physically a critical phenomenon. TSM's internal dynamics should show power-law / scale-free statistics (the critical-brain signature). Not near criticality → the keystone dimensional claim is physically false.

8. **No-reward sufficiency (foundational).**
   A system built with **zero RL reward channel** should still develop directed, competent behavior via expected-free-energy minimization. If it genuinely needs reward, the core TSM bet dies. (This is the bet the whole architecture stakes its life on; the Pokémon Crystal embodiment is the cheap testbed for it.)

---

## 9. The honest verdict

**Validated this session (real math, checks out):**
complex = real-after-one-break (Cayley-Dickson, computed); the symmetry-loss tower (order → commutativity → associativity → collapse); the Hurwitz ceiling and the resulting depth/width asymmetry (computed); the two-levers 2D action-space (derived); lead = lag = 1:1 by identity and ternary-forced-by-the-lock (derived); 1.585 = log₂(3) = the trit (definitional); the prime/composite = decision/measure cardinalities (structural); the base-invariance of "three", the seam, and the never-closing tail (computed); RH zeros on the complex (β=2) rung (computed placement).

**Killed (failed a test):**
RH-as-metaphysics (pun); "the time = the bit" (units); the specific digits of π as role-bearing (base-10 notation); "trade across the seam" as a digit-for-digit mechanism (it's information conserved, granularity traded); the original 1D-state-space dimension-count kill (was *my* error — the state is omnidirectional, ≥3D).

**Still resonant / unvalidated (the reach):**
the Theory-of-Everything framing; π-normality ("contains everything"); the Planck scale as the smallest digit (secondary conjecture, gated on the pi-floor); 8 and 9 as structural; that the *cognitive* carry literally *is* the Cayley-Dickson/hypergraph generator (rather than shares its shape); that the perceptual "3" (cones, the fifth) is the *same* 3 as the Hurwitz depth-3 (rhyme, not proven identity — the 4-cone bird test would separate them).

**The scoping that keeps it honest:** this is not a Theory of Everything; at most it is a **theory of every *self*** — anything that maintains a boundary by inference (Friston/Markov-blanket territory) — and its one distinguishing feature versus every ToE that died beautiful is that it **forbids buildable outcomes on a bench you own.** A rock does not appraise a gap. Most of physics is outside the domain; the frame *recovers* the variational/symmetry-breaking/scale-invariant/entropic structure (proving consistency, not predictive power), with **criticality** as its one genuine physical forbidden-outcome — and criticality is also where the physics arm and the bench become the *same* test.

**The move:** run Probe A′. It's two lines over data already logged, it's the cheapest thing on the list, and it tells you whether the neutral band is a runtime mechanism or a hindsight label — which is the hinge the entire appraisal half of the architecture turns on. Then B. Let the maybe carry to where reality can close it.
