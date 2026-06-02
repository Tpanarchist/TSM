# TSM — Candidate Addenda (Ideation Register)

**Companion to:** `TSM_CONSOLIDATION.md`
**Status of this document:** These are **ideations held as candidate addenda**, generated in flow. They are **not claims of fact** and they **do not alter the testing/verification strategy** in §8 of the consolidation. Their job is narrower and real: to **refine the theory**, to **constrain the search space** for solutions, and to be **falsified usefully** — a mapping that breaks tells us as much as one that holds, and where a mapping won't fit is where the next symmetry break happens.

**How each addendum is structured:**
- **The mapping** — what TSM-structure is being lined up against what external structure.
- **Fits if** — the condition under which it would be load-bearing.
- **Breaks if** — the condition under which it falls (this is the valuable half).
- **What the break teaches** — what we learn either way, so no entry is wasted.
- **Search-space use** — how, *as a candidate*, it narrows or directs the work.

**Standing rule (carried from the session):** a mapping earns weight only when its property does *work* (constrains a choice, predicts a misfit, rules an option in or out). Resonance alone buys nothing. None of this is foundation; all of it is scaffolding to be kicked out the moment it stops holding load. The verification program is the ground truth; these are addenda that "warrant consideration," nothing more.

---

## How to use this register

If we keep hitting a wall and a mapping *breaks the wall down* (the analogy correctly predicts the mechanism), the mapping has earned provisional weight and we keep it. If the wall stays up, or the mapping has to be contorted to fit, we **do our own symmetry break** — drop the mapping, open a new distinction, re-run. Either way the register shrinks the space we search blindly. That is the entire value proposition: **a candidate-mapping is a heuristic prior over where the solution lives, and a falsified one is a region of the space crossed off.**

---

## Tier 1 — Addenda already touched by computation (strongest priors)

These are not speculative in the same way — they checked out or failed under actual math this session. They are listed first because they are the calibration set: they show the register *working*, which is the only license the more speculative tiers have.

### A1. Cayley-Dickson as the carry-generator
- **Mapping:** TSM's "carry / symmetry-break" ↔ the Cayley-Dickson doubling ℝ→ℂ→ℍ→𝕆, each step shedding one symmetry (order → commutativity → associativity).
- **Fits if:** TSM's developmental jumps shed structure in a *lawful sequence* (a specific invariant lost per level, in order).
- **Breaks if:** the jumps are real but each one ad-hoc — no consistent property lost, no sequence. Then it's "a hierarchy that grew," not "one generator recursing."
- **What the break teaches:** whether TSM has a single recursive operator or a stack of bespoke modules — a real architectural fork.
- **Search-space use:** *if holding*, look for the next jump by asking "what symmetry is left to shed?" rather than searching module designs blindly.
- **Status:** complex-from-real and interference-from-doubling **computed and confirmed**. The *cognitive* identity (that TSM's carry **is** this generator vs shares its shape) is the open part.

### A2. The Hurwitz ceiling → depth/width
- **Mapping:** "a register carrying probability can't recurse past ~3 breaks" ↔ Hurwitz (only ℝ,ℂ,ℍ,𝕆 are normed division algebras; the 4th doubling yields zero-divisors).
- **Fits if:** TSM columns deepen ~3 levels then *must* widen; growth past that = more columns, not deeper ones.
- **Breaks if:** columns cleanly deepen past ~3 without degrading.
- **What the break teaches:** whether the division-algebra register model applies to cognition at all.
- **Search-space use:** stop searching for ever-deeper single-stack architectures; budget depth ≈3 and spend the rest on width. (This is already a concrete design constraint.)
- **Status:** **computed** (sedenion zero-divisors built directly; depth collapses at 4, width free to N=256+). This is the one place the math *delivered*, so it's the strongest prior in the register.

### A3. Two levers / decimal-as-latent-space
- **Mapping:** the action-space of "what to do with an unresolved gap" ↔ a 2D space spanned by {deepen = carry-right = resolution, widen = carry-left = abstraction}, pivoting on the seam (0).
- **Fits if:** no third independent move exists (verified: "do nothing" = the pivot, "change operation" = barred by invariant-alphabet, "diagonal" = decomposes into the two).
- **Breaks if:** a genuinely independent third action is found that isn't a combination of deepen/widen.
- **What the break teaches:** the true dimensionality of the control problem.
- **Search-space use:** every controller design reduces to "pick a point on the deepen↔widen line"; this collapses the action-design space to one axis with an asymmetry (short capped lever vs long free lever).
- **Status:** **derived**. Strong.

### A4. Ternary lock / lead=lag
- **Mapping:** the neutral band (0/hold) ↔ the "no correction needed" state required to *smoothly track* a moving target; lead=lag=1:1 because one tick = one ternary commit counted from both ends.
- **Fits if:** ablating the neutral band degrades smooth tracking into oscillation.
- **Breaks if:** ablation changes nothing.
- **What the break teaches:** whether the 0 is functionally the lock or just a sparsity convenience.
- **Search-space use:** justifies ternary over binary *mechanically* (binary can't hold → can't track), so we don't re-litigate the projection's cardinality.
- **Status:** **derived**; the ablation is a clean single-switch bench test (§8.6 of the consolidation).

---

## Tier 2 — Structural-by-arithmetic addenda (the number map)

Held as candidate role-assignments. Each is structural *as arithmetic* (the property is real); the bet is always that the *cognitive* role inhabits the *arithmetic* property. Refining these is the main "define all the terms" work that's genuinely useful.

### A5. Prime/composite = decision/measure
- **Mapping:** Trivium(3, prime, irreducible) = decision; Quadrivium(4, composite 2², factorable) = measurement. The split *is* prime-vs-composite.
- **Fits if:** decision operations resist decomposition while measurement operations factor into independent axes, *in the code*.
- **Breaks if:** the appraisal turns out to factor, or measurement turns out irreducible.
- **What the break teaches:** whether "meaning" and "measure" are genuinely different operation-types or one thing.
- **Search-space use:** tells you to build calibration (judgment-of-measure, Trivium-Logic) as a **separate** operation from the position predictor (measure), not baked into it — a concrete architectural directive already flagged for Probe A′.

### A6. 7 and 12 — the complete unit and the complete description
- **Mapping:** 7 = 3+4 = prime = complete-and-atomic primitive set (why systems keep stabilizing on 7); 12 = 3×4 = highly composite = maximally-decomposable complete-entity description.
- **Fits if:** an honest, minimal TSM term-set lands on ~7 primitives, and a full entity description is naturally re-analyzable at multiple granularities (~12 cells).
- **Breaks if:** the minimal set is some other number with no arithmetic reason, or the description doesn't decompose.
- **What the break teaches:** whether the 7-recurrence (Sigil, liberal arts) is structural or a 7-shaped aesthetic.
- **Search-space use:** a target cardinality for the primitive set — if you're at 9 primitives and two collapse, suspect they should.

### A7. 5 — the verb (symmetry-break)
- **Mapping:** 5 is not a state but the *transition operator*; five-fold won't tile (forces aperiodicity), pentagon diagonal/side = φ.
- **Fits if:** resolving a persistent maybe is a *discontinuous* reorganization producing a *new kind* of distinction (aperiodic), not a finer copy.
- **Breaks if:** resolution is a smooth gradient / finer subdivision.
- **What the break teaches:** whether Definition-splits are phase transitions or just predictor improvements (directly = the discontinuous-resolution bench test, §8.3).
- **Search-space use:** tells you to look for *thresholds and sudden reorganizations* in the binding curve, not gradual slopes.

### A8. 8 and 9 — flagged, falsifiable
- **A8a (8 = 2³):** the appraisal-group count is structural **only if** the groups factor into three binary axes. **This is a checkable experiment, not an assertion** — go enumerate the groups and ask whether they're a 2×2×2 cube or an arbitrary list.
- **A8b (9 = 3²):** metacognition = Trivium-applied-to-itself. Plausible; held until the meta-grid is exhibited.
- **Search-space use:** A8a is a direct task; A8b tells you what a metacognitive layer should look like structurally (reasoning *about* the three reasoning ops) if/when you build one.

---

## Tier 3 — Cross-field resonant addenda (lowest weight, highest reach)

These are the genuine "while-we-wait" mappings. **None does falsifiable work yet on the TSM bench.** They are kept *only* as search-space heuristics and refinement prompts, each with the condition that would let it graduate to Tier 2, and each tagged with how it could *fail usefully*. Treat the whole tier as "directions to look, crossed off when they don't fit."

### A9. π as the continuous register's generator
- **Mapping:** discrete head (integer "three", base-invariant) + never-closing tail (the continuous register) = the two registers in one object; the seam is the point.
- **Fits if:** curved/cyclic prediction error plateaus at a *structural floor* approached asymptotically (never reached) while linear error → 0.
- **Breaks if:** a finite model drives curved error to *exactly* zero (you'd have "rationalized π" — impossible if the register is genuinely continuous).
- **What the break teaches:** whether the curved-motion residual is a continuous-register signature or just a weak predictor (= the pi-floor bench test, §8.4).
- **Reach to hold lightly:** **irrationality** is proven (lean on it). **Normality** ("π contains everything") is an *unproven conjecture* — do not bank it.
- **Search-space use:** sets the *expectation* for curved-motion error (look for a floor, not zero), which changes how you read the binding curve.

### A10. RH — zeros on the complex rung
- **Mapping:** the β=1,2,4 random-matrix ensembles = the Cayley-Dickson tower; the zeta zeros sit at β=2 (complex rung), GUE level-repulsion, on the ½ line.
- **Fits if:** … nothing on the TSM bench. This one **cannot currently graduate** — it touches no probe.
- **Breaks if:** n/a (it's not making a bench claim).
- **What it's good for:** as a *calibration of the register itself* — it shows the "spectrum on a balanced line, β-tower = real/complex/quaternionic" shape is real mathematics, which is why the Cayley-Dickson prior (A1/A2) is trustworthy. **RH-as-metaphysics ("½ is my neutral band") stays KILLED** — additive-identity-zero ≠ roots-of-ζ; same word, unrelated objects.
- **Search-space use:** essentially none operationally; kept as a note that the tower-placement is correct, not as a thread to pull.

### A11. 1.585 bits = the trit
- **Mapping:** log₂(3) = ternary information content = BitNet b1.58's name = TSM's per-commit cost.
- **Fits if:** (always, definitionally — this is structural).
- **Breaks if:** n/a as a definition; **the *derived* form to keep is "prediction bandwidth = bits-per-commit ÷ tick-time"** (a real rate), compared across the biological system and TSM — *not* "the time equals the bit," which is **KILLED** (units).
- **Search-space use:** gives a real cross-system quantity (bits/s of prediction) to compare brain vs bench, instead of a numeral coincidence.

### A12. Music & color — continuous-held-by-discrete
- **Mapping:** frequency (continuous) held by a discrete generator-and-carry: music generator = 3 (the fifth, 3:2), carry = 2 (the octave), with a never-closing residual (Pythagorean comma, ~23.5 cents, like the pi-floor); color = continuous spectrum → 3-basis (trichromacy, genuinely 3D by Grassmann).
- **Fits if:** the "continuous register held by a small-integer generator with a never-closing residual" shape keeps appearing where TSM predicts a continuous register.
- **Breaks if:** the perceptual "3" turns out to be the *same* 3 as the Hurwitz depth-3 — **test: the 4-cone bird.** A tetrachromat almost certainly still has the ~3 depth ceiling, which would prove the perceptual-3 and the algebraic-3 are *different threes* (rhyme, not identity).
- **What the break teaches:** which "3"s in the map are one structural 3 vs separate phenomena sharing a numeral — a direct guard against the numerology gravity well.
- **Search-space use:** flags that *multiple distinct 3s exist* and must not be fused; keeps the map honest.

### A13. Planck scale — secondary conjecture
- **Mapping:** the resolution floor (carry-right terminates) ↔ a physical smallest length.
- **Fits if:** **gated** — only if the pi-floor (A9) is real *and* a principled bridge from operational-floor to physical-floor is found (none exists now).
- **Breaks if:** the bench floor (A9) is finitely closable (then there's no floor to be physical).
- **What the break teaches:** whether "resolution bottoms out" is an epistemic property of inference or a physical property of spacetime (and Planck-discreteness is itself an *unsettled* physics question — LQG yes, strings no).
- **Search-space use:** essentially none until A9 resolves; ranked secondary, conditional, conjectural. Kept on the table, off the critical path.

### A14. Criticality — the physics arm's one real prediction
- **Mapping:** the invariant-alphabet/dimensional claim requires scale-invariance; scale-invariance is physically a critical phenomenon.
- **Fits if:** TSM's internal dynamics show power-law / scale-free statistics (the critical-brain signature).
- **Breaks if:** the system runs well far from criticality.
- **What the break teaches:** whether the dimensional/positional keystone is physically real or a formal convenience.
- **Search-space use:** gives a *measurable internal signature* to watch for (§8.7), and is the one place the physics arm and the bench become the same test.

---

## The scope line (kept explicit so the register stays a tool, not a creed)

This register is **scaffolding for refining TSM**, not a Theory of Everything and not a claim that downstream work is pre-solved. Two things stay true no matter how many mappings hold:

1. **A unifying shape does not pre-compute the contents.** Least action is the shape of mechanics and pre-solved zero orbits; a frame tells you where to look, never what you'll find. Every mapping that holds still leaves the integral to do.
2. **The verification program (consolidation §8) is the ground truth.** These addenda are downstream of it and cannot move it. They earn weight only by constraining the search or predicting a misfit; they lose it the moment they have to be contorted to fit.

**The intended workflow:** hit a wall → check the register for a candidate mapping → if it breaks the wall (predicts the mechanism), keep it and narrow; if it doesn't, **symmetry-break** — drop it, open a new distinction, re-run. The register's value is exactly its falsifiability: a broken mapping is a crossed-off region of the search space, which is progress.

**First, still:** Probe A′. The register tells you where to look; A′ tells you whether there's anything there to find. Map freely — and let the probe decide which of these addenda were describing the territory and which were describing the map.
