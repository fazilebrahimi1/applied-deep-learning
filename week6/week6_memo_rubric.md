---
title: "Week 6 Memo Rubric"
subtitle: "Distillation — Capacity, Calibration, and the Data Ceiling"
author: "ECBS5200 — Practical Deep Learning Engineering"
titlepage: false
toc: false
geometry: margin=1in
---

# Week 6 Technical Note — Rubric

**ECBS5200 — Practical Deep Learning Engineering for Applied ML**

**Deliverable:** Week 6 Technical Note (final memo)
**Format:** 2–3 pages maximum (not counting tables, figures, or experiment log)
**Total points:** 100

> **Status:** All bands refined after homework integration test on
> Kaggle T4 (2026-05-05). Excellent bands describe mastery; Satisfactory
> and Needs-Improvement bands describe the specific failure modes
> anticipated based on prior-week patterns and the Week 6 mechanism
> work. May be re-calibrated after live cohort submissions.

## Overview

The Week 6 Technical Note is the closing memo of the term. It tests
whether you can **name the model property your deployment actually
depends on**, evaluate candidate recipes against that property
(including cheap post-hoc baselines, not just KD), defend a recipe
choice grounded in your own measured numbers, and synthesize the
term's three-week compression arc (label, weight, knowledge) into one
coherent engineering position.

The week's thesis is that **different model-improvement techniques
transfer different properties at different costs**. Distillation
transfers *distributional structure* — the relative probabilities
across all 113 classes — measurable in NLL and JS divergence but
not in ECE or argmax agreement. Temperature scaling transfers *top-1
calibration* and reproduces a substantial fraction of KD's ECE win
for ~zero training cost — but it cannot reshape the per-class
distribution. The applied ML skill is matching property to recipe,
not implementing KD.

Your memo should demonstrate you can name the property, evaluate the
recipe shortlist for your scenario, read the noise floor honestly,
and defend a specific recipe under specific constraints.

Note: this memo's Prompt 5 doubles as the seed for your **Final Model
Decision Dossier** (15% of the course grade, due before the final exam).
The Week 6 memo is the smaller, in-scope version; the dossier asks for
the full deployment-defense document.

## Rubric

### 1. What Did Distillation Transfer? (25 points)

*Focus: the property. What did KD transfer that vanilla didn't, what
remains untransferred, and what kind of property is it? Decompose with
per-tier F1, per-tier ECE, NLL, JS divergence, and per-class evidence;
articulate the mechanism in target-shape terms.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 22–25 | **Evidence:** Reports per-tier F1 Δ (distilled − vanilla) with paired-bootstrap 95% CIs from the lab. Reports per-tier ECE Δ. Reports homework Part 3 Test A (JS divergence per tier between teacher and each student): KD reduces JS but a substantial residual remains, especially in the tail. Reports homework Part 3 Test B (fraction of ECE gain reproducible by post-hoc temperature scaling) AND the metric-specific finding that NLL is *not* reproducible by temperature scaling — the gap that survives post-hoc fitting is the distributional-structure gap. **Reasoning:** Names the property KD transfers irreplaceably: **distributional structure** — the relative probabilities across all 113 classes, measurable in NLL and JS divergence but not in ECE or argmax agreement. Articulates the target-shape mechanism (CE has a one-hot target; KD has a dense teacher target — both losses produce gradients on every logit, the difference is *target shape* not gradient support). References the teacher's tail F1 (~0.198) as the upper bound on what KD could conceivably transfer for tail capacity. Bonus: cites homework Part 1 per-class breakdown to characterize whether the head-tier lift is concentrated or uniform. |
| **Satisfactory** | 16–21 | **Evidence:** Per-tier F1 Δ with CIs reported. Per-tier ECE reported. JS divergence per tier reported but not connected to the distributional-structure claim. Part 3 Test B reported but the ECE-vs-NLL split is not articulated metric-by-metric. Per-class breakdown cited but not integrated with the property argument. **Reasoning:** Mechanism described in vague information-theoretic language ("KD transfers more information") without naming target shape (one-hot vs dense) or distributional structure as the irreplaceable property. May reproduce the discarded-from-the-lab claim that "CE has gradient only on the true-class logit" — flag as a misconception in feedback; softmax-CE has gradient `(p_j − 1[j=y])` on every logit. Cites teacher tail F1 as a number but doesn't tie it to the upper-bound argument. |
| **Needs Improvement** | 0–15 | **Evidence:** Aggregate macro F1 Δ only, or per-tier without CIs. Per-tier ECE missing. JS divergence not computed. Part 3 Test B treated as "KD has slightly different ECE than vanilla+T" without engaging the metric-specific story. **Reasoning:** Treats distillation as a single-axis improvement ("KD works" / "KD didn't work") without naming an axis or a property. May claim "KD transfers calibration" without distinguishing top-1 ECE from full-distribution NLL. No mechanism articulation, or a mechanism claim that turns out to be wrong. |

**What we're looking for:** that you can name what KD transferred
*specifically* — distributional structure, measurable in NLL and JS
divergence, not reproducible by post-hoc temperature scaling — and
distinguish it from properties that *are* reproducible cheaply (top-1
ECE). A student who writes *"head F1 lifted +0.021 with CI [+0.007,
+0.036]; mid +0.019 [-0.009, +0.047] null. JS divergence to teacher
dropped from 0.42 to 0.31 on head and from 0.58 to 0.51 on tail — KD
reduces the gap but a substantial residual remains. ECE: post-hoc
temperature scaling on the vanilla student (T≈1.4) actually beats
distillation (0.025 vs 0.055) — the cheap recipe wins on top-1
calibration. NLL: distillation wins (1.33 vs 1.43) — the gap that
survives temperature scaling is the distributional-structure gap KD
transfers irreplaceably. Mechanism: temperature scaling rescales
softmax sharpness uniformly so it can fix top-1 confidence but cannot
change relative probabilities of non-top-1 classes; KD transfers the
teacher's per-class shape (the 12% chance it's class 53, 7% chance
class 89). The property KD uniquely buys is distributional shape, and
the right metric to measure it is NLL or JS divergence — not ECE."* is
showing the property-first framing the rubric rewards.

### 2. What Didn't Distillation Fix? (15 points)

*Focus: the data ceiling. Distinguish task / data / method as causes,
defend with cross-model evidence.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 13–15 | **Evidence:** Names the data ceiling explicitly — that tail F1 stays low for vanilla student (0.125), distilled student (0.130), AND teacher (0.198) — and cites the cross-model pattern as evidence that scale alone doesn't escape it. References the paired-bootstrap finding from the data-confound study (CLAUDE.md): scale beyond 395M produces no significant macro F1 advantage when training data is held constant; the train→train+test data shift moved the number more than scale did. **Reasoning:** Distinguishes three candidate causes of tail failure — (a) the task is intrinsically hard on rare categories, (b) the data is insufficient (per-class examples too few), (c) the method (KD + CE) is the wrong tool — and argues from evidence which cause(s) are most consistent with the cross-model pattern. Names additional measurements that would distinguish them: e.g., more per-class tail data via careful relabeling or synthetic augmentation; stratified resampling that gives tail equal representation; comparison against a dataset where tail classes have similar feature-space density. |
| **Satisfactory** | 8–12 | **Evidence:** Identifies the data ceiling as a phenomenon and cites the cross-model tail F1 numbers, but uses only the student-vs-teacher comparison (skips the data-confound paired-bootstrap finding from CLAUDE.md). **Reasoning:** Distinguishes data from method as causes but doesn't engage with the task-vs-data sub-question. Proposes "more data" as the fix without articulating WHICH kind of data (per-class additions for tail classes vs more total examples) or what other interventions would distinguish task-difficulty from data-insufficiency. |
| **Needs Improvement** | 0–7 | **Evidence:** Tail failure observed without cross-model context — claims distillation "didn't help the tail" without engaging that the teacher's tail F1 is also low. **Reasoning:** Treats the failure as a method failure ("KD doesn't work on long-tail") rather than a data-ceiling phenomenon. No engagement with what additional measurements would falsify the data-ceiling explanation. May confuse "data ceiling" with "task is impossible" — these are different claims. |

**What we're looking for:** intellectual honesty about a negative result.
A student who writes *"Tail F1 stayed low across vanilla (0.125),
distilled (0.130), and the 32B teacher itself (0.198). The teacher
trained on the full train+test set and could not exceed 0.20 on the
tail; the student inherits at most what the teacher has. This pins the
failure on data, not method or scale — adding 5 more parameter orders of
magnitude bought no additional tail capacity in the controlled
comparison (paired bootstrap CI [-0.008, +0.045] on Qwen3-32B vs
ModernBERT-large with data held constant). The distinguishing test
between 'task-hard' and 'data-insufficient' would be careful tail-class
augmentation: if 50 additional examples per tail class lift teacher tail
F1 above 0.30, the cause is data; if not, the cause is the task itself"*
is showing the cross-model reasoning the rubric rewards.

### 3. Hyperparameters and the Noise Floor (15 points)

*Focus: read the T_d × α grid as a measurement, separate signal from
noise using the lab's bootstrap structure as the reference.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 13–15 | **Evidence:** Reports the 6-config grid summary (T_d × α sweep + reference): macro F1, ECE, NLL, head/mid/tail F1 per config. Identifies which configs land within the bootstrap noise floor of the lab's reference (T_d=4, α=0.7) and which are outside it. **Reasoning:** Names a specific tier and a specific 95% CI from the lab as the noise floor reference (e.g., "the lab's head-tier paired bootstrap CI was [+0.007, +0.036] — width ~0.029 on a measurement of similar n; differences smaller than ~0.03 on a tier with similar sample size are noise-bound"). Observes that **the lab default doesn't dominate any single metric** in the grid — it ranks #3 on macro F1, #5 on ECE, #1 on NLL, #1 on head F1, but isn't the F1 winner. Notes that on F1 metrics the grid's spread (~0.01 on head F1, ~0.013 on macro F1) is inside the lab's CI width — so claiming a "best F1 config" on point-estimate evidence alone is overclaiming. ECE shows real spread (~0.032) outside typical noise. Ties to an interpretation: F1 is data-bounded (no hyperparameter choice escapes the ceiling) but ECE is hyperparameter-sensitive (different (T_d, α) configs produce real calibration differences). If a grid point produces a metric outside the noise floor, treats it as candidate signal but explicitly notes it would need a re-seed or reshuffle test to confirm. |
| **Satisfactory** | 8–12 | **Evidence:** Reports the grid table; identifies a "best" config on macro F1 (likely T_d=1, α=0.7 at 0.2831) without engaging that the spread between this and the worst grid config is smaller than the lab's CI width. **Reasoning:** Names the best config without the noise-floor caveat — treats the grid as a hyperparameter leaderboard rather than a measurement. May correctly note that ECE varies more than F1 across the grid, but doesn't articulate WHY (calibration is hyperparameter-sensitive, capacity is data-bounded). Lab CI may be cited but not used as a noise reference. |
| **Needs Improvement** | 0–7 | **Evidence:** Grid reported as a table only; no engagement with whether differences are real or noise-bound. **Reasoning:** Declares a winner on point estimates with no measurement discipline. No reference to the lab's bootstrap CI. May claim a config "beats" another by 0.005 on macro F1 — a difference smaller than typical bf16 nondeterminism on a single seed. |

**What we're looking for:** that you can read a hyperparameter sweep
without overclaiming. A student who writes *"The lab's head-tier paired
bootstrap CI was [+0.007, +0.036], width ~0.029 on n=5,155. My grid's
head F1 ranged from 0.61 to 0.64 — a span comparable to the CI width,
suggesting the lab default was within the noise floor and the grid
points don't clearly distinguish themselves from it on head F1. On tail
the picture is different: T_d=8, α=0.9 produced a tail F1 of [X.XXX] —
[Y points] above the lab default. With n=210 the tail CI width is
~0.06; the grid difference is/isn't clearly outside that floor and I
would treat it as candidate signal that I'd want to confirm with a
re-seed before claiming it generalizes"* is showing exactly the
disciplined hyperparameter-reading the rubric rewards.

### 4. Three Compressions, Three Lessons (15 points)

*Focus: cross-week synthesis. Identify which compressions interact and
which are orthogonal, defend with numbers.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 13–15 | **Evidence:** Names the three compressions explicitly — label compression (the merge map + MIN_CLASS_COUNT=5 filter from Week 4–5 pipeline reveal), weight compression (int8/int4 quantization from Week 5), knowledge compression (KD from Week 6). Cites at least one number from each week. **Reasoning:** Identifies which two compressions interact most strongly on this dataset and which two are nearly orthogonal, defending with a measurement. Strong answers might argue: weight compression and knowledge compression both operate on the deployment-time model and can interact through the calibration channel (a quantized distilled student may lose more ECE than a quantized vanilla student because the soft target's information is more numerically fragile); label compression operates at the data-loading stage and is largely orthogonal to either. Other defensible patterns are possible — what we grade is whether the student picks a pair, names the interaction (or non-interaction), and grounds the claim in a number. |
| **Satisfactory** | 8–12 | **Evidence:** Names the three compressions and cites at least one number per week. **Reasoning:** Treats them as a list rather than identifying which pair interacts. May claim "they all matter" or "they're all useful" without picking a pair to argue about. Cross-week numbers cited but not used to support a structural claim. Doesn't articulate which combined recipes have been measured (vanilla CE on full data) vs which are open empirical questions (quantized distilled student, distillation on the merged-then-filtered label space). |
| **Needs Improvement** | 0–7 | **Evidence:** Discusses one or two compressions only. **Reasoning:** No cross-week measurements cited. Treats Week 6 as independent of Weeks 4–5. May confuse the three compressions (e.g., calling temperature scaling a "compression" when it's a calibration fix). |

**What we're looking for:** intellectual reach across the term. The
specific pair-naming is genuinely open; we grade whether the answer is
defended with measurement rather than vibes. A student who picks an
interaction and grounds it ("weight + knowledge compression both touch
deployment-time calibration: my Week 5 numbers showed int4 ECE drift on
the encoder of +0.04 post-scaling; my Week 6 numbers show distillation
buys ~0.08 of ECE improvement; whether quantizing a distilled student
preserves that improvement is an empirical question I haven't measured
but would want to before deploying that combined recipe") is showing
exactly the cross-week reach the rubric rewards.

### 5. Defend the Deployment (30 points)

*Focus: the capstone. Synthesize lab + homework + cross-week evidence
into a defensible recommendation under specific constraints.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 27–30 | **Evidence:** Names the deployment scenario (A: high-throughput batch triage / B: regulated escalation review / C: long-tail rare-class monitoring) and its primary + secondary metrics. Recommends a specific recipe from the homework Part 2 shortlist (vanilla / vanilla + post-hoc T / distilled / one tuned grid config / the wildcard recipe). Uses **specific numbers from at least three sources**: (a) §3c threshold/coverage table (vanilla coverage @ 0.70, distilled accuracy @ 0.70, the 95% SLA threshold for each model), (b) per-tier F1 / ECE / NLL / JS divergence from the lab and homework, (c) homework Part 3 metric-by-metric findings (which recipe wins ECE, which wins NLL, which wins JS-to-teacher). **Reasoning:** **Matches metric to scenario.** Articulates which constraint binds for which recipes. If Scenario A or B selected, engages with the temperature-scaling-overshoot finding from Part 3 Test B and explains whether KD or vanilla+T wins under the scenario's primary metric. Wildcard prediction from Part 2 cited as evidence the student understands what additional configs *could* deliver. Names a specific constraint that would flip the recommendation, the alternative recipe, and the threshold of the constraint at which the flip happens. Cross-week reach: if quantization is in scope, names what Week 5's findings imply about the combined recipe. |
| **Satisfactory** | 18–26 | **Evidence:** Scenario named, recipe recommended, grounded in two evidence axes (typically F1 and one calibration number). §3c numbers referenced loosely — student writes "vanilla had lower accuracy at 0.70" without giving 0.781 and 0.832. **Reasoning:** Part 3 finding cited in a sentence but not integrated metric-by-metric with the recommendation — student notes "temperature scaling helped calibration" without distinguishing ECE (where it overshoots distillation) from NLL (where distillation wins). Wildcard slot from Part 2 either skipped or cited without justification. Constraint-flip named as a vague "if requirements changed" rather than a specific threshold + alternative. Cross-week reach absent or generic. |
| **Needs Improvement** | 0–17 | **Evidence:** Recommendation made on macro F1 alone. No §3c numbers. No reference to which scenario was chosen, or scenario-mismatch (recommends a recipe that fails the scenario's hard constraint). **Reasoning:** No engagement with whether distillation is necessary vs whether temperature scaling alone would do. No constraint-binding analysis — recommendation reads as opinion. No metric-matching to scenario. No cross-week reach. Wildcard slot ignored or treated as throwaway. |

**What we're looking for:** the synthesis moment of the term —
measurement discipline becoming engineering judgment. The strongest
answers **match metric to scenario** and engage with the metric-
specific calibration finding from Part 3 Test B. A student who picked
Scenario B (regulated escalation review) and writes *"Scenario B's
decision rule is a top-1 confidence threshold with a 95% accuracy
SLA — top-1 confidence calibration is what binds, so ECE is the
relevant metric. On ECE, vanilla + post-hoc temperature scaling
(T≈1.39, ECE≈0.025) outperforms the distilled student (ECE≈0.055).
Vanilla acc @ 0.70 = 0.781, distilled acc @ 0.70 = 0.832 (from §3c)
— distilled gives more auto-routed volume at the SLA but at the cost
of worse top-1 calibration. **My recommendation: vanilla + post-hoc
temperature scaling.** Cheaper recipe, better on the metric the
deployment consumes. The constraint that would flip my answer: if the
deployment switched to consuming the full probability distribution
(ensemble combination, downstream Bayesian step, second-class human
review for ambiguous cases), NLL becomes the relevant metric and
distilled wins by ~0.10 NLL — distillation's per-class probability-
shape transfer (visible in JS divergence to teacher: vanilla 0.42,
distilled 0.31) is something temperature scaling cannot reproduce.
**My wildcard prediction in Part 2 was distilled + post-hoc T**, which
I expect would dominate the distilled-only recipe by another ~0.02
ECE — preserving KD's NLL win while getting the cheap T fix on top.
Cross-week reach: if I also need to quantize, Week 5's findings imply
~0.04 ECE drift post-scaling on encoder int4; the vanilla+T recipe at
ECE 0.025 has more headroom under that drift than the distilled
student at ECE 0.055 does."* is demonstrating exactly the engineering
judgment the rubric rewards: matches metric to scenario, integrates
Part 3 Test B metric-by-metric, names the constraint that would flip,
defends the wildcard prediction, and makes cross-week reach concrete.

## General Notes

- **Conciseness is valued.** A tight 2-page memo covering all five sections clearly will score higher than a 4-page memo that buries insights in filler.
- **Exact numbers will vary.** Different bootstrap seeds, different hyperparameter grid points across re-runs, different Kaggle session randomness will produce slightly different results. We grade reasoning about YOUR results, not whether you match specific targets.
- **No single right answer.** The deployment recommendation, the compression-pair argument, and the hyperparameter winner are all genuinely open. We grade the quality of the argument, not the conclusion.
- **Intellectual honesty is rewarded.** A student who writes "the grid difference at T_d=8 was within my CI width and I'm treating it as inconclusive" earns credit for engineering maturity — never loses credit for caution.
- **Cross-week integration is the term capstone.** Sections 4 and 5 specifically reward students who treat Week 6 as the closing of an arc, not a standalone topic. The dossier (15% of the course grade) is the larger version of this synthesis — the memo's Prompt 5 is its seed.
- **Tables and figures do not count toward the page limit.** Use them.
- **Readings can be cited.** Hinton 2015, Stanton 2021, Hebbalaguppe 2024, Busbridge 2025 (all in `readings/week6/`) are particularly relevant. Name the author in-line; we are not grading citation format.
- **AI tools are allowed** for coding, experimentation, and drafting prose, but you must understand and be able to explain every claim in your memo.

---

*Version: 2026-05-05 (v2) — restructured for property-first framing
after reviewer feedback on homework reframe. Section 1 now expects
JS-divergence + NLL evidence (homework Part 3 Test A); Section 5 now
expects scenario-matched metric reasoning + wildcard recipe defense
(homework Part 2). Subject to recalibration after first cohort
submissions.*
