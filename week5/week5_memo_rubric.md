---
title: "Week 5 Memo Rubric"
subtitle: "Quantization — Toolbox, Measurement, Deployment"
author: "ECBS5200 — Practical Deep Learning Engineering"
titlepage: false
toc: false
geometry: margin=1in
---

# Week 5 Technical Note — Rubric

**ECBS5200 — Practical Deep Learning Engineering for Applied ML**

**Deliverable:** Week 5 Technical Note
**Format:** 2–3 pages maximum (not counting tables, figures, or experiment log)
**Total points:** 100

> **Status:** All bands sharpened after integration test on Kaggle T4 (2026-04-22). Excellent bands describe mastery; Satisfactory and Needs-Improvement bands describe the specific failure modes the integration test surfaced. May be re-calibrated after live cohort submissions.

## Overview

The Week 5 Technical Note demonstrates that you can take a working fine-tuned model, apply multiple quantization techniques on constrained hardware, measure the resulting trade-offs at tier resolution (not just aggregate), and make a defensible deployment decision against specific constraints. You are not graded on which configuration you recommend — you are graded on whether you measured honestly, whether your conclusions are supported by evidence from your own runs, and whether you can articulate where the measurement discipline stops carrying and the engineering judgment begins.

The week's thesis is that quantization is a toolbox, not a technique. Your memo should demonstrate you can reach into that toolbox, pick the right tool for a specific constraint, and justify the pick with per-tier evidence rather than folklore.

## Rubric

### 1. Accuracy-Efficiency Frontier on YOUR Models (20 points)

*Focus: reading a multi-dimensional trade-off across precisions and models, producing or referencing the Pareto frontier from your own measurements.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18–20 | **Evidence:** Reports the Pareto frontier (F1 vs latency AND F1 vs VRAM) across fp16/int8/int4 for both encoder and decoder, using specific numbers from your own homework runs. Identifies at least one Pareto-dominant point and at least one Pareto-dominated point by name. **Reasoning:** Articulates why a specific (model, precision) combination would win for a specified workload, grounded in the numeric trade-off. Acknowledges hardware dependence — e.g., notes that int8 being slower than fp16 on T4 is a consequence of bitsandbytes' mixed-precision algorithm overhead and references the LLM.int8 paper's own sub-6.7B caveat, OR cites the 2026 production stack evidence that AWQ/FP8 would change the picture. |
| **Satisfactory** | 12–17 | **Evidence:** Reports own-run numbers from the six-config summary table, but presentation is one-axis heavy — argues mostly about F1, or names the "fastest config," without constructing the frontier as a multi-axis object. No Pareto plot reproduced or referenced, or a plot is included but the memo prose doesn't read it. **Reasoning:** Identifies a "winner" on some axis but doesn't articulate why it's Pareto-dominant versus merely first on one dimension. Acknowledges hardware dependence in a generic sentence ("these numbers are T4-specific") without naming a paper, a production-stack alternative, or a specific mechanism the different stack would change. |
| **Needs Improvement** | 0–11 | Reports only aggregate macro-F1 without engaging latency or VRAM. Asserts quantization benefits ("int8 saves memory," "quantization is faster") without grounding the claim in the student's own measurements. Treats quantization as a single technique rather than a toolbox (no acknowledgment that int8/int4 are different operations with different trade-offs). May cite Week 3 numbers or the lab example in place of own-run evidence. |

**What we're looking for:** that you can read the shape of a trade-off rather than reducing it to a single number. A student who writes "int8 was 2.4× slower than fp16 at batch 32 on my T4, so at our concurrency target int8 is Pareto-dominated by fp16 — but on an H100 the same int8 kernel is still slower than fp16, which means the whole bitsandbytes int8 path is deployment-disadvantaged for this small-model regime regardless of hardware" is showing exactly the kind of engineering reasoning the rubric rewards. A student who writes "int8 saves memory" in the abstract without ever touching their own numbers has not done the work.

### 2. Where the Damage Lands — Per-Tier Analysis (25 points)

*Focus: per-tier Δacc with bootstrap CIs, interpretation of aggregate-vs-tier divergence, honest handling of noise-limited findings.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 22–25 | **Evidence:** Reports per-tier Δaccuracy (head/mid/tail) with example-level bootstrap CIs for BOTH models at int8 AND int4 vs fp16 baseline — four tier-breakdowns total. Cites specific class names from the three named classes tracked through the precisions. **Reasoning:** Identifies the pattern (e.g., decoder int4 hurts head and mid most, tail less; encoder int4 has mixed tier effects). Uses BOTH criteria to separate real effects from noise-limited ones: (a) CI structure AND (b) the count of flipped predictions behind the Δacc. A narrow CI on a tier with only a handful of flips (e.g., tail tier n=210, ~5 flips producing a 2-3 pp Δacc) is treated as directional — not claimed as a generalized effect without a reshuffle test. Articulates what the macro-F1 number alone would have hidden — the specific tier information that changes a deployment decision. |
| **Satisfactory** | 15–21 | **Evidence:** All four tier breakdowns reported from own runs. **Reasoning:** Relies on the CI-excludes-zero criterion alone — e.g., claims "encoder int4 improved the tail, the CI proves it" without engaging the flip-count layer. If the flip-count concern is mentioned at all, it's a caveat sentence rather than a load-bearing piece of the reasoning. Pattern description is accurate at the surface ("int4 hurts head more than tail") but doesn't articulate what the macro-F1 number hid. Cross-model comparison present but loose — doesn't identify structural differences between encoder and decoder tier patterns. Named-class anchor may be included but treated as decoration rather than integrated with the tier-level claim. |
| **Needs Improvement** | 0–14 | Aggregate macro-F1 only; no per-tier breakdown. OR: per-tier reported at one precision only (int4 but not int8, or vice versa). OR: CIs computed but not interpreted — numbers reported without engaging what the interval means. Overclaims unsupported by the evidence (e.g., "int4 devastated the tail on both models" when the decoder tail CI crosses zero). Treats the per-tier view as a checkbox rather than the measurement discipline the week is about. |

**What we're looking for:** the measurement discipline the week is explicitly about. A student who writes "tail-tier Δacc on the encoder at int4 was +2.8% with 95% CI [+0.5%, +5.2%] — narrow, but excludes zero — while head-tier was −2.8% with tight CI. In absolute terms this is 6 examples moving across 210 tail predictions, so I'd characterize this as real but modest, and I would not claim it generalizes without a reshuffle test on a different val split" is showing graduate-level noise-aware reasoning. This section carries the most weight because it trains the skill that transfers to any production deployment.

### 3. Calibration Under Compression (15 points)

*Focus: ECE pre- and post-temperature-scaling across precisions, Week-4 connection, deployment implications of calibration drift.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 13–15 | **Evidence:** Reports ECE at fp16/int8/int4 pre-scaling AND post-scaling for both models (a 2×3 table, or equivalent prose treatment of six measurements). **Reasoning:** Identifies whether temperature scaling recovers calibration under quantization, and interprets the residual. If T-scaling recovers ECE fully, notes the implication for deployments that use confidence thresholds. If T-scaling only partially recovers, articulates whether this is acceptable for a specified deployment context. References Week 4's calibration work — the student's own numbers from Week 4 ECE measurement, or the framework they built there. |
| **Satisfactory** | 8–12 | **Evidence:** Reports ECE pre AND post for most configs but misses the full 2×3 structure (e.g., reports fp16 + int4 but skips int8, or reports post-scaling only). Homework split-val (Part 3) finding is present but brief. **Reasoning:** Notes T-scaling "worked" or "didn't work" without quantifying the post-scaling residual against the fp16 baseline. Deployment implication is generic ("calibration matters for confidence thresholds") rather than tied to a specific number and a specific gate (e.g., "at 0.8 confidence threshold, which fraction of predictions land there and what's their accuracy"). Week 4 connection is a sentence-level reference without integrating the Week 4 framework. |
| **Needs Improvement** | 0–7 | Missing pre-scaling or post-scaling measurements, or only one model covered. No engagement with whether T-scaling recovered calibration — the one-parameter fit is mentioned but its efficacy is not characterized. Treats calibration as trivia rather than a deployment-relevant failure mode orthogonal to accuracy. Split-val exercise from homework ignored. |

**What we're looking for:** that you understand quantization has a second failure mode beyond accuracy, and that Week 4's calibration toolkit extends naturally to Week 5's compression setting. A student who writes "decoder int4 ECE landed at 0.070 post-scaling vs 0.069 fp16 post-scaling — T-scaling fully recovered — so for a deployment that uses 0.8-confidence thresholds for human-review routing, the quantized decoder is calibration-equivalent to fp16. That wasn't obvious going in" is showing exactly the cross-week integration the rubric rewards.

### 4. Operational Decision Under Constraints (30 points)

*Focus: synthesizing Pareto + per-tier + calibration into a defensible deployment recommendation against specific constraints.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 27–30 | **Evidence:** Makes a specific (model, precision) recommendation for the stated deployment constraints — **T4 GPU, 100 requests/second sustained (≈10 ms/ex batched latency ceiling), 0.20 macro-F1 floor, 0.05 post-scaling ECE ceiling** — grounded in your own measured numbers. Explicitly names which constraints bind for which configs. If no configuration meets all four constraints, names which constraint you would relax and justifies why. **Reasoning:** Integrates at least three evidence sources (latency, per-tier F1 holding above threshold, calibration post-scaling, VRAM headroom) into the recommendation rather than making it on one axis. References the 2026 production stack namecheck — acknowledges that your T4 bitsandbytes numbers understate what AWQ on vLLM would achieve, and names what would change your recommendation if the hardware assumption changed. Cites at least one spotlight paper where it bears on the decision (e.g., AWQ for the production framing, Lee 2025 for the "quantization magnifies inherent weaknesses" caveat). |
| **Satisfactory** | 18–26 | **Evidence:** Recommendation made, grounded in one or two evidence axes (typically F1 and latency). Constraint-by-constraint check is present but incomplete — often skips VRAM headroom or treats calibration as a secondary consideration even though it may be a binding constraint. **Reasoning:** Treats the constraint envelope as a checklist to pass rather than as binding structure that *shapes* the recommendation. Does not clearly identify which constraint binds for which configs, or does so without tying to specific numbers. Paper citation present but feels pasted-in — not integrated with the argument. Hardware-dependence acknowledgment exists but is generic (e.g., "these are T4-specific") without naming what the different stack would change mechanistically. |
| **Needs Improvement** | 0–17 | Recommendation made without constraint-by-constraint evidence, or based on a single metric (typically macro-F1 alone). Treats the deployment question as opinion rather than a measurement-backed decision. Paper citation absent or unconnected to the argument. No hardware-dependence acknowledgment. May recommend a config that fails one or more stated constraints without noting the failure. |

**What we're looking for:** the synthesis moment of the week. This is where measurement discipline becomes engineering judgment. A student who writes *"Under the stated constraints only the encoder meets the ECE ceiling — post-scaling ECE for all three decoder precisions sits around 0.07, above the 0.05 ceiling, so the decoder is out on calibration at any precision. Within the encoder set, fp16 meets the latency budget with a ~4× headroom at batch 32, uses the least VRAM, and matches int4 and int8 on macro-F1 within measurement noise. I recommend encoder fp16. This answer would shift on two axes. If we raised the ECE ceiling to 0.08 (reasonable for many non-gated deployments) the decoder configs become viable, and decoder fp16 becomes the F1 winner. If we relaxed the hardware to H100 + vLLM with AWQ, the decoder int4 path likely becomes the most memory-efficient option — 2026 benchmarks show AWQ running roughly 4× faster than bitsandbytes at the model scales those benchmarks measured (Lin et al., MLSys 2024, on a 32B model — the exact ratio doesn't transfer to our 149M/494M scale, but the qualitative direction does). Under the T4-bitsandbytes constraint envelope we were given: encoder fp16."* is demonstrating exactly the kind of decision-making a senior applied ML engineer produces: binds to the constraints, names the binding ones explicitly, and makes the counterfactuals concrete. The rubric rewards this structure over any specific recommendation.

### 5. Revisiting Week 3 (10 points)

*Focus: updating a prior position with new evidence; cross-week synthesis.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 9–10 | **Evidence:** References your own Week 3 deployment memo recommendation specifically. **Reasoning:** Articulates whether the Week 5 quantization evidence reinforces, complicates, or shifts that Week 3 recommendation. Identifies which specific Week 5 findings caused the update (or the lack thereof). Names specifically what evidence would further change your mind. |
| **Satisfactory** | 5–8 | Acknowledges the Week 3 recommendation by name (encoder or decoder) but the update reasoning is loose. Claims "my view didn't change" or "my view was reinforced" without citing *which* Week 5 finding supports the claim. If the student recommended decoder in Week 3, may not engage with the decoder's Week 5 calibration failure as a genuine challenge to that position. If the student recommended encoder, may claim vindication without distinguishing noise from signal. Doesn't name what additional evidence would change their mind. |
| **Needs Improvement** | 0–4 | Ignores the Week 3 recommendation. Treats Week 5 as independent of the prior deployment question. Refuses to commit to a position without providing reasoning. OR: updates position but provides no engagement with the specific Week 5 evidence that drove the shift. |

**What we're looking for:** intellectual honesty about updating in response to evidence. A student whose Week 3 recommendation was "deploy the decoder" and whose Week 5 measurements make them MORE confident — because the decoder maintains tier-level stability under int4 that matters for their specific imagined deployment — earns full marks. So does a student whose confidence shifts sideways: "I still prefer the decoder for accuracy-critical use, but the Week 5 calibration work added a T-scaling step I hadn't previously scoped. The total deployment recipe has grown, not the architecture choice." A student who refuses to commit to a position or ignores the prior recommendation loses credit not for the conclusion but for the lack of engagement.

## General Notes

- **Conciseness is valued.** A tight 2-page memo covering all five sections clearly will score higher than a 5-page memo that buries insights in filler.
- **Exact numbers will vary.** Different bootstrap seeds, different quantization kernel versions, different batch-size fallbacks will produce different results. We grade reasoning about YOUR results, not whether you match specific targets.
- **No single right answer.** The deployment recommendation, the tier-pattern interpretation, the calibration characterization are all genuinely open. We grade the quality of the argument, not the conclusion.
- **Intellectual honesty is rewarded.** A student who writes "my int8 measurement was noisier than I expected and I'm treating the int8 vs fp16 latency gap as directional rather than precise" earns credit for engineering maturity — never loses credit for caution. A student who says "I don't know whether this pattern generalizes without a reshuffle test" with a reasoned justification is showing exactly what we want.
- **Tables and figures do not count toward the page limit.** Use them.
- **Readings can be cited.** If your argument uses a paper from `readings/week5/` (the four spotlights, or QLoRA / BF16-or-Death / Diminishing Outliers / GPTQ from the reading list), name the author in-line. We are not grading citation format.
- **AI tools are allowed** for coding, experimentation, and drafting prose, but you must understand and be able to explain every claim in your memo.

---

*Version: 2026-04-22 — all bands sharpened post integration test. Subject to recalibration after first cohort submissions.*
