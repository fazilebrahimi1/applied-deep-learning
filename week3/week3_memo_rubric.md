---
title: "Week 3 Memo Rubric"
subtitle: "Encoder vs Decoder — The Adaptation Trade-off"
author: "ECBS5200 — Practical Deep Learning Engineering"
titlepage: false
toc: false
geometry: margin=1in
---

# Week 3 Technical Note — Rubric

**ECBS5200 — Practical Deep Learning Engineering for Applied ML**

**Deliverable:** Week 3 Technical Note
**Format:** 2-3 pages maximum (not counting tables, figures, or experiment log)
**Total points:** 100

## Overview

The Week 3 Technical Note demonstrates that you can apply LoRA to two different architectures, compare their behavior at the systems level and the class level, assess whether an intervention transfers across architectures, measure deployment cost, and make a justified recommendation for a specific operational context.

## Rubric

### 1. Systems-Level Comparison: Encoder vs Decoder (20 points)

*Focus: architecture, parameter efficiency, aggregate metrics, connection to prior weeks.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18-20 | **Evidence:** Reports accuracy, macro F1, and zero-F1 count for both models. States trainable parameter share for both (e.g., 2.3% vs 0.46%). Compares encoder LoRA to full fine-tuning results from Weeks 1-2 — did training 2% of the parameters change quality? **Reasoning:** Interpretation is precise, supported by reported numbers, and distinguishes what the aggregate metrics show from what they hide. |
| **Satisfactory** | 12-17 | Reports the comparison with at least two metrics, but omits parameter efficiency or connection to prior weeks. Interpretation is present but vague or unsupported. |
| **Needs Improvement** | 0-11 | Reports only one model, or reports both without structured comparison. No connection to the broader course arc. |

**What we're looking for:** A systems-level view — architecture, parameter budget, overall quality. This section is about "what are these two models and how do they compare at 30,000 feet?" Leave per-class analysis for Section 3.

### 2. Class Weighting on the Decoder (20 points)

*Focus: whether an intervention that worked on the encoder transfers to the decoder, and why or why not.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18-20 | **Evidence:** Reports the class weighting result on the decoder with accuracy, macro F1, and zero-F1 count. Compares effect sizes quantitatively — how large was the macro F1 change on the decoder vs how large was it on the encoder in Week 2? **Reasoning:** Offers a mechanistic explanation for why the intervention transferred differently across architectures, grounded in observed results. The explanation engages with what each model already knew before class weighting was applied. |
| **Satisfactory** | 12-17 | Reports the result and notes it differs from Week 2, but the explanation is vague ("it didn't help as much") or the comparison lacks magnitude (doesn't say by how much it differed). |
| **Needs Improvement** | 0-11 | Missing the class weighting experiment, or reports the result without any comparison to the encoder or any attempt at explanation. |

**What we're looking for:** The student should grapple with *why* the same trick had different effects. We do not require a specific causal explanation — we require that the explanation is grounded in observed evidence, not speculation disconnected from the data.

### 3. Per-Class Analysis: Rare vs Common (25 points)

*Focus: distributional diagnosis — where do models differ, not just how much.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 22-25 | **Evidence:** Breaks down performance by frequency tier (rare vs common). Reports mean F1 and zero-F1 counts per tier for both models. Notes qualitative differences — e.g., whether entire class tiers go to zero F1 on one model but not the other. Uses the scatter plot or per-class data to identify specific patterns. **Reasoning:** Claims about where the models differ are supported by tiered or class-level evidence, not just aggregate metrics. |
| **Satisfactory** | 15-21 | Reports per-class or per-tier metrics but the analysis stays surface-level — notes a pattern without supporting it with specific numbers or examples from the data. |
| **Needs Improvement** | 0-14 | Missing per-class analysis, or reports only aggregate metrics. No evidence of looking beyond accuracy and macro F1. |

**What we're looking for:** This section carries the most weight because the aggregate metrics hide the most important story. We want evidence that the student looked at *where* the models differ and supported their claims with specific data from their own run. A student who just repeats the aggregates and says "the decoder is better" has missed the week.

### 4. Latency and the Deployment Recommendation (20 points)

*Focus: translating quality and speed into a situated engineering recommendation.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18-20 | **Evidence:** Reports concrete latency numbers for both models (single-example, batched, or both). Translates the speed difference into practical terms (throughput at scale, cost, SLA implications). **Reasoning:** Makes a deployment recommendation for a specific, named operational context (e.g., real-time triage vs nightly batch vs compliance audit). Defends the recommendation with evidence and acknowledges what would change the answer. |
| **Satisfactory** | 12-17 | Reports latency numbers but does not translate them into operational terms. May give a recommendation without tying it to a specific use case, or give a generic "it depends" without specifying on what. |
| **Needs Improvement** | 0-11 | Missing latency analysis, or gives a recommendation with no supporting evidence. |

**What we're looking for:** An engineering judgment call, not a right/wrong answer. Either model can be the right choice. A student who recommends the encoder for one context and the decoder for another — with numbers — is showing exactly the reasoning we want. A student who recommends the decoder because "it has higher F1" without engaging with cost has missed the deployment dimension entirely.

### 5. What You'd Do Differently (15 points)

*Focus: forward-looking reasoning informed by observed results.*

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 13-15 | **Evidence:** Proposes a specific next experiment with a clear hypothesis and prediction. Identifies what they believe is the most plausible bottleneck on this task, citing evidence from their own runs. **Reasoning:** The proposed experiment targets the identified bottleneck. The student explains why they chose this over alternatives. |
| **Satisfactory** | 8-12 | Proposes a next step but it is vague ("try a bigger model") or disconnected from what they observed. May identify a bottleneck without supporting evidence from their runs. |
| **Needs Improvement** | 0-7 | Missing, or proposes something with no connection to the results (e.g., "use GPT-4" without reasoning). |

**What we're looking for:** The student's proposal should be informed by what they learned — not a generic suggestion, but something that addresses a specific gap they identified in their own experiments. We grade the reasoning chain, not whether they arrived at a particular answer.

## General Notes

- **Conciseness is valued.** A tight 2-page memo that covers all five sections clearly will score higher than a 5-page memo that buries insights in filler.
- **Exact numbers will vary.** Different random seeds and training runs produce different results. We grade reasoning about YOUR results, not whether you hit specific targets or see specific patterns.
- **No single right answer.** The deployment question, the class weighting explanation, and the bottleneck diagnosis are all genuinely open. We grade the quality of the argument and the evidence behind it, not the conclusion.
- **Intellectual honesty is rewarded.** A student who notes that an observed difference is small relative to run-to-run variation, or who qualifies a claim because their evidence is limited, earns credit for engineering maturity — never loses credit for caution.
- **Tables and figures do not count toward the page limit.** Use them.
- **AI tools are allowed** for coding and experimentation, but you must understand and be able to explain everything in your memo.
