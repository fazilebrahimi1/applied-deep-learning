---
title: "Week 2 Memo Rubric"
subtitle: "Controlled Improvement and Error Analysis"
author: "ECBS5200 — Practical Deep Learning Engineering"
titlepage: false
toc: false
geometry: margin=1in
---

# Week 2 Technical Note — Rubric

**ECBS5200 — Practical Deep Learning Engineering for Applied ML**

**Deliverable:** Week 2 Technical Note
**Format:** 2-3 pages maximum (not counting tables, figures, or experiment log)
**Total points:** 100

## Overview

The Week 2 Technical Note demonstrates that you can run controlled experiments, interpret the accuracy-F1 trade-off from class weighting, test whether batch size matters, analyze where your model fails, and assess whether your metrics are trustworthy. Engineering judgment — including knowing when a result is too small to trust — matters more than hitting specific numbers.

## Rubric

### 1. Experiments: What You Tried and What Happened (20 points)

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18-20 | Reports a clear experiment log with at least 3 controlled experiments. Each experiment changes exactly one variable and documents what was held constant. Results are presented in a comparison table with accuracy, macro F1, and zero-F1 classes. The student identifies which interventions produced meaningful differences and which did not. |
| **Satisfactory** | 12-17 | Reports experiments and results but the experimental design is unclear (e.g., multiple variables changed at once) or the comparison is incomplete (e.g., missing metrics, no table). May not distinguish meaningful from noise-level differences. |
| **Needs Improvement** | 0-11 | Missing experiment log, or experiments are described without quantitative comparison. No evidence of controlled methodology. |

**What we're looking for:** Evidence of disciplined experimentation — one variable at a time, everything else held constant, results compared systematically. The experiment template fields (variable changed, held constant, prediction, result, meaningful?) should be visible in the reasoning even if not literally present.

### 2. Class Weighting: Did It Help? What Did It Cost? (20 points)

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18-20 | Reports the class weighting result with both accuracy AND macro F1. Explains the trade-off clearly: accuracy drops because the model predicts rare classes more often (some incorrectly), while F1 improves because classes that had F1=0 now have nonzero F1. Notes the number of classes rescued from zero. May compare sqrt-inverse to sklearn balanced. |
| **Satisfactory** | 12-17 | Reports the result and notes that accuracy dropped while F1 improved, but the explanation of WHY is vague or incomplete. May not mention zero-F1 classes or the trade-off mechanism. |
| **Needs Improvement** | 0-11 | Reports only one metric, or reports both without interpreting the divergence. Does not explain the trade-off. |

**What we're looking for:** Understanding of the accuracy-F1 paradox — not just "accuracy went down and F1 went up" but WHY that happens and whether the trade-off is worthwhile.

### 3. Batch Size: Did It Matter? Why Might It? (20 points)

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 18-20 | Reports the batch size comparison as a controlled experiment (one variable changed). Compares the effect magnitude to the validation noise floor and honestly assesses whether the difference is trustworthy. If the effect is small, says so rather than overclaiming. Offers a plausible mechanism (e.g., more optimizer steps per epoch gives rare classes more gradient influence) while acknowledging uncertainty. |
| **Satisfactory** | 12-17 | Reports the comparison and notes the effect direction, but does not compare to the noise floor. May overclaim from a single-run comparison or present a mechanism as settled fact. |
| **Needs Improvement** | 0-11 | Missing batch size experiment, or reports a result without any interpretation or comparison to noise. |

**What we're looking for:** Honest assessment of a potentially small effect. A student who reports "the difference was X, which is within/above the noise floor, so I can/cannot confidently say batch size matters" earns more credit than one who claims batch size is important based on a 0.5-point F1 difference.

### 4. Error Analysis: What Does the Confusion Matrix Reveal? (25 points)

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 22-25 | Computes per-class F1 by frequency tier (head/mid/tail) and shows the performance gradient. Builds a confusion matrix and categorizes where tail-class errors land (head vs mid-tier vs other tail). Identifies the dominant pattern (e.g., tail errors mostly go to semantically similar mid-tier classes). Inspects hard examples and identifies at least one structural pattern in the model's confident wrong predictions. |
| **Satisfactory** | 15-21 | Computes per-class F1 and notes the frequency-performance correlation, but confusion analysis is incomplete or superficial. May not categorize error destinations or inspect hard examples. |
| **Needs Improvement** | 0-14 | Missing error analysis, or reports only aggregate metrics without any per-class or confusion analysis. |

**What we're looking for:** Evidence that the student looked BEYOND the aggregate metrics. The confusion matrix analysis should reveal something about the model's failure mode — is it a frequency problem or a similarity problem? Hard-example inspection should show at least one concrete pattern.

### 5. Val Set Reliability: How Trustworthy Are Your Numbers? (15 points)

| Level | Points | Criteria |
|---|---|---|
| **Excellent** | 13-15 | Counts validation examples per class and identifies the sparse classes (17 with exactly 1 example, 38 with ≤5). Connects this to macro F1 reliability: single-example classes contribute coin-flip F1 to the average. Revisits experiment comparisons and assesses whether observed differences are larger than the noise floor. |
| **Satisfactory** | 8-12 | Notes that some classes have few validation examples but does not quantify the effect on macro F1 stability. May not connect the sparsity to the trustworthiness of experiment comparisons. |
| **Needs Improvement** | 0-7 | Missing, or mentions "noisy metrics" without any supporting analysis. Does not examine per-class validation counts. |

**What we're looking for:** Epistemic honesty. The student should demonstrate that they understand their metrics have a noise floor and can estimate where that floor is. A student who says "my batch size experiment showed a 1-point improvement but that's within the noise from 17 coin-flip classes" is showing real engineering judgment.

## General Notes

- **Conciseness is valued.** A tight 2-page memo that covers all five sections clearly will score higher than a 5-page memo that buries insights in filler.
- **Exact numbers will vary.** Different random seeds produce different results. We grade reasoning about YOUR results, not whether you hit specific targets.
- **Controlled methodology matters.** Explicit credit for: changing one variable at a time, distinguishing signal from noise, and not overclaiming from single runs.
- **Tables and figures do not count toward the page limit.** Use them.
- **AI tools are allowed** for coding and experimentation, but you must understand and be able to explain everything in your memo.
