# Distillation is mostly a calibration regularizer. There's a cheaper one.

A one-parameter post-hoc fix reproduced more of distillation's ECE improvement than the distillation itself did. KD won on NLL. The recipes are not interchangeable, and which one you want depends on the metric your deployment actually consumes.

## The numbers

A ModernBERT-base student (149M params) on a 113-class long-tail consumer-complaint dataset. Two arms trained the same way — same data, same seed, same epochs, same hyperparameters — except for the loss. Vanilla was plain cross-entropy on hard labels. Distilled used Hinton-style KD against a Qwen3-32B teacher with `T_d=4`, `α=0.7`. We then took the *vanilla* model and fit a single scalar temperature on a held-out calibration fold, in the seconds it takes to run LBFGS on a logit array.

Three arms, evaluated on the same eval fold:

| | ECE | NLL | Macro F1 |
|---|---:|---:|---:|
| Vanilla raw | 0.1308 | 1.5515 | 0.2833 |
| Vanilla + temperature scaling | 0.0263 | 1.4297 | 0.2833 |
| Distilled (KD) | 0.0546 | 1.3321 | 0.2872 |

Temperature scaling dropped ECE by **0.1045**. KD dropped it by **0.0762**. The ratio is **137%**. The cheap one-parameter fix didn't just match the eighty-minute T4 training run that distilled from a thirty-two billion parameter teacher. On the calibration metric most production confidence-gating systems consume, it beat it by 37%.

KD got its win back on NLL: 1.3321 versus 1.4297 post-T-scaling. About 0.10 lower. Macro F1 moved from 0.2833 to 0.2872 — within bootstrap noise on most tiers.

## Why this happens

Temperature scaling divides every logit by a single scalar `T > 0`, then re-softmaxes. The operation has two structural properties.

**It is argmax-invariant.** Dividing all logits by the same positive scalar cannot change which logit is largest. Accuracy and macro F1 don't move. Twenty-eight point three three percent before, twenty-eight point three three percent after, every time.

**It applies the same monotone transform everywhere.** T-scaling preserves the ordering of logits and applies a single global power-like reshaping to the entire distribution: `p_i' / p_j' = (p_i / p_j)^(1/T)`. It can change how concentrated the distribution is — sharpen with `T < 1`, flatten with `T > 1` — but it cannot learn that class 53 should move toward class 12 on some examples and away from class 89 on others. The non-top-1 geometry is only globally sharpened or flattened, never selectively reshaped per example.

ECE asks whether top-label confidence tracks empirical accuracy. A uniform rescale of confidences is exactly what closes that gap, and exactly all it does.

NLL is the cross-entropy of the model's full softmax against true labels. It penalizes probability mass placed on wrong classes, not just miscalibration on the winning one. A global power-like transform can shift NLL some — the top-class probability moves — but the *example-by-example shape* of the non-top-1 mass is locked in by the underlying model. T-scaling can't reach it.

KD trains the student to match the teacher's full softmax on every example. The student sees the teacher's non-top-1 mass for each input and pulls toward it. That changes distributional shape per example, not just confidence sharpness globally. So KD has access to a lever T-scaling structurally lacks, and in this run it cashed that lever in as the NLL win.

## What this isn't

It isn't a claim that KD is useless. It's a claim that KD's calibration win — the calibration role Han Guo and colleagues connected to distillation in 2021 — is reproducible by a much cheaper tool when the downstream consumer cares only about top-1 confidence.

KD's irreplaceable property is the distributional-shape transfer. If your deployment ensembles model outputs, ranks the top-k for human review, feeds into Bayesian downstream inference, or otherwise consumes mass beyond the argmax, temperature scaling can't get you there. KD is the recipe here that has access to per-example non-top-1 mass; T-scaling cannot touch it.

If your deployment gates on a single confidence threshold — escalate-to-human-if-confidence-below-T, route-by-confidence, classification-with-rejection — top-1 calibration is the property you need, and T-scaling will get you there at zero training cost and zero inference cost.

## The Han Guo connection

Han Guo et al., RepL4NLP 2021, *An Overview of Uncertainty Calibration for Text Classification and the Role of Distillation*, drew an algebraic connection between distillation and temperature scaling: both apply a softening operation on the softmax with a related mathematical form. The distillation term `KL(P(x; θ*, T) || P(x; θ, T))` where `P(x; θ, T) = softmax(f(x; θ)/T)` is, in their words, "similar to the equation of temperature scaling." That structural similarity is what made our T-scaling result possible to predict ahead of time.

Their main empirical claim is about something different from what we measured. They asked: if I take a calibrated teacher (an ensemble + T-scaling on top), and distill it into a cheap student, how much of the teacher's calibration improvement transfers? They report that "40.8% (111.2%) of the improvements from adding ensembles (temperature scaling) as extra components in teacher models are transferred to students models via distillation." Distillation is, in their setup, a faithful conduit — about 111% of T-scaling's teacher-side calibration gain ends up in the student.

Our finding is the head-to-head complement of theirs. They show that KD transfers a teacher's T-scaling gain into a student. We show that, on this setup, you don't need to do that — you can put the temperature scaling directly on the vanilla student and reach the same or better ECE without the distillation step at all. Both observations follow from the same underlying algebraic connection their paper makes explicit; ours just lands on the cheaper recipe.

The headline ratio doesn't transfer between setups. Different teacher, different student, different task, different recipe — different number. The *direction* is robust: when the downstream metric is top-1 calibration, a fitted temperature is in the same league as a trained distillation, and which one wins on any given setup can flip.

## The engineering lesson

A deployment recommendation has to start with which property of the model your downstream consumes. Not "which model has the best F1." Not even "which is better calibrated." The single word "calibration" hides at least two different metrics that respond to different operations.

For top-1 confidence gating: try temperature scaling first. Cheap. In this run, better than KD on ECE. Argmax-invariant, so accuracy is preserved exactly. The cost of trying it is the cost of fitting one scalar on a held-out calibration fold. It's the right baseline to beat before paying for anything more expensive.

For full-distribution consumption — ensembling, top-k retrieval, distillation chains where this student becomes someone else's teacher, Bayesian downstream — knowledge distillation. KD can shape the non-top-1 mass per example, which is the thing those use cases consume. T-scaling on top of KD is still a good idea for top-1 calibration; the two recipes correct different defects and stack cleanly.

The mistake to avoid is treating "calibration" as a single number. ECE and NLL respond to different operations because they measure different things, and a recipe that wins on one doesn't have to win on the other.

## Caveats worth knowing

The 137% number is on a 149M student, a 113-class long-tail classification task, with `T_d=4`, `α=0.7`, on a Kaggle T4. The qualitative direction — temperature scaling reproduces a substantial fraction of KD's ECE gain, on metric-specific terms — should be expected on similar setups. The exact ratio should not.

T-scaling is not literally free. It needs a held-out calibration fold to fit. On small datasets, that's a real cost. A 50/50 cal/eval split is standard; on tiny datasets, that's enough overhead to think about.

And KD here is doing what its name says — transferring knowledge through the teacher's full softmax. If "distillation" in your stack actually means "train the student on the teacher's argmax-sampled outputs" — which is what most closed-API "distillation" actually is — none of this generalizes, because that recipe throws away the distributional shape KD's NLL win depends on. The distinction matters more than the news cycle implies.

## What to do with this

Before paying for distillation, run a temperature-scaled vanilla student through your deployment's calibration target. If it meets the target, skip the KD. Save the training compute for the next thing. If it doesn't meet the target and your downstream metric is NLL or top-k or anything that consumes non-top-1 mass, distillation is the recipe that can actually reach it.

The discipline is the same as the rest of the toolbox: name the property the deployment consumes, pick the cheapest recipe that delivers it, measure on your own setup.

## Reproduce

All three artifacts are on the Hugging Face Hub. Load them on a free Kaggle account and the numbers in the table above should come back to four decimals.

- Vanilla student checkpoint + val predictions: [`earino/ecbs5200-week6-vanilla-baseline`](https://huggingface.co/earino/ecbs5200-week6-vanilla-baseline)
- Distilled student checkpoint + val predictions: [`earino/ecbs5200-week6-distilled-student`](https://huggingface.co/earino/ecbs5200-week6-distilled-student)
- Teacher logits (Qwen3-32B, train+test, fp16): [`earino/ecbs5200-week6-teacher-logits`](https://huggingface.co/datasets/earino/ecbs5200-week6-teacher-logits)

---

*Numbers from Week 6 of ECBS5200, an applied deep learning course at CEU Vienna. The teacher was Qwen3-32B with a LoRA adapter on train+test combined, post-hoc temperature-scaled to T=1.25 — standard practice for KD pipelines, and if anything it gives the distilled student a calibrated head start. Both students were ModernBERT-base, 149M parameters, trained on the same train+test split for the same number of epochs, with the loss function as the only difference. Temperature scaling on the vanilla student fit a single scalar on the cal half of val (3,215 examples) and evaluated on the eval half.*

*Hinton-style KD here means the Hinton, Vinyals, and Dean 2015 recipe — soft targets via the teacher's full softmax — not the news-cycle "train on the teacher's argmax outputs." Foundational reference: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), arXiv 1503.02531. Post-hoc temperature scaling as used here is from **Chuan Guo et al. 2017** ([On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), ICML 2017) — a different Guo from the Han Guo paper cited below, worth flagging because both are central to this post. Han Guo, Pasunuru, and Bansal 2021: [An Overview of Uncertainty Calibration for Text Classification and the Role of Distillation](https://aclanthology.org/2021.repl4nlp-1.29/), RepL4NLP. Full course materials at [earino.github.io/applied-deep-learning](https://earino.github.io/applied-deep-learning/).*
