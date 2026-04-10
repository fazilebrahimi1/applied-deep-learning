# Experiment Log: April 8-9, 2026

## The Question

Can we find a teacher model with 0.300+ macro F1 on our 113-class CFPB complaint classification task for Week 6 distillation? And more broadly — what actually moves the needle on long-tail classification?

## Starting Point

| Model | Params | Macro F1 | Acc | Zero-F1 | Notes |
|---|---|---|---|---|---|
| TF-IDF + LogReg | — | 0.132 | 54.2% | 70 | Instant |
| ModernBERT-base full FT (3ep) | 149M | 0.209 | 56.6% | 46 | 32 min, T4 |
| ModernBERT-base LoRA (3ep) | 3.5M trainable | 0.211 | 56.4% | 47 | 32 min, T4 |
| Qwen2.5-0.5B LoRA cls (3ep) | 2.3M trainable | 0.240 | 57.0% | 37 | 54 min, T4 |

All prior numbers were from fixed 3-epoch runs with no early stopping. We never found the actual peak.

---

## Experiment 1: LoRA Ceiling Search (Qwen2.5, more epochs)

**Hypothesis:** The 3-epoch baselines left performance on the table. More epochs with early stopping should find the true ceiling.

**Setup:** Qwen2.5-1.5B and 3B, LoRA + classification head, cosine schedule, eval every half-epoch, early stopping patience=3. Run on Colab Blackwell RTX PRO 6000 (95 GB VRAM, cc 12.0).

### Results

| Model | Batch | Best Macro F1 | Acc | Zero-F1 | Peak Epoch | Time |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 128 | 0.2505 | 57.7% | 39 | 4.0 | 29 min |
| Qwen2.5-3B | 64 | 0.2697 | 58.8% | 33 | 3.0 | 39 min |

**Finding:** The 3-epoch baselines were close to the ceiling. More epochs didn't help much. The 1.5B peaked at epoch 4, the 3B at epoch 3.

**Incident:** The 3B had a transient collapse at epoch 2.5 — val loss spiked to 3.28, accuracy dropped to 24%, then recovered to a new best at epoch 3. Likely a catastrophic batch or numerical instability in the classification head at lr=2e-4.

**Key observation:** Val loss is NOT a reliable proxy for macro F1 on long-tail tasks. Val loss started climbing well before macro F1 peaked in every run.

---

## Experiment 2: MegaTrain Paper Analysis

**Paper:** arXiv:2604.05091 — "MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU"

**Question:** Could we use MegaTrain to full fine-tune a large decoder as a teacher model?

**Analysis:** MegaTrain stores params + optimizer in host (CPU) RAM. The formula is 12 bytes/param (2B weights + 2B gradients + 8B Adam states).

| Hardware | Host RAM | Max Model |
|---|---|---|
| Kaggle T4 | 31 GB | ~1.5B (useless — LoRA already works) |
| Colab H100 | 83 GB | ~3B (marginal gain over LoRA) |
| Mac M4 | 128 GB | ~5-7B (interesting but slow) |

**Finding:** MegaTrain is designed for machines with 480GB-1.5TB host RAM. On our available hardware, LoRA already works well enough that MegaTrain's advantage (full fine-tune) is marginal. Also, MegaTrain defaults to bf16 which is broken on T4.

**Decision:** Don't pursue MegaTrain. Focus on what we can do with LoRA.

---

## Experiment 3: Qwen3-8B-Base Teacher (H100 + Mac M4)

**Hypothesis:** A newer, larger model (Qwen3-8B-Base, 7.6B params) should beat the 3B decoder.

### Attempt 3a: 4-bit QLoRA on H100 (lr=5e-5)

**Result:** Peaked at **0.226 macro F1** at epoch 6.5. Much worse than the 3B.

**Diagnosis:** Learning rate too low. At 5e-5, the LoRA adapters barely moved. The model converged to a sharp minimum that memorized but didn't generalize. Train loss dropped to 0.48 while val loss climbed to 1.91.

### Attempt 3b: bf16 on H100, no quantization (lr=5e-5)

Same lr problem, plus we discovered the 8B at batch=64 without gradient checkpointing used **91 GB of 95 GB VRAM**, causing PyTorch allocator thrashing (0.82 it/s — same speed as the 14B with grad checkpointing). Research confirmed: at 96% VRAM utilization, the caching allocator constantly fails to find contiguous blocks, calling cudaFree/cudaMalloc in a loop.

**Fix:** batch=48 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → dropped to 59 GB, speed jumped to 1.77 it/s.

### Attempt 3c: bf16 on Mac M4 128GB (lr=1e-4)

**Setup:** Qwen3-8B-Base, bf16, no quantization, no grad checkpointing, batch=16, lr=1e-4 (matching official Qwen3 LoRA recommendation from LLaMA-Factory configs). Running at ~5 s/step on MPS backend.

**Learning curve:**

| Epoch | Macro F1 | Acc | Zero-F1 | Val Loss |
|---|---|---|---|---|
| 0.5 | 0.1775 | 52.6% | 55 | 1.670 |
| 1.0 | 0.2166 | 55.9% | 44 | 1.548 |
| 1.5 | 0.2367 | 56.8% | 43 | 1.505 |
| 2.0 | 0.2417 | 58.3% | 41 | 1.471 |
| 2.5 | 0.2583 | 58.3% | 38 | 1.601 |
| 3.0 | 0.2627 | 59.4% | 37 | 1.512 |
| **3.5** | **0.2713** | 58.3% | **36** | 1.873 | **← best** |
| 4.0 | 0.2647 | 58.7% | 33 | 1.811 |
| 4.5 | 0.2585 | 57.6% | 34 | 2.239 |
| 5.0 | 0.2631 | 57.8% | 35 | 2.161 | early stop |

**Final: 0.2713 macro F1 at epoch 3.5. Early-stopped at epoch 5.0. Total wall time: 28.8 hours on Mac M4.**

Note: epoch 4.0 had the best zero-F1 count (33) but lower macro F1 (0.2647). The best-model restore used the epoch 3.5 checkpoint based on macro F1.

### Key Finding: Model Scale Doesn't Solve Long-Tail Classification

| Model | Params | Best Macro F1 |
|---|---|---|
| Qwen2.5-1.5B LoRA | 1.5B | 0.2505 |
| Qwen2.5-3B LoRA | 3B | 0.2697 |
| Qwen3-8B LoRA | 8B | 0.2713 |

Going from 1.5B to 8B (5x parameters) gained only +0.02 macro F1. The bottleneck is not model capacity — it's rare-class gradient signal.

---

## Experiment 4: Hardware Speed Testing

### Mac M4 128GB Speed Test

| Model | Batch | it/s | 1 epoch | 3 epochs |
|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 0.67 | 22 min | 67 min |
| Qwen2.5-1.5B | 64 | 0.23 | 65 min | 194 min |
| Qwen3-8B | 16 | ~0.20 | ~5 hrs | ~15 hrs |

Mac M4 is ~5x slower than T4 on the 0.5B, ~15x slower than H100 on the 1.5B. Usable for overnight runs, not for iteration.

### GPU Memory Thrashing Discovery

At 96% VRAM utilization on H100 (91 GB / 95 GB), PyTorch's caching allocator thrashes. The 8B model without gradient checkpointing at batch=64 ran at the same speed as the 14B with gradient checkpointing. Solution: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + reduce batch to 75% VRAM target.

### Qwen3 Model Availability

- `Qwen3ForSequenceClassification` exists in transformers ≥ 4.51
- Base variants: 0.6B, 1.7B, 4B, 8B, 14B (not 32B — only post-trained with thinking mode)
- The 14B-Base in bf16 (28 GB) fits on H100 with gradient checkpointing but runs at only 0.83 it/s
- The 8B-Base in bf16 (16 GB) is the sweet spot for single-GPU training

---

## Experiment 5: ModernBERT Kitchen Sink (v1)

**Hypothesis:** Classical ML techniques applied to a small encoder can match the giant decoders.

**Tricks applied:**
1. Full fine-tune (not LoRA)
2. Sqrt-inverse class weighting
3. 256 tokens (up from 128)
4. Label smoothing (0.1)
5. Cosine schedule with 15% warmup
6. Effective batch=64 (micro=16, accum=4)
7. Early stopping on macro F1 (patience=4)
8. Grad norm clipping (1.0)

**Result:** **0.2406 macro F1, 37 zero-F1 classes, 52.4% accuracy. 13 min on Colab GPU.**

+15% relative improvement over vanilla ModernBERT (0.209), but doesn't beat the decoders. Accuracy drops (52.4% vs 56.6%) because class weighting trades majority-class accuracy for rare-class F1.

---

## Experiment 6: Kitchen Sink v2 (Logit Adjustment + Confusion Analysis)

**Hypothesis:** Post-hoc logit adjustment (Menon et al., ICLR 2021) — subtracting τ·log(class_prior) from logits at inference — should further boost rare-class predictions.

**Logit adjustment sweep:**

| τ | Macro F1 | Accuracy | Zero-F1 |
|---|---|---|---|
| 0.0 | 0.2401 | 49.5% | 36 |
| 0.5 | worse | — | — |
| 1.0+ | worse | — | — |

**Finding:** Logit adjustment didn't help. Best τ was 0.0 (no adjustment). The class weighting already shifted decision boundaries — logit adjustment on top was redundant.

### Confusion Matrix Analysis (Cascade Viability)

Where do tail-class errors actually land?

```
Tail class errors:
  Confused → HEAD:      17 (25.4% of errors)
  Confused → mid:       44 (65.7% of errors)  ← the real problem
  Confused → tail:       6 (9.0% of errors)
```

Only 25% of tail errors go to head classes. 66% go to mid-tier classes — semantically similar categories that the model struggles to distinguish.

**Initial conclusion:** Cascade won't help because tail classes aren't eaten by head classes.

**But Eduardo's counter-theory:** The confusion matrix reflects representations learned under head-class gradient dominance. Remove the head classes from training, and Model B learns different representations. The errors might change.

---

## Experiment 7: Cascade Classification

**Theory (Eduardo's):** Head classes dominate gradient updates even with class weighting. Training a specialist model WITHOUT head classes gives it full representational capacity for the remaining classes.

**Architecture:**
- Model A (Router): Classify into {6 head classes with ≥2000 examples, OTHER}
- Model B (Specialist): Trained only on non-head examples (107 classes, 26K examples)
- Inference: Router predicts → head class or OTHER → if OTHER, specialist picks specific class

### v1: HEAD_MIN_COUNT=500 (too low — 21 head classes)
Killed. 21 classes made the router almost as hard as the original problem.

### v2: HEAD_MIN_COUNT=2000 (6 head classes), batch=32 specialist

**Router:** 0.695 macro F1 on 7-class problem. Zero zero-F1 classes throughout. Peaked at epoch 7.5.

**Specialist (batch=32):** Peaked at 0.226 macro F1 at epoch 5.

**Cascade combined:** 0.225 macro F1, 42 zero-F1 classes.

**Result: Worse than the flat model.** Error propagation + specialist not learning better than the flat model.

### v3: Same but with best-model restore fix + batch=4 specialist

**Eduardo's hypothesis:** Batch=32 with 107 classes still starves rare classes of gradient signal. Batch=4 gives 8x more gradient updates per epoch.

**Specialist (batch=4) learning curve:**

| Epoch | Macro F1 | Zero-F1 | vs batch=32 at same epoch |
|---|---|---|---|
| 0.5 | 0.085 | 73 | 4x better (0.021) |
| 1.0 | 0.145 | 60 | +54% (0.094) |
| 2.0 | 0.208 | 47 | +21% (0.171) |
| 3.0 | 0.241 | 39 | +17% (0.206) |
| 5.0 | 0.260 | 36 | +15% (0.226 peak) |
| **5.5** | **0.266** | **32** | **+18% (0.226 peak)** |

**Cascade combined (batch=4):** **0.2624 macro F1, 58.6% accuracy, 33 zero-F1 classes.**

### Batch Size Effect (Specialist Only)

| Specialist Config | Peak Macro F1 | Zero-F1 | Peak Epoch |
|---|---|---|---|
| Batch=32 | 0.226 | 42 | 5.0 |
| **Batch=4** | **0.266** | **32** | **5.5** |

**The plateau moved.** Batch size was a real lever — +18% relative improvement. Eduardo's batch size intuition was validated.

---

## Summary Table: Everything We Tried

| # | Model | Method | Macro F1 | Acc | Zero-F1 | Time | Hardware |
|---|---|---|---|---|---|---|---|
| — | TF-IDF + LogReg | — | 0.132 | 54.2% | 70 | instant | — |
| — | ModernBERT-base | Full FT vanilla (3ep) | 0.209 | 56.6% | 46 | 32m | T4 |
| — | ModernBERT-base | LoRA (3ep) | 0.211 | 56.4% | 47 | 32m | T4 |
| 1 | Qwen2.5-1.5B | LoRA cls, batch=128 | 0.250 | 57.7% | 39 | 29m | Blackwell |
| 1 | Qwen2.5-3B | LoRA cls, batch=64 | 0.270 | 58.8% | 33 | 39m | Blackwell |
| 3a | Qwen3-8B | QLoRA 4-bit, lr=5e-5 | 0.226 | 56.2% | 40 | — | H100 |
| 3c | Qwen3-8B | LoRA bf16, lr=1e-4 | 0.271 | 58.3% | 36 | 28.8 hrs | Mac M4 |
| 5 | ModernBERT-base | Kitchen sink v1 | 0.241 | 52.4% | 37 | 13m | Colab |
| 6 | ModernBERT-base | Kitchen sink + logit adj | 0.240 | 49.5% | 36 | 13m | Colab |
| 7a | ModernBERT-base | Cascade, batch=32 spec | 0.225 | 56.5% | 42 | 23m | Colab |
| **7b** | **ModernBERT-base** | **Cascade, batch=4 spec** | **0.262** | **58.6%** | **33** | **42m** | **Colab** |

---

## Key Findings

### 1. Model Scale Is NOT the Lever for Long-Tail Classification
Going from 1.5B → 8B parameters (5x) gained only +0.02 macro F1. The bottleneck is rare-class data scarcity, not model capacity.

### 2. Batch Size Matters More Than Model Size
The specialist with batch=4 beat the specialist with batch=32 by +18% relative macro F1. Smaller batches give rare classes more frequent gradient updates. This was Eduardo's core intuition throughout the session.

### 3. Class Weighting Helps But Has a Ceiling
Sqrt-inverse class weighting added +15% relative improvement (0.209 → 0.241). But it can't overcome the fundamental data scarcity — classes with 5-10 examples still can't learn good boundaries.

### 4. Cascade Architecture Shows Promise
Removing head classes from the specialist's training set + small batch size brought a 149M encoder (0.262) within striking distance of an 8B decoder (0.271). The cascade has room for improvement (better router, better specialist tuning).

### 5. Logit Adjustment Is Redundant with Class Weighting
Post-hoc logit adjustment (Menon et al., 2021) did nothing on top of class weighting. They target the same decision boundary shift.

### 6. Val Loss Is Not a Reliable Metric for Long-Tail Tasks
In every run, val loss started climbing well before macro F1 peaked. Use macro F1 for early stopping, not loss.

### 7. PyTorch Memory Thrashing Is Real
At >90% VRAM utilization, the caching allocator thrashes and can cut throughput by 2x. Use `expandable_segments:True` and target ~75% VRAM.

---

## Not Yet Tried (Ranked by Expected Impact)

1. **ModernBERT-large (395M params)** — Bigger encoder with kitchen sink + batch=4. Might be the simplest path to 0.300+. No cascade complexity needed.

2. **Cascade with better router** — Current router uses class weighting on a 7-class problem that doesn't need it. Higher LR, bigger batch, no weighting could improve routing accuracy and reduce error propagation.

3. **Data augmentation for rare classes** — Use Claude to generate paraphrases of the 5-20 example classes. The research says this is the highest-impact intervention for long-tail classification.

4. **Cascade with decoder specialist** — Use encoder as fast router, decoder as specialist for the hard cases. Pay decoder latency only on the ~45% of examples that aren't head classes.

5. **Triple cascade** — Router → Specialist → Micro-specialist for zero-F1 classes. Diminishing returns likely due to error propagation, but worth understanding the limit.

6. **Flat model with batch=4** — We never tested a flat 113-class model with batch=4. The cascade overhead might not be necessary if batch size alone moves the flat model's plateau.

---

## Infrastructure Notes

- **Colab GPU allocation varies:** Got Blackwell RTX PRO 6000 (95 GB) on first session, H100 on later sessions.
- **Mac M4 training is viable** for overnight runs. MPS backend works with bf16 since PyTorch 2.6. Key settings: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `dataloader_num_workers=0`, `dataloader_pin_memory=False`.
- **HF rate limiting** hit us on unauthenticated requests. Always include HF token in notebooks.
- **Qwen3-Base models** exist for 8B and 14B but NOT 32B. The 32B is post-trained only (with thinking tokens).
- **torch_dtype deprecation:** Use `dtype` instead of `torch_dtype` in `from_pretrained` to avoid warnings.

---

## Files Created

```
verification/find_lora_ceiling.py          # Exp 1: LoRA ceiling search
verification/find_lora_ceiling_v2.py       # Exp 1: batch size variant (not run)
verification/find_teacher_ceiling.py       # Exp 3: Qwen3-8B teacher
verification/modernbert_kitchen_sink.py    # Exp 5: kitchen sink v1
verification/modernbert_kitchen_sink_v2.py # Exp 6: + logit adjustment
verification/cascade_experiment.py         # Exp 7: cascade v1 (HEAD=500)
verification/cascade_experiment_v2.py      # Exp 7: cascade v2 (HEAD=2000)
verification/cascade_experiment_v3.py      # Exp 7: + best-model fix
verification/mac_speed_test/speed_test.py  # Exp 4: Mac speed test
verification/mac_speed_test/train_8b_teacher.py  # Exp 3c: 8B on Mac
```

All `.py` files have corresponding `.ipynb` notebooks generated via jupytext.

---

## Session Context

- **Date:** April 8-9, 2026
- **Course:** ECBS5200 Applied Deep Learning, CEU Vienna
- **Instructor:** Eduardo Ariño de la Rubia
- **Dataset:** determined-ai/consumer_complaints_medium, 113 classes, 57,846 train / 6,430 val
- **Triggered by:** Reading the MegaTrain paper (arXiv:2604.05091) and asking whether it could help build a better teacher model for Week 6 distillation.
