---
title: Syllabus
layout: default
nav_order: 2
---

# ECBS5200 — Data Science 4: Practical Deep Learning Engineering for Applied ML (2 credits)

**Schedule:** 6 Wednesdays · two blocks each day (13:30–15:10 and 15:30–17:10)  
**Dates:** Apr 8, Apr 15, Apr 22, Apr 29, May 6, May 13 (2026)  
**Location:** CEU Vienna, QS D-002 Tiered (except Apr 15: QS B-511)  
**Instructor:** [Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/) · D-207 · Office hours by appointment · rubiae@ceu.edu

---

## Background and overall aim

This course teaches students to treat a model as a **trainable, inspectable, compressible artifact** rather than as a remote service endpoint. Students work individually on one cumulative deep-learning problem over six weeks and learn to fine-tune, improve, adapt, analyze, compress, and justify a model under realistic constraints of **quality, latency, memory, cost,** and **operational usefulness**.

Throughout the course, students compare their encoder against an instructor-provided **decoder reference system** — a stronger but far more expensive model — and learn to reason about when the quality gap justifies the cost and when it does not.

The course is designed as a professional bridge from prior machine-learning coursework and recent exposure to LLM application development toward the kind of technical ownership expected in junior applied ML roles. Class time prioritizes **training discipline, failure diagnosis, experiment comparison, compression trade-offs, cross-regime operating-point reasoning,** and **engineering judgment** over mathematical derivations or prompt-centric workflows.

---

## Course prerequisites

* Working knowledge of **Python** and basic machine-learning concepts (train/validation/test, common metrics, overfitting, class imbalance)

* **Git** basics and comfort using Jupyter/Colab or Kaggle notebooks

* Prior coursework in machine learning is required. Prior exposure to building LLM applications in code is helpful but not required.

**Pre-work (2–3 hours, due before Apr 8):** short modules on tokenization and truncation, pretrained encoders for classification, train vs. eval mode, confusion matrices and macro-F1, calibration at a practical level, LoRA/PEFT basics, quantization basics, and the basic idea of distillation. A short readiness quiz checks prerequisite understanding.

---

## Waiting list handling

MS in Business Analytics students have first priority. Other students may be admitted space-permitting, provided prerequisites are met.

---

## Learning outcomes

By the end of the course, students will be able to:

1. Audit a labeled text dataset for **label structure, class imbalance, and obvious noise**, and define a defensible supervised task.

2. Fine-tune a pretrained text model in **PyTorch**, save and compare checkpoints, and explain training behavior using learning curves and validation metrics.

3. Improve a model through controlled changes to **optimization, regularization, and training setup**, and distinguish real improvement from noise.

4. Apply **PEFT/LoRA** to adapt a pretrained model under compute constraints and justify the trade-off relative to standard fine-tuning.

5. Perform serious **error analysis** using per-class metrics, confusion patterns, slices, and calibration-aware reasoning, compare failure modes against a decoder reference system, and use that analysis to choose among candidate models.

6. Benchmark a model under runtime constraints, apply **quantization**, evaluate whether compression disproportionately harms the model’s weakest slices, and compare the resulting operating point against a decoder alternative on **cost, latency, and quality**.

7. Train a smaller student model from precomputed teacher outputs, compare **distillation** against other compression options, and justify a final operating-point recommendation — including cross-regime reasoning about encoder vs. decoder trade-offs — in both technical and stakeholder-facing terms.

---

## Learning activities and teaching methods

* Short lectures focused on engineering decisions and trade-offs

* Hands-on labs each week built around one cumulative individual model line

* Weekly **closed-book quizzes** (beginning Week 2) to verify understanding of the prior week’s work

* Weekly **tight technical memos** and logged experimental artifacts

* Final **written model decision dossier** and cumulative **written exam**

**Software/compute:** Python, PyTorch, Hugging Face (Transformers/Datasets/PEFT), scikit-learn, and a lightweight experiment logging setup (e.g. Weights & Biases, MLflow, TensorBoard, or instructor-specified structured logs). Labs are designed for **free-tier GPU notebooks** (Kaggle / Colab–class environments) with checkpointing and resumable runs as first-class requirements.

**Coding tools:** Agentic coding tools and AI coding assistants are **allowed but not required**. Students remain fully responsible for understanding, explaining, debugging, and justifying all submitted work.

---

## Assessment

* **Weekly technical memos + logged artifacts (best 5 of 6): 45%**  
* **Weekly closed-book quizzes (best 4 of 5, Weeks 2–6): 15%**  
* **Final model decision dossier: 15%**  
* **Final cumulative exam (closed book, one cheat sheet allowed): 20%**  
* **Participation / professionalism: 5%**

Late work: −10% per day. All submissions must disclose data sources, licenses, and any AI assistance.

---

## Course contents (weekly plan)

### Week 1 — Fine-tuning a real model and auditing the data (Apr 8)

* **Lecture:** From prior ML coursework to practical deep learning engineering. Dataset audit, label distribution, sample inspection, train/validation/test discipline, pretrained encoders for classification, and the anatomy of a supervised fine-tuning loop. Includes a **motivating benchmark** showing what a stronger decoder-style reference model achieves on the same task — and what it costs in latency and compute.

* **Lab:** Work on a shared business-facing text classification task. Inspect the data, review a simple classical baseline, fine-tune a pretrained classifier in PyTorch, save checkpoints, and inspect the first learning curves and validation results.

* **Deliverable:** **Week 1 Technical Note**: data audit, baseline vs. neural comparison, best checkpoint, and first logged run table.

### Week 2 — Controlled improvement and first-pass failure analysis (Apr 15)

* **Lecture:** Deliberate model improvement: learning rate, scheduler/warmup, weight decay, dropout, gradient clipping, early stopping, class weighting, and reproducibility. Early failure analysis through confusion matrices, per-class metrics, and hard-example review.

* **Lab:** Diagnose a flawed run, implement 1–2 controlled interventions, compare original vs. improved runs, and perform first-pass confusion analysis on the shared task.

* **Deliverable:** **Week 2 Improvement and Failure Memo**: experiment table, explanation of what changed and why, and first-pass confusion analysis.

### Week 3 — Parameter-efficient adaptation with LoRA / PEFT (Apr 22)

* **Lecture:** Why parameter-efficient adaptation exists. LoRA/PEFT concepts, trainable parameter counts, memory/runtime trade-offs, and when efficient adaptation is preferable to brute-force fine-tuning.

* **Lab:** Implement a LoRA/PEFT path on the course task, compare it to the prior best model, and examine the trade-off in trainable parameters, memory, wall-clock behavior, and quality.

* **Deliverable:** **Week 3 Adaptation Report**: LoRA configuration, comparison against the prior best model, one logged ablation, and a short explanation of what LoRA changed and what it bought.

### Week 4 — Deep error analysis, slices, calibration, and comparative decoder reasoning (Apr 29)

* **Lecture:** Beyond headline metrics. Slice analysis, class-frequency effects, hard-example review, thresholding where relevant, calibration at a practical level, and data-centric interpretation of model failure. Introduction to **comparative regime analysis**: how and why the decoder reference system fails differently from the encoder.

* **Lab:** Analyze failure patterns in depth, define and evaluate the course’s required slices, and make an evidence-based model recommendation before compression. Complete a **bounded decoder comparison exercise**: given encoder and decoder predictions on a hard slice, diagnose where each model wins, where each fails, and explain what this reveals about the two regimes.

* **Deliverable:** **Week 4 Error and Selection Dossier**: slice analysis, calibration/threshold reasoning where applicable, a **regime comparison subsection** analyzing encoder vs. decoder failure modes on at least one required slice, and a clear recommendation of the strongest pre-compression candidate.

### Week 5 — Runtime constraints, quantization, and decoder economics (May 6)

* **Lecture:** Turning a trained model into a usable artifact. Inference mode, latency and throughput measurement, memory footprint, checkpoint size, and quantization as operational compression. Why compression must be evaluated on the model’s weakest slices, not just on headline metrics. **Explicit comparison of encoder operating points against the decoder reference** on cost, latency, and quality.

* **Lab:** Benchmark candidate models, quantize at least one serious contender, compare quality/latency/memory trade-offs, re-run the weakest Week 4 slices on the quantized model, and compare the best encoder and quantized encoder against the decoder reference on cost and quality.

* **Deliverable:** **Week 5 Operating Point Memo**: best pre-quantization model, quantized comparison, slice degradation analysis, an **encoder vs. decoder operating-point comparison** (is the decoder’s remaining advantage worth its cost? when would you still choose it?), and a constraint-driven deployment recommendation with a brief stakeholder-facing summary.

### Week 6 — Distillation and final model decision (May 13)

* **Lecture:** Teacher–student learning as capability transfer. Hard labels vs. soft targets, temperature, blended distillation loss, and when distillation is preferable to—or weaker than—quantization. The **decoder reference system as distillation teacher**: transferring capability from an expensive decoder regime into a cheap, deployable encoder. Final model choice as an engineering decision, not a leaderboard decision.

* **Lab:** Train a smaller student model from precomputed teacher outputs (generated by the decoder reference system), compare the distilled student against the best adapted model and the quantized model, and assemble the final written model decision dossier.

* **Deliverable:** **Final Model Decision Dossier**: comparison of the best adapted model, quantized variant, distilled student, and decoder reference as external benchmark; final recommendation with **explicit cross-regime reasoning** (where the encoder is good enough, where the decoder still wins, what the gap costs, whether distillation closes enough of it); technical justification; stakeholder-facing summary; and next steps given more time or compute.

---

## Academic integrity, data governance, and ethics

Use only approved/licensed data. Clearly document provenance, splits, and any preprocessing decisions. All outside code and AI assistance must be cited.

AI coding tools are allowed, but students are individually accountable for:
* understanding what the code does,
* verifying results,
* debugging failures,
* and explaining decisions without tool assistance when assessed.

Students may not submit work they cannot explain.

---

## Contact details

* **Instructor:** [Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/) — rubiae@ceu.edu — D-207 — office hours by appointment


---

### Notes on scope and overlap (for CEU coordination)

This course emphasizes **post-training model ownership**: supervised fine-tuning, disciplined comparison, PEFT/LoRA, error analysis, quantization, distillation, cross-model-regime reasoning, and final engineering trade-offs. A decoder-style reference system is used as a comparative benchmark and distillation teacher — students do **not** train or fine-tune decoder models. The course does **not** spend class time on prompt engineering, agent frameworks, MCP/tool orchestration, chatbot construction, or broad LLM application development workflows; those topics are addressed elsewhere in the curriculum. It also does **not** attempt frontier-scale pretraining, RLHF pipelines, or broad multimodal generation.
