# Applied Deep Learning

**A six-week graduate course in post-training deep learning engineering — fine-tune, adapt, analyze, compress, and ship a real model under real constraints.**

Course materials for **ECBS5200** at Central European University's MS in Business Analytics program.

> **🌐 Live site (recommended starting point):** [earino.github.io/applied-deep-learning](https://earino.github.io/applied-deep-learning/)
>
> Slides, lab notebooks, homework notebooks, memo rubrics, readings, and instructor blog posts — all browsable in one place.

---

## What this course is

Most ML courses teach you to use models as service endpoints — prompt them, chain them, build apps around them. This course teaches you to treat a model as a **trainable, inspectable, compressible artifact you own**.

Over six weeks, students work on one cumulative problem: classifying 113 consumer-financial-complaint categories with extreme class imbalance (2,666:1 head-to-tail ratio). Each week adds one engineering capability — fine-tuning, controlled improvement, parameter-efficient adaptation, error diagnosis, quantization, distillation. The capstone is a written deployment recommendation defended against fixed constraints: *is the cheap model good enough, or is the expensive one worth the cost?*

**The course is taught entirely on free-tier Kaggle T4 GPUs.** No cloud spend required to follow along.

---

## What students leave with

- Fine-tuning a pretrained transformer (encoder + decoder) on a hard long-tail classification task
- Per-tier evaluation (head / mid / tail) instead of aggregate metric chasing
- Bootstrap confidence intervals on every comparison — point estimates aren't claims
- LoRA and QLoRA adaptation, with measurement of when they help and when they don't
- ECE, temperature scaling, and calibrated confidence gating for deployment
- int8 / int4 quantization through bitsandbytes, plus the 2026 production stack (AWQ, FP8, GGUF) by name
- Knowledge distillation via Hinton-style soft targets, with the cheaper alternatives (T-scaling, label smoothing) measured head-to-head
- Reading and writing measurement-driven memos defensible against the rubric
- The discipline of *naming the property your deployment consumes, then picking the cheapest recipe that delivers it*

---

## Three headline findings from the course

The students don't just read about these — they measure them on their own runs and write deployment memos against them.

| Finding | Where it lives |
|---|---|
| **Scale doesn't beat data composition on this long-tail task.** Qwen3-32B vs ModernBERT-large with training data held constant: paired-bootstrap 95% CI [-0.008, +0.045] includes zero. | [Blog #8: You Can't Scale Your Way Out of a Data Problem](https://earino.github.io/applied-deep-learning/site/blog/08_cant_scale_out_of_data_problem.html) |
| **bitsandbytes int8 is 2.5× slower than fp16 on sub-7B models on T4.** Peak VRAM goes *up* by 44–63%, not down. The LLM.int8 paper itself documents this — tutorials skip it. | [Blog #11: Your int8 Quantization Is 2.5× Slower Than fp16](https://earino.github.io/applied-deep-learning/site/blog/11_int8_is_slower.html) |
| **Post-hoc temperature scaling reproduces 137% of distillation's ECE gain.** KD only wins on NLL — because T-scaling is argmax-invariant and uniformly reshapes, it can't touch non-top-1 mass per example. | [Blog #12: Distillation is mostly a calibration regularizer](https://earino.github.io/applied-deep-learning/site/blog/12_temperature_scaling_beats_distillation.html) |

The full set of [twelve instructor blog posts](https://earino.github.io/applied-deep-learning/site/blog/) documents the build process, the surprises, and the engineering lessons.

---

## Key information

| | |
|---|---|
| **Instructor** | [Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/), Professor of Practice |
| **Format** | 6 Wednesdays, two 100-minute blocks per day (lecture + lab) |
| **Dates** | Apr 8, 15, 22, 29, May 6, 13 (2026) |
| **Location** | CEU Vienna Campus |
| **Compute** | Free-tier GPU notebooks (Kaggle T4) |
| **Base model** | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M params, Apache 2.0) |
| **Decoder for comparison** | [Qwen 2.5 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| **Distillation teacher** | Qwen3-32B with LoRA adapter ([on HF Hub](https://huggingface.co/earino/ecbs5200-week6-teacher-logits)) |
| **Dataset** | [Consumer Complaints (Medium)](https://huggingface.co/datasets/determined-ai/consumer_complaints_medium) — 113 classes, 85,708 examples |

---

## Course schedule

| Week | Topic | Date | Materials |
|:---:|:---|:---:|:---:|
| — | [Pre-work](pre-work/) (~2-3 hours, 8 modules) | Before Apr 8 | ✅ |
| 1 | [Fine-tuning and data audit](week1/) | Apr 8 | ✅ |
| 2 | [Controlled improvement and failure analysis](week2/) | Apr 15 | ✅ |
| 3 | [Parameter-efficient adaptation (LoRA/PEFT)](week3/) | Apr 22 | ✅ |
| 4 | [Error diagnosis: slices, calibration, cross-model analysis](week4/) | Apr 29 | ✅ |
| 5 | [Quantization: toolbox, measurement, deployment](week5/) | May 6 | ✅ |
| 6 | [Distillation and final model decision](week6/) | May 13 | ✅ |

Each week contains:

- **Slides** (HTML + PDF) — presented in lecture block
- **Lab notebook** — completable in the 80-minute lab session, with predict-then-observe rhythm and embedded bug hunts
- **Homework notebook** — extended analysis with the weekly memo prompts embedded
- **Memo rubric** — what the weekly technical note must demonstrate
- **Quiz** — short paper quiz at the start of class, testing the previous week's material (Weeks 2-6)
- **Readings** — curated paper PDFs referenced in lecture

---

## Assessment structure

| Component | Weight | Notes |
|---|---:|---|
| Weekly memos (best 5 of 6) | **45%** | Submitted as HTML via Moodle, due Wednesday morning before next class |
| Weekly quizzes (best 4 of 5, Weeks 2-6) | **15%** | In-class paper quiz, ~15 min at start of lecture |
| Final model decision dossier | **15%** | One-shot deployment recommendation defended across all six weeks of measurements |
| Final paper exam (~May 20) | **20%** | Closed-book |
| Participation | **5%** | |

Memos are graded on **reasoning over conclusion** — there is no single "right" recommendation. The rubric rewards measurement honesty, intellectual coherence, and named trade-offs.

---

## Quick start

### For enrolled students
1. Complete the [pre-work modules](pre-work/) before April 8 (~2-3 hours)
2. Each Wednesday: arrive ready for a paper quiz, attend lecture, complete the lab in Block 2
3. Work through the homework notebook before the next Wednesday; the memo is embedded
4. Submit memos as HTML via Moodle
5. Build the dossier across the term; submit before the final exam

### For self-study learners (no enrollment, just curious)
1. Read the [live site](https://earino.github.io/applied-deep-learning/) end-to-end as a structured 30-hour course
2. Run notebooks locally, on [Kaggle](https://www.kaggle.com/) (free T4 GPU), or in [Colab](https://colab.research.google.com/)
3. Read the [instructor blog](https://earino.github.io/applied-deep-learning/site/blog/) for the design decisions behind the course
4. You don't need to do the memos — but if you want to learn the discipline, write one for any week you cared about
5. Open issues with corrections, questions, or "I tried this and got a different number"

### For instructors adapting the materials
- Everything is CC BY 4.0; reuse with attribution
- The instructor solutions are **not** in this public repo (only student-facing materials are released)
- For questions about teaching adaptations or the source notebooks: [rubiae@ceu.edu](mailto:rubiae@ceu.edu)

---

## Trained checkpoints on Hugging Face Hub

Students can skip retraining and load the canonical checkpoints directly. All are open-weight.

| Repo | What | Used in |
|---|---|---|
| [`earino/ecbs5200-week3-encoder-lora`](https://huggingface.co/earino/ecbs5200-week3-encoder-lora) | ModernBERT-base + merged LoRA classifier, full data | Week 3, 4, 5 |
| [`earino/ecbs5200-week3-decoder-lora`](https://huggingface.co/earino/ecbs5200-week3-decoder-lora) | Qwen 2.5 0.5B + merged LoRA classifier head | Week 3, 4, 5 |
| [`earino/ecbs5200-modernbert-large-control-train-test`](https://huggingface.co/earino/ecbs5200-modernbert-large-control-train-test) | ModernBERT-large + LoRA, kitchen sink, train+test | The data-confound control |
| [`earino/ecbs5200-qwen3-32b-phase1-v4-teacher-canonical`](https://huggingface.co/earino/ecbs5200-qwen3-32b-phase1-v4-teacher-canonical) | Qwen3-32B + LoRA, temperature-scaled, on train+test | Week 6 distillation teacher |
| [`earino/ecbs5200-week6-vanilla-baseline`](https://huggingface.co/earino/ecbs5200-week6-vanilla-baseline) | ModernBERT-base, plain CE, train+test (Week 6 vanilla student) | Week 6 |
| [`earino/ecbs5200-week6-distilled-student`](https://huggingface.co/earino/ecbs5200-week6-distilled-student) | ModernBERT-base, KD from Qwen3-32B (Week 6 distilled student) | Week 6 |
| [`earino/ecbs5200-week6-teacher-logits`](https://huggingface.co/datasets/earino/ecbs5200-week6-teacher-logits) | Precomputed Qwen3-32B logits on train+test (79,278 × 113, fp16) | Week 6 |

---

## Repository structure

```
├── syllabus.md                # Full course syllabus
├── utils/
│   └── data_utils.py          # Shared data loading (used by all notebooks)
├── data/
│   ├── label_list.json        # 113 canonical class labels
│   ├── label_merge_mapping.json
│   ├── train_indices.json     # Canonical train split (57,846)
│   └── val_indices.json       # Canonical val split (6,430)
│
├── pre-work/                  # 8 modules + readiness quiz (~2-3 hours)
│   ├── 01_tokenization/
│   ├── 02_pretrained_encoders/
│   └── ... through 09_how_llms_are_built/
│
├── week1/                     # Fine-tuning and data audit
│   ├── slides.html / slides.pdf
│   ├── week1_lab.ipynb
│   ├── week1_homework.ipynb
│   └── week1_memo_rubric.md
├── week2/                     # Controlled improvement + readings/ (9 papers)
├── week3/                     # LoRA / PEFT + readings/
├── week4/                     # Error diagnosis, slices, calibration + readings/
├── week5/                     # Quantization + readings/ (10 papers)
├── week6/                     # Distillation + readings/ (13 papers)
│
├── site/                      # Live course-site sources
│   ├── blog/                  # 12 instructor blog posts on the course design
│   ├── week1.md ... week6.md  # Per-week landing pages
│   └── pre-work.md
│
└── experiments/               # Reproducible training notebooks for selected runs
```

---

## Dataset

[Consumer Complaints (Medium)](https://huggingface.co/datasets/determined-ai/consumer_complaints_medium) — real consumer complaints filed with the US Consumer Financial Protection Bureau.

- **113 canonical classes** after a documented merge step (raw dataset has 153 labels; classes with fewer than 5 examples are dropped — this happens transparently in `utils/data_utils.py` and is itself discussed in Week 4)
- **Canonical split**: 57,846 train / 6,430 val / 21,432 test, stratified, seed=42 — frozen for the term so all comparisons are apples-to-apples
- **Loaded automatically** via `utils.data_utils.load_course_data()` — no manual download needed

The 2,666-to-1 head/tail class-frequency ratio is the central engineering challenge of the course. Most weeks return to "what does the tail look like under *this* recipe?"

---

## Prerequisites

- Working knowledge of Python and basic ML concepts
- Comfort with Jupyter notebooks
- Prior coursework in machine learning (required for enrollment; helpful for self-study)
- Complete the [pre-work modules](pre-work/) before the first class (or, for self-study, before Week 1's notebook)

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International). You are free to share and adapt this material with attribution.

---

## Author

**[Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/)** ([rubiae@ceu.edu](mailto:rubiae@ceu.edu))

Professor of Practice, Central European University. Former Senior Director of Data Science at Meta; Chief Data Scientist at Domino Data Lab.

---

## Contributing

Found an error, a stale claim, or a number that doesn't reproduce? **Open an issue** — the course is built around the discipline of "measure on your own setup" and a counterexample is welcome data. Pull requests for typos, broken links, and clarifying language are also welcome.

For the curated reading list, course-design discussions, and revisions to claims as the field moves, follow the [instructor blog](https://earino.github.io/applied-deep-learning/site/blog/).
