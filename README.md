# Applied Deep Learning

**A hands-on course in post-training deep learning engineering — fine-tune, adapt, analyze, compress, and justify a real model under real constraints.**

Course materials for ECBS5200 at Central European University's MS in Business Analytics program.

---

## What's This Course About?

Most ML courses teach you to use models as service endpoints — prompt them, chain them, build apps around them. This course teaches you to treat a model as a **trainable, inspectable, compressible artifact**. Something you own, something you can open up, something you can make smaller and faster and cheaper.

Over six weeks, you work on one cumulative problem: classifying consumer financial complaints into 113 categories. You fine-tune a pretrained encoder, improve it, adapt it with LoRA, analyze where it fails, compress it via quantization, and distill knowledge from a stronger (but 1000x more expensive) decoder reference system. At the end, you write a recommendation: is the cheap model good enough, or is the expensive one worth the cost?

---

## Key Information

| | |
|---|---|
| **Instructor** | [Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/), Professor of Practice |
| **Format** | 6 Wednesdays, two 100-minute blocks per day |
| **Dates** | Apr 8, 15, 22, 29, May 6, 13 (2026) |
| **Location** | CEU Vienna Campus |
| **Compute** | Free-tier GPU notebooks (Kaggle T4) |
| **Base Model** | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M params, Apache 2.0) |

---

## Quick Start

```bash
git clone https://github.com/earino/applied-deep-learning.git
cd applied-deep-learning
```

Open any `.ipynb` notebook in Kaggle, Colab, or your local Jupyter environment. No build steps required.

---

## Repository Structure

```
├── syllabus.md               # Full course syllabus
├── utils/
│   └── data_utils.py         # Shared data loading (used by all notebooks)
├── data/
│   ├── label_list.json       # 113 canonical class labels
│   ├── label_merge_mapping.json
│   ├── train_indices.json    # Canonical train/val split
│   └── val_indices.json
│
├── pre-work/                  # 8 modules + readiness quiz (2-3 hours)
│   ├── 01_tokenization/
│   ├── 02_pretrained_encoders/
│   ├── ...
│   └── readiness_quiz/
│
├── week1/                     # Fine-tuning and data audit
│   ├── slides.html
│   ├── slides.pdf
│   ├── week1_lab.ipynb
│   ├── week1_homework.ipynb
│   └── week1_memo_rubric.md
│
├── week2/                     # Controlled improvement and error analysis
│   ├── slides.html
│   ├── slides.pdf
│   ├── week2_lab.ipynb
│   ├── week2_homework.ipynb
│   ├── week2_memo_rubric.md
│   └── readings/             # 9 paper PDFs (Zhang 2017 – Shi 2024)
│
├── site/blog/                 # Instructor blog posts on course design
│
├── experiments/               # Reproducible training notebooks
│
└── ...                        # Weeks 3-6 released weekly
```

---

## Course Structure

| Week | Topic | Date |
|:---:|:---|:---:|
| — | [Pre-work](pre-work/) (2-3 hours) | Before Apr 8 |
| 1 | [Fine-tuning and data audit](week1/) | Apr 8 |
| 2 | [Controlled improvement and error analysis](week2/) | Apr 15 |
| 3 | Parameter-efficient adaptation (LoRA/PEFT) | Apr 22 |
| 4 | Error analysis, slices, calibration, decoder comparison | Apr 29 |
| 5 | Quantization and decoder economics | May 6 |
| 6 | Distillation and final model decision | May 13 |

Each week contains:
- **Slides** (HTML + PDF) — presented in Block 1
- **Lab notebook** — completable during the Block 2 lab session
- **Homework notebook** — analysis, experiments, and embedded memo prompts
- **Memo rubric** — what the weekly technical note should cover
- **Readings** — curated paper PDFs referenced in the lecture (starting Week 2)

Materials are released weekly before each class.

---

## Dataset

[Consumer Complaints (Medium)](https://huggingface.co/datasets/determined-ai/consumer_complaints_medium) — real consumer complaints filed with the US Consumer Financial Protection Bureau. 113 issue categories after label cleanup. Loaded automatically via `utils/data_utils.py`.

---

## For Students

1. Complete the [pre-work modules](pre-work/) before April 8
2. Each week, open the lab notebook and work through it during Block 2
3. After class, work through the homework notebook — it includes your memo prompts
4. Submit your memo as HTML via Moodle by Wednesday morning before the next class

---

## Prerequisites

- Working knowledge of Python and basic ML concepts
- Git basics and comfort with Jupyter notebooks
- Prior coursework in machine learning (required)
- Complete the [pre-work](pre-work/) before the first class

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International). You are free to share and adapt this material with attribution.

---

## Author

**[Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/)** (rubiae@ceu.edu)

Professor of Practice, Central European University. Former Senior Director of Data Science at Meta; Chief Data Scientist at Domino Data Lab.

---

## Contributing

Found an error? Have a suggestion? Issues and pull requests are welcome.
