---
title: Home
layout: default
nav_order: 1
---

# Applied Deep Learning

**ECBS5200** · Central European University · Spring 2026

A hands-on graduate course in post-training deep learning engineering. Fine-tune, adapt, analyze, compress, and justify a real model under real constraints.

Over six weeks, you work on one cumulative problem: classifying consumer financial complaints into 113 categories. You fine-tune a pretrained encoder, improve it, adapt it with LoRA, analyze where it fails, compress it via quantization, and distill knowledge from a stronger (but 1000x more expensive) decoder reference system. At the end, you write a recommendation: is the cheap model good enough, or is the expensive one worth the cost?

**Instructor:** [Eduardo Ariño de la Rubia](https://www.linkedin.com/in/earino/) · **6 Wednesdays** · Apr 8 – May 13, 2026
**Compute:** Free-tier GPU notebooks (Kaggle T4) · **Base model:** [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M params)

## Course Schedule

| Week | Topic | Date | Materials |
|:---:|:---|:---:|:---:|
| — | [Pre-work](site/pre-work.html) | Before Apr 8 | 8 modules |
| 1 | [Fine-tuning and data audit](site/week1.html) | Apr 8 | Released |
| 2 | [Controlled improvement and failure analysis](site/week2.html) | Apr 15 | Released |
| 3 | [Parameter-efficient adaptation (LoRA/PEFT)](site/week3.html) | Apr 22 | Released |
| 4 | [Error diagnosis: slices, calibration, and cross-model analysis](site/week4.html) | Apr 29 | Released |
| 5 | [Quantization and decoder economics](site/week5.html) | May 6 | Released |
| 6 | Distillation and final model decision | May 13 | — |

## Latest from the Blog

**[Your int8 Quantization Is 2.5× Slower Than fp16](site/blog/11_int8_is_slower.html)** — The LLM.int8 paper from 2022 told you this would happen. The blog tutorials skip that part.

[All posts →](site/blog/)

