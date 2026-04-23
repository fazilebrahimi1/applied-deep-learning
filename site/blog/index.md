---
title: Blog
layout: default
nav_order: 10
has_children: true
---

# Blog

Notes from building this course — what worked, what broke, what surprised us.

---

**[Your int8 Quantization Is 2.5× Slower Than fp16](11_int8_is_slower.html)**
The LLM.int8 paper from 2022 told you this would happen. The blog tutorials skip that part.

**[No magic, but first: a transformers PR](10_small_decoder_ecosystem_gap.html)**
The core teaching arc of my applied deep learning course at CEU is simple: **there's no magic**. Every part of a modern NLP pipeline — the data mix, the tokenizer, the model weights, the training re

**[When Better Means Worse](09_when_better_means_worse.html)**
I applied class weighting to a 113-category text classifier. Accuracy dropped five points. Macro F1 improved. Ten classes that the model had never once predicted correctly started getting nonzero F1.

**[You Can't Scale Your Way Out of a Data Problem](08_cant_scale_out_of_data_problem.html)**
I fine-tuned a Qwen3-8B on a 113-category consumer complaint classification task with a severe long tail using the same basic LoRA classification setup I used on the smaller models (rank 16, alpha 32,

**[Half a Percent: Thoughts and Results on Decoders for Text Classification](07_half_a_percent.html)**
I put a classification head on [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and trained 0.46% of its parameters with LoRA. On a 113-class consumer complaints task with a brutal long tail, 

**[44% Accuracy Without a Single Training Example](06_the_decoder_is_catching_up.html)**
I gave Opus 4.6 a list of 113 CFPB complaint categories and asked it to classify consumer complaints. No labeled examples in the prompt. No fine-tuning.

**[Your T4 Training Is 4x Slower Than It Should Be](05_bf16_on_t4.html)**
`torch.cuda.is_bf16_supported()` returns `True` on a Tesla T4. It's telling you the truth. The problem is you didn't ask the right question.

**[When Your "Ready to Use" Dataset Has the Same Category Listed Twice](04_the_label_problem.html)**
`determined-ai/consumer_complaints_medium` is a good teaching dataset. Pre-split, business-realistic, multiclass, compute-friendly. 64,000 training examples. 153 issue categories.

**[What Do You Teach When Compression Doesn't Produce a Speedup?](03_when_compression_doesnt_help.html)**
The plan for Week 5 was clean: students apply quantization to their model, benchmark the results, see the speedup, analyze whether compression hurts their weakest classes. Standard compression week. E

**[The Questions Model Cards Don't Answer](02_model_cards_wont_tell_you.html)**
Model cards tell you GLUE scores, parameter counts, and license terms. They don't tell you what happens when a non-expert runs the full workflow on constrained hardware.

**[Picking a Model for Teaching](01_picking_a_model_for_teaching.html)**
Picking a model for a research project: check the benchmarks, try it, move on. Picking a model for a teaching lab that 30 students will run for six weeks on free hardware: different exercise entirely.

