---
title: "Picking a Model for Teaching"
layout: default
parent: Blog
nav_order: 1
---

# Picking a Model for Teaching

Picking a model for a research project: check the benchmarks, try it, move on. Picking a model for a teaching lab that 30 students will run for six weeks on free hardware: different exercise entirely.

I'm building a deep learning course at CEU where students fine-tune one model across six weeks — training, LoRA adaptation, error analysis, quantization, distillation. The model has to survive all of it on a free Kaggle T4. Benchmarks don't tell you if that's going to work.

## What "works for teaching" actually means

The model needs to:

- Fine-tune in under 40 minutes on a free T4 GPU
- Save and reload checkpoints deterministically
- Support LoRA adapters that produce bit-identical predictions after save/reload
- Quantize to int8 for the compression week
- Be open-licensed so students can use it for anything

I started with DeBERTa-v3-base — strong benchmarks, well-known. A custom activation function broke the quantization pipeline. Killed it. ModernBERT-base (149M parameters, Apache 2.0, 2024) passed every check. Training, checkpointing, LoRA save/reload — all clean. Then I tried to quantize it.

## What broke

Int8 quantization cut the model from 571 MB to 144 MB. Beautiful. But inference on GPU ran 12x slower. No error. No warning. ONNX Runtime had silently fallen back to CPU because its GPU provider doesn't support the quantized operations.

I tried four alternatives. The `optimum` library crashed with an illegal memory access that poisoned the CUDA session. `torch.compile` gave a 5% speedup — noise. Raw ONNX export worked but was slightly slower than plain PyTorch.

The underlying problem: a 149M parameter model on a GPU with 10x memory headroom doesn't need compression for speed. There's nothing to compress against.

## Why this is a better lesson

The plan: students apply quantization, see speedup. The honest version is more valuable. Students do the full exercise, measure carefully, discover it doesn't help for speed, hit the silent CPU fallback, and then reason about whether the 4x size reduction matters for their deployment scenario.

"I benchmarked it and the speedup wasn't worth it" is a more useful thing to say in an interview than "my professor showed me a demo where it worked."

## The gap

Eight verification notebooks. Every one broke at least once before passing. Two days from "this model has good benchmarks" to "I can build a semester on this."

If you're building anything that other people have to run reliably — a course, an internal tool, a product — model benchmarks are the beginning of the conversation, not the end.
