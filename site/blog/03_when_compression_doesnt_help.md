---
title: "What Do You Teach When Compression Doesn't Produce a Speedup?"
layout: default
parent: Blog
nav_order: 3
---

# What Do You Teach When Compression Doesn't Produce a Speedup?

The plan for Week 5 was clean: students apply quantization to their model, benchmark the results, see the speedup, analyze whether compression hurts their weakest classes. Standard compression week. Every applied ML course has one.

The problem: our model has 10 GB of headroom on a T4. It isn't memory-bound. It isn't compute-bound. There's nothing to compress against.

## Four approaches, zero speedup

INT8 quantization through ONNX Runtime cut the model from 571 MB to 144 MB on disk — a genuine 4x size reduction. But on GPU, inference ran 12.7x slower. ONNX Runtime silently fell back to CPU because its CUDA provider lacks kernels for the quantized operations. No warning. No exception. Just slow.

The `optimum` library crashed with an illegal memory access that poisoned the entire CUDA session. Raw ONNX export worked but was slightly slower than plain PyTorch. `torch.compile` gave a 5% speedup — noise.

A 149M parameter encoder on a T4 with 10x memory headroom does not benefit from compression for speed. The arithmetic is clear in retrospect. It wasn't obvious going in.

## The temptation to fake it

At this point there's a decision to make. The course has a compression week on the schedule. The compression didn't produce a speedup. The options:

**Option A:** Pick a different model where quantization works dramatically — something in the 1B+ range where INT8 is the difference between fitting in VRAM and not. The demo works. Students see impressive numbers. But now they're using a model they didn't train, for an exercise disconnected from their semester-long project.

**Option B:** Skip the compression week. Replace it with something else. But compression is in the learning outcomes. Cutting it because the numbers are boring teaches students that you only try things when you already know the answer.

**Option C:** Keep the exercise exactly as planned. Students apply compression. They benchmark carefully. They discover it doesn't help for speed. They find the silent CPU fallback. They have to reason through the result and write about it.

## Why Option C is the right answer

Option C sounds like a failure. Students do the work and nothing happens. Three things make it worth it:

**How to benchmark honestly.** Most students have never measured inference latency, throughput, or memory footprint. Just learning to benchmark — warmup batches, CUDA synchronization, median not mean, separating data loading from model computation — is a skill they'll use on every model they ever deploy.

**That compression isn't free or universal.** The default assumption for anyone who's read about quantization but never applied it is "make model smaller, model goes faster." The reality depends on model size, hardware, kernel support, and whether you're actually bottlenecked. Discovering this through measurement is different from being told it in a lecture.

**The silent fallback trap.** ONNX Runtime loading an INT8 model, silently falling back to CPU, running 12.7x slower with no error and no warning — this is exactly the kind of production failure that burns real engineering time. Finding it in a lab and learning to verify execution providers is directly applicable operational knowledge.

## Context makes it stick

One slide in the lecture shows what happens with a 7B parameter model. 28 GB in fp32. Doesn't fit on a T4 at all. INT8 quantization brings it to 7 GB — fits easily. Same technique, same workflow, dramatically different outcome.

The students ran the workflow on a model where it doesn't matter. They know the workflow cold. When they encounter a model where it does matter, they'll know what to do and they'll know how to verify that it actually worked.

## The general point

There's a strong temptation in course design to make every exercise produce a satisfying result. Train a model: accuracy goes up. Apply LoRA: parameter count goes down. Quantize: inference gets faster. Nice clean narrative.

Applied ML is measurement, reasoning, and trade-offs — and sometimes the measurement says "this didn't help." A course that only shows students wins hasn't prepared them for the job, where most of the work is figuring out what to try, trying it, and discovering it didn't move the needle.

The compression week works better when the compression doesn't produce a speedup. That's not a consolation. That's the design.
