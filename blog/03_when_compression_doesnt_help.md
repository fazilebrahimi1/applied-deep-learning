# What Do You Teach When Compression Doesn't Produce a Speedup?

The plan for Week 5 was clean: students apply quantization to their model, benchmark the results, see the speedup, analyze whether compression hurts their weakest classes. Standard compression week. Every applied ML course has one.

The model is ModernBERT-base. 149M parameters. Running on a Kaggle T4 with 15 GB of VRAM. Peak training memory: about 5 GB.

That's 10 GB of headroom. The model isn't memory-bound. It isn't compute-bound. There's nothing to compress against.

## Four approaches, zero speedup

INT8 quantization through ONNX Runtime cut the model from 571 MB to 144 MB on disk. 4x smaller. Genuinely useful for deployment packaging. But loading the quantized model on GPU produced no error — inference just ran 12.7x slower. ONNX Runtime silently fell back to CPU because its CUDA provider doesn't have kernels for the quantized operations (`MatMulInteger`, `DynamicQuantizeLinear`). No warning. No exception. Just slow.

The `optimum` library's ONNX optimizer with operator fusion and fp16 conversion crashed on a Cast node in ModernBERT's attention graph with `cudaErrorIllegalAddress`. This poisoned the CUDA context — every GPU operation in the session failed after that, even on unrelated models.

Raw `torch.onnx.export` from CPU, loaded with `onnxruntime.InferenceSession` on GPU — the simplest possible ONNX path — worked cleanly. CUDA provider verified. Quality preserved perfectly. Speedup: 0.95x. Slightly slower than plain PyTorch.

`torch.compile`: one line of code, worked fine. 1.05x. Noise.

A 149M parameter encoder on a T4 with 10x memory headroom does not benefit from compression for speed. The arithmetic is clear in retrospect. It wasn't obvious going in.

## The temptation to fake it

At this point there's a decision to make. The course has a compression week on the schedule. The compression didn't produce a speedup. The options:

**Option A:** Pick a different model where quantization works dramatically. Maybe something in the 1B+ range where INT8 is the difference between fitting in VRAM and not. The demo works. Students see impressive numbers. But now they're using a model they didn't train, for an exercise disconnected from their semester-long project.

**Option B:** Skip the compression week. Replace it with something else. But compression is in the learning outcomes. It's a real skill. Cutting it because the numbers are boring teaches students that you only try things when you already know the answer.

**Option C:** Keep the exercise exactly as planned. Students apply compression. They benchmark carefully. They discover it doesn't help for speed. They find the silent CPU fallback. They have to reason through the result and write about it.

## Why Option C is the right answer

Option C sounds like a failure. Students do the work and nothing happens. But look at what they actually learn:

**How to benchmark honestly.** Most students have never measured inference latency, throughput, or memory footprint. Just learning to benchmark — warmup batches, CUDA synchronization, median not mean, separating data loading from model computation — is a skill they'll use on every model they ever deploy.

**That compression isn't free or universal.** The default assumption for anyone who's read about quantization but never applied it is "make model smaller, model goes faster." The reality is that it depends on model size, hardware, the quantization framework's kernel support, and whether you're actually bottlenecked. Discovering this through measurement is different from being told it in a lecture.

**The silent fallback trap.** ONNX Runtime loading an INT8 model with `CUDAExecutionProvider`, silently falling back to CPU, and running 12.7x slower — with no error and no warning — is exactly the kind of production failure that burns real engineering time. Finding this in a lab exercise, having to diagnose why inference suddenly got slow, and learning to verify execution providers is directly applicable operational knowledge.

**How to reason about deployment trade-offs.** The 4x size reduction is real. Whether it matters depends on the deployment scenario. Shipping to a CPU inference server? The size reduction matters and INT8 is faster on CPU than fp32. Running on a GPU with headroom? Size doesn't matter and there's no speed gain. The operating point memo students write — "given these constraints, here's what I'd ship" — is more interesting to write when the answer isn't obvious.

**The difference between "it didn't work" and "it wasn't needed."** These are different conclusions requiring different reasoning. The first is a debugging problem. The second is an engineering judgment. Students who can articulate why compression wasn't needed for their specific model on their specific hardware are demonstrating the kind of thinking that matters in applied ML roles.

## Context makes it stick

One slide in the lecture shows what happens with a 7B parameter model. 28 GB in fp32. Doesn't fit on a T4 at all. INT8 quantization brings it to 7 GB — fits easily. Same technique, same workflow, dramatically different outcome.

The students ran the workflow on a model where it doesn't matter. They know the workflow cold. When they encounter a model where it does matter, they'll know what to do and they'll know how to verify that it actually worked. And they won't assume it helped just because they applied it.

`torch.compile` gets five minutes. Not as a Week 5 topic — as professional hygiene, alongside `model.eval()` and `torch.no_grad()`. One line of code, always do it for inference, sometimes you get 5%, sometimes 30%, depends on the model and hardware. Move on.

## The general point

There's a strong temptation in course design to make every exercise produce a satisfying result. Train a model: accuracy goes up. Apply LoRA: parameter count goes down. Quantize: inference gets faster. Nice clean narrative.

But applied ML isn't a narrative. It's measurement, reasoning, and trade-offs — and sometimes the measurement says "this didn't help." A course that only shows students wins is a course that hasn't prepared them for the job, where most of the work is figuring out what to try, trying it, and discovering it didn't move the needle.

The compression week works better when the compression doesn't produce a speedup. That's not a consolation. That's the design.

The course runs April and May 2026 at CEU Vienna. Previous post: [Model Cards Won't Tell You If Your Course Will Work](#).
