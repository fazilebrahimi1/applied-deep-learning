# Picking a Model for Teaching

Spent the last two days trying to break a model.

Not because there's anything wrong with it — ModernBERT-base is genuinely good. Fast, modern architecture, great benchmarks. I'm building a deep learning course at CEU where students fine-tune one model across six weeks, and I needed to know if this model would survive the whole semester on free Kaggle GPUs.

Turns out "good model" and "model I can hand to 30 students on free hardware" are very different questions.

The first candidate was DeBERTa-v3-base. Better known, strong benchmarks. Initially killed it because a custom operation broke the quantization path I was planning. Later, the quantization approach changed entirely — which reopened the question of whether DeBERTa should have stayed in the running. More on that in a future post. For now: the decision to go with ModernBERT held up after re-examination, but the *reasons* turned out to be different than the original ones.

ModernBERT loaded fine. Trained fine. Checkpointed and reloaded fine. LoRA adapters saved and reloaded with bit-identical predictions across three independent cycles. All great.

Then I tried to quantize it for the compression week.

Int8 quantization cut the model from 571 MB to 144 MB. Beautiful. But when I loaded the quantized model on GPU, inference ran 12x slower. No error. No warning. ONNX Runtime silently fell back to CPU because its GPU provider doesn't support the quantized operations. You just have to... know that, apparently.

I tried four different approaches to get GPU-accelerated compression working. The optimum library crashed with an illegal memory access that poisoned the entire CUDA session. torch.compile gave a 5% speedup — basically noise. Raw ONNX export worked but was slightly slower than plain PyTorch.

A 149M parameter model on a GPU with 10x memory headroom doesn't need compression for speed. There's nothing to compress against.

Here's where it gets interesting for the course: the original plan was "students apply quantization, see speedup." The honest version is better. Students do the full exercise, measure carefully, discover it doesn't help for speed, hit the silent CPU fallback, and then reason about whether the 4x size reduction matters for their deployment scenario.

"I benchmarked it and the speedup wasn't worth it" is a more useful thing to be able to say in an interview than "my professor showed me a demo where it worked."

The dataset had its own surprise. 153 issue categories — except "Incorrect information on credit report" and "Incorrect information on your report" are the same category from different time periods. They're the model's top confusion pair. The model isn't confused. The labels are.

Eight verification notebooks. Every one broke at least once before passing. The total gap between "this model has good benchmarks" and "I can build a semester on this" took two days to close.

If you're building anything that other people have to run reliably — a course, an internal tool, a product — model benchmarks are the beginning of the conversation, not the end.
