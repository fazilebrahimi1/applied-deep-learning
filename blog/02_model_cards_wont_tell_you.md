# Model Cards Won't Tell You If Your Course Will Work

Picking a model for a research project: check the benchmarks, try it, move on. Picking a model for a teaching lab that 30 students will run for six weeks on free hardware: different exercise entirely.

There's no model card field for "will this survive a Kaggle session timeout" or "does the ONNX export silently produce wrong outputs" or "will students get non-deterministic results when they reload their LoRA adapters because they didn't know about `modules_to_save`."

This is the story of evaluating two encoder models — DeBERTa-v3-base and ModernBERT-base — for a deep learning course built on Kaggle T4 GPUs. The benchmarks were close. The operational realities were not.

## What the model cards say

DeBERTa-v3-base and ModernBERT-base are both strong base-sized encoders. On GLUE, they're within a point of each other (88.1 vs 88.4 macro). They trade wins across subtasks — DeBERTa takes CoLA and MNLI, ModernBERT takes MRPC and RTE. If benchmarks were the whole story, it's a coin flip.

The model cards also tell you the basic specs. DeBERTa: ~184M parameters, 128K vocabulary, disentangled attention, SentencePiece tokenizer, 512 max length. ModernBERT: 149M parameters, 50K vocabulary, RoPE + alternating attention, standard tokenizer, 8K max length.

None of this tells you what happens when a student tries to export the model to ONNX and the graph silently produces incorrect predictions.

## What the model cards don't say

A course that runs on Kaggle T4s needs a model that can do all of these things reliably:

- Fine-tune with a manual training loop (students need to see the guts)
- Save and reload checkpoints (Kaggle sessions die constantly)
- Apply LoRA adapters, save them, reload them into a fresh model, get the same predictions
- Export to ONNX for size quantization
- Resume training after an interruption without corrupting optimizer state

These aren't exotic requirements. They're the basic operations of the course across six weeks. And each one is a place where a model can fail in ways that no benchmark captures.

**DeBERTa's ONNX export has a documented history of correctness bugs.** Transformers issues and PRs show that DeBERTa's graph tracing has produced incorrect outputs — constants inserted where dynamic computations should be, dtype mismatches that silently corrupt results. An ONNX file exists in the model repo, but students won't be downloading a pre-made file. They'll be running the export themselves, hitting whatever version-specific behavior their Kaggle environment has that day.

A model that exports but silently produces wrong results is the worst possible failure in a classroom. A crash, you can debug. Wrong predictions with no error message? That's a student spending three hours convinced their training code is broken.

**DeBERTa's LoRA reload has a well-documented pitfall.** When you add LoRA to a sequence classification model, the classification head is randomly initialized. If you don't explicitly include it in `modules_to_save`, it won't be saved with the adapter. On reload, you get a fresh random head on top of your trained adapter. The result: metrics that swing wildly between runs, or predictions that look inverted.

This is documented. It's avoidable with the right configuration. It's also exactly the kind of thing that every third student will get wrong, generating support tickets that eat instructor time and student confidence.

**DeBERTa's 128K vocabulary changes the quantization math.** About 98M of DeBERTa's 184M parameters are embeddings — over half the model. INT8 dynamic quantization targets linear layers, not embeddings. Published results show DeBERTa getting roughly 2x size reduction from INT8 quantization (705 MB to 350 MB), not the 4x that ModernBERT gets (571 MB to 144 MB). The quantization exercise is less dramatic with half your parameters untouched.

**DeBERTa's tokenizer adds friction.** No fast tokenizer implementation. SentencePiece dependency. Users still report initialization issues tied to slow-to-fast conversion. In isolation, each of these is minor. In a Kaggle notebook where students are copying code from the lab handout, tokenizer setup that "usually works" is a support burden.

None of this shows up in GLUE scores.

## What ModernBERT gets right for teaching

ModernBERT isn't a perfect model. Its ONNX graph crashes ONNX Runtime's CUDA provider due to a Cast node bug. torch.compile gives a 5% speedup — barely measurable. It's a newer model with a smaller community and fewer Stack Overflow answers.

But across eight verification notebooks on Kaggle T4 GPUs, it passed every operational test:

- Fine-tuning completes with ~5 GB of 15 GB VRAM used
- Checkpoints save and reload with zero metric delta
- LoRA adapters save and reload with bit-identical predictions across three independent cycles — no `modules_to_save` configuration needed
- ONNX INT8 export produces a clean 4x size reduction
- Training resumes after simulated session death with optimizer state preserved
- Standard tokenizer, no SentencePiece, no fast-tokenizer gaps

The LoRA result is worth emphasizing. Three separate reload cycles. Fresh base model each time. Zero F1 delta. Exact prediction match. For a course where students checkpoint every week and reload every Monday, "boringly reliable" is the most important property a model can have.

## The pedagogical frame

Model selection for a course is model selection under constraints that benchmarks don't measure. The constraints:

**Students are beginners with the tooling.** They will get versions wrong, copy code imperfectly, restart kernels at bad times, and run cells out of order. The model that degrades gracefully under student-grade operation is more valuable than the model that's 0.3 points better on MNLI.

**The model anchors six weeks of progressive work.** A bug in the LoRA reload path doesn't just break Week 3 — it breaks every subsequent week that builds on Week 3's artifact. The cost of a model-specific gotcha compounds across the semester.

**The compute is free and constrained.** Kaggle T4s with 16 GB VRAM, 30-hour weekly quotas, sessions that time out. The model needs to train fast enough for students to iterate during a 70-minute lab, checkpoint reliably enough to survive timeouts, and fit in memory with enough headroom that batch size choices aren't life-or-death.

**The learning outcome isn't "use this specific model."** It's "understand fine-tuning, adaptation, error analysis, compression, and distillation well enough to do it on any model." The model is a vehicle for the skills, and a vehicle that breaks down every other week is a bad vehicle regardless of its top speed.

Under these constraints, a model that's marginally better on benchmarks but has documented ONNX correctness bugs, a LoRA reload pitfall, tokenizer friction, and less effective quantization is the wrong choice — even if the benchmarks are close.

## The general principle

For any context where reliability matters more than peak performance — teaching, internal tools, products with diverse users — the evaluation has to go beyond model cards. The questions that matter are:

- What happens when someone does the obvious wrong thing? (Does it fail loudly or silently?)
- What's the support burden? (What will users/students get wrong, and how much time does that cost?)
- Does the full workflow work, end to end, on the actual target hardware?
- What's the compound cost of a failure? (A bug in week 1 vs a bug in an isolated script)

The course runs April and May 2026. More posts coming as the materials come together — including how a "ready to use" dataset turned out to have label schema drift baked into it, and the surprisingly interesting question of what to teach when compression doesn't produce a speedup.
