---
title: "The Questions Model Cards Don't Answer"
layout: default
parent: Blog
nav_order: 2
---

# The Questions Model Cards Don't Answer

Model cards tell you GLUE scores, parameter counts, and license terms. They don't tell you what happens when a non-expert runs the full workflow on constrained hardware.

For anything where reliability matters more than peak performance — teaching, internal tools, products with diverse users — you need a different set of questions.

## Four questions that matter

**1. What happens when someone does the obvious wrong thing?**

Does the system crash (good) or silently produce wrong results (catastrophic)?

DeBERTa-v3-base has a documented history of ONNX export correctness bugs — constants inserted where dynamic computations should be, dtype mismatches that silently corrupt outputs. No error, no warning. A student spends three hours debugging training code that was never the problem.

A crash is a twenty-minute fix. Silent corruption is a three-hour mystery.

**2. What's the support burden of the default configuration?**

How many users will hit a gotcha that's documented but non-obvious?

DeBERTa's LoRA reload requires explicitly listing the classification head in `modules_to_save`. Skip it and you get a fresh random head on top of your trained adapter. The result: metrics that swing wildly between runs. It's documented. It's avoidable. It's also exactly the kind of thing every third student gets wrong.

The question isn't whether a workaround exists. It's how many support tickets it generates.

**3. Does the full workflow work end-to-end on the target hardware?**

Not "does training work" — does training → checkpointing → adaptation → export → quantization → reload work, in sequence, on the actual hardware?

ModernBERT-base passed eight verification notebooks on a Kaggle T4. Every one broke at least once before passing. The full pipeline is a different test than any individual step.

**4. What's the compound cost of a failure?**

A bug in an isolated script costs an afternoon. A bug in Week 3 of a six-week cumulative course breaks Weeks 4, 5, and 6.

Three LoRA reload cycles. Fresh base model each time. Zero F1 delta. Exact prediction match. That result isn't exciting. It means Week 4 can trust Week 3's artifact.

## Boring reliability

DeBERTa and ModernBERT score within a point of each other on GLUE. Under these four questions, they're not close. One has silent export bugs, a LoRA reload pitfall, tokenizer friction, and less effective quantization. The other is boringly reliable.

For a course where students checkpoint every week and reload every Monday, "boringly reliable" is the most important property a model can have. The evaluation that matters happens in the verification notebooks you run before you commit.
