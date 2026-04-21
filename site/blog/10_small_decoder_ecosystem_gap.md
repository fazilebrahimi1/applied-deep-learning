---
title: "No magic, but first: a transformers PR"
layout: default
parent: Blog
nav_order: 1
---

# No magic, but first: a transformers PR

The core teaching arc of my applied deep learning course at CEU is simple: **there's no magic**. Every part of a modern NLP pipeline — the data mix, the tokenizer, the model weights, the training recipe — is a thing you could inspect and rebuild yourself, given enough time. The easiest way to demonstrate that claim is to teach with models where every part actually is public.

For the Week 3 decoder baseline, we picked Qwen 2.5-0.5B. It worked fine: 57.5% accuracy, 0.236 macro F1 — enough to beat the ModernBERT encoder baseline (56.5% / 0.209) on the same 113-class consumer-complaint task, and give students a real quality/latency trade-off to argue about.

But Qwen is only open-*weights*. The training data isn't public, the training logs aren't public, the exact recipe isn't documented. If a student asks "could we retrain this from scratch?" the honest answer is no. That's precisely the opposite of "no magic."

So I went looking for a small decoder that was *fully* open — weights, training data, logs, recipe. Apache 2.0. Around 1B or smaller.

OLMo 2 was the obvious choice. Not the only fully-open small decoder in the wild; Pythia and TinyLlama both exist. But it was the most recent one I found with a genuinely strong end-to-end openness story: weights public, **Dolma** (the 2.3T-token training corpus) public with per-source token counts and filtering code, training logs public. OLMo is about as close as the field gets to end-to-end scientific openness — data artifacts, tooling, checkpoints, and training process all exposed in a way that makes retraining and forensic inspection materially more realistic than for a weights-only release.

You cannot load OLMo 2 for sequence classification.

```python
AutoModelForSequenceClassification.from_pretrained("allenai/OLMo-2-0425-1B")
# ValueError: ...
```

Granite 4.0 350M — same story. SmolLM2 — same story. A few others too. The models can classify. The adapter between HuggingFace's standard classification interface and these models simply hadn't been written.

That kind of missing-head error sounds like a model limitation. In this case, it wasn't.

## Why it was missing

I opened `modular_olmo.py`. It's tiny. Every class is a thin subclass of its Llama equivalent:

```python
class OlmoAttention(LlamaAttention): ...           # tweaked qkv clipping
class OlmoDecoderLayer(LlamaDecoderLayer): ...     # swap in OlmoLayerNorm
class OlmoForCausalLM(LlamaForCausalLM): pass      # literally identical
```

The `modular_*.py` file isn't runtime inheritance — it's a **code-generation template**. The `transformers` tooling reads it and produces a standalone `modeling_*.py` where the class body is spelled out in full. At import time, `OlmoForCausalLM.__mro__` doesn't include any Llama class; the generated file stands alone. The modular file just expresses: *"give me the Llama thing, with these substitutions, and generate me a clean standalone module."*

Which meant adding classification to the OLMo family was a few lines:

```python
class OlmoForSequenceClassification(LlamaForSequenceClassification):
    pass
```

Plus auto-mapping registration, tests, and docs.

## The PR

[huggingface/transformers #45551](https://github.com/huggingface/transformers/pull/45551) — *Add ForSequenceClassification heads for the OLMo family*. **+64 / -8 across 13 files.** Adds classification heads for all three OLMo generations: **OLMo, OLMo2, and OLMo3.**

End-to-end verification on a T4 with the course's 113-class task. 30 CI checks green — torch tests, tokenization, training CI, tensor-parallel, exotic-models, slow-tests for `auto, olmo, olmo2, olmo3`. HuggingFace maintainer **Matt** ([@Rocketknight1](https://github.com/Rocketknight1)) picked it up, ran the full slow-test suite, requested zero changes, and merged within ~24 hours of approval. The PR ships because Matt did the actual work of shipping it — the kind of fast, careful maintainer review that keeps open-source libraries healthy.

After merge, `AutoModelForSequenceClassification.from_pretrained("allenai/OLMo-2-0425-1B", num_labels=113)` just works.

## The deeper point

The interesting thing isn't that I wrote a small PR. It's what the PR being small *means*.

Modern decoder LLMs are architectural near-clones of each other. OLMo's divergence from Llama is three small tweaks — LayerNorm instead of RMSNorm, optional qkv clipping, a rotary-embedding dtype choice. The forward-pass math stays close to a proven recipe, because at this point the wheel already works. **OLMo's scientific contribution isn't architecture. It's the unusually reproducible training story** — Dolma, open training code, open hyperparameters, open checkpoints along the way. The architecture is a near-commodity.

Across many modern decoder families — Qwen, Gemma, Mistral, Phi — the architectural distance from Llama is smaller than their separate lineages suggest. Larger and more optimized variants diverge in important ways: attention kernels, long-context strategies, efficiency tricks. But at the small-model end, the deltas are often narrow. In `transformers`, that often means the missing task-specific plumbing is shorter than the model name suggests.

If you're blocked on a HuggingFace-supported model by a "this doesn't have a classification head" error, check the `modular_*.py` source before assuming it's architectural. For a surprisingly large class of these omissions, the fix is closer to **engineering glue than research** — often just tracing the modular pattern, registering the class, and wiring tests and docs.

The ecosystem gap for small decoders is real. It's also mostly waiting for PRs that haven't been written yet. That's one fewer of them as of today.

---

*The 113-class consumer-complaint course materials, including the encoder/decoder comparison and the small-decoder benchmarks, are at [earino.github.io/applied-deep-learning](https://earino.github.io/applied-deep-learning/). The PR discussed in this post is [huggingface/transformers #45551](https://github.com/huggingface/transformers/pull/45551), merged 2026-04-22.*
