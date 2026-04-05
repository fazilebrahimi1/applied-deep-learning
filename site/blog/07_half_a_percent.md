---
title: "Half a Percent: Thoughts and Results on Decoders for Hard Text Classification"
layout: default
parent: Blog
nav_order: 7
---

# Half a Percent: Thoughts and Results on Decoders for Hard Text Classification

[BERT’s founding argument](https://arxiv.org/abs/1810.04805) was that bidirectional attention produces better sentence representations than left-to-right attention. That argument launched a decade of “use an encoder for classification.” The argument was sound. The field moved on.

I put a classification head on a 0.5B decoder and trained 0.46% of its parameters with LoRA. On a 113-class consumer complaints task with a brutal long tail, it beat a fully fine-tuned 149M encoder on the same data, on the same free Kaggle T4. Accuracy was slightly higher (57.0% vs 56.6%). Macro F1 was materially higher (0.240 vs 0.209). It also rescued 9 additional rare classes from zero F1.

This is not an architecture-isolated comparison. The decoder is larger and much more pretrained. That is the point, not a loophole. In 2026, practitioners do not choose between perfectly matched models that differ only in attention direction. They choose between model families as they actually exist.

## The comparison nobody makes

Most evidence for “encoders win at classification” compares fine-tuned encoders against prompted decoders: zero-shot GPT, few-shot Claude, instruction-tuned label generation. Those comparisons are real, but they are asymmetric. A model trained for the task is being compared against one inferring the task from general knowledge.

The more interesting comparison is supervised classifier versus supervised classifier: an encoder with a classification head against a decoder adapted for classification on the same training data.

That comparison is still rare. When [Yousefiramandi and Cooney](https://arxiv.org/abs/2512.12677) ran it in late 2025, the classification-head decoder beat instruction-style classification and was competitive with, and in some cases better than, BERT-style baselines. On my 113-class CFPB setup, the decoder beat the encoder on every quality metric except speed.

## Where scale pays off

Not all classification tasks are the same. Binary sentiment on balanced movie reviews is a very different problem from 113 complaint categories where 37 classes have fewer than 50 training examples.

The decoder has 494 million parameters and was pretrained on 18 trillion tokens. The encoder has 149 million parameters and 2 trillion tokens of pretraining. For common classes with thousands of examples, both models learn the task well enough. For rare classes with 5 or 8 examples, the larger model’s pretraining appears to matter more. LoRA is not teaching the decoder language from scratch. It is aligning a large pretrained representation space to the label set.

That does not mean decoders now dominate classification everywhere. On balanced binary tasks, strong encoders like DeBERTa-v3 [still win many benchmarks](https://alex-jacobs.com/posts/beatingbert/) and run much faster. The interesting regime is harder: many labels, severe imbalance, and a long tail where rare-class performance matters.

## The real question is cost

The decoder is about 19x slower per example: 58 milliseconds versus 3. For a one-off batch of 64,000 complaints, that is 7 minutes versus 3. Both finish before your coffee gets cold. But latency compounds. If you process millions of documents a day, 19x slower means 19x more GPU-hours. That is not an inconvenience. That is a line item that determines whether your pipeline runs on one T4 or an array of A10s.

A 0.031 improvement in macro F1 might not survive a cost-benefit analysis when the compute bill goes up by an order of magnitude. For a startup or a cost-center pipeline, the encoder’s speed advantage is not a minor footnote. It is the reason the encoder exists.

The gap widens with decoder size. In my runs, a 1.5B decoder reached 58.3% accuracy and a 3B decoder reached 58.7%, while the encoder stayed at 56.6%. Better quality, but each step up multiplies the inference cost again.

The question is not which model is better. The question is whether the rare-class improvement pays for itself in your specific deployment scenario. Sometimes it does. Often it does not.

## What changed

Yi Tay [documented](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) something many practitioners noticed but few said out loud: encoder models did not disappear because they stopped working. They disappeared from the frontier because the field unified around more flexible decoder-centric ecosystems, and those models kept getting stronger.

The result is that the old encoder advantage in inductive bias is no longer enough to settle the practical question. In some settings, especially where labels are many and training signal is sparse, decoder scale can outweigh it.

The default tutorial story for classification is still encoder-first. For two-class sentiment on 50,000 balanced examples, that is still often the right instinct. For 113 imbalanced classes where the rare ones matter, it may not be.

