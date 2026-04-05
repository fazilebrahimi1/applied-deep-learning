---
title: "When Your \"Ready to Use\" Dataset Has the Same Category Listed Twice"
layout: default
parent: Blog
nav_order: 4
---

# When Your "Ready to Use" Dataset Has the Same Category Listed Twice

`determined-ai/consumer_complaints_medium` is a good teaching dataset. Pre-split, business-realistic, multiclass, compute-friendly. 64,000 training examples. 153 issue categories.

Except it doesn't have 153 categories. It has about 120 — plus duplicates from a form change.

## What happened

The CFPB updated their complaint form in April 2017. Category names were reworded: "my" became "your," abbreviations were expanded. Historical complaints kept their original labels. The dataset captured both.

So "Incorrect information on credit report" and "Incorrect information on your report" sit side by side. Same concept. Different strings. 7,600 examples of one, 7,200 of the other. The model's top confusion pair — by a wide margin — isn't a modeling failure. The labels are broken.

A merge mapping collapses these duplicates: 153 raw labels down to 120. Three lines of code, not a research project.

## The filtering trap

After merging, 120 classes remain with an extreme long tail. Some have 14,000 examples. Seven have fewer than 5. Our first instinct: let students choose their own filtering threshold as a Week 1 data audit exercise. "Given a business scenario, decide what to keep."

We built it. Then we realized it was busywork. A class with 1 example is just noise. No student learns anything by "discovering" that you can't train a classifier on one example. And every student picking a different threshold means different label sets, different metrics, and grading headaches.

We baked in MIN_CLASS_COUNT=5. Seven classes dropped. Sixteen examples lost. 113 canonical classes, same for everyone.

## Where the real exercise lives

The 113 remaining classes have a dramatic imbalance. The top class holds 23% of all training examples. The bottom 30 classes share less than 2%. A model that learns only the head classes gets 54% accuracy and 0.13 macro F1. Accuracy says "not bad." Macro F1 says "you learned nothing useful about 70 of your 113 classes."

That gap — and what to do about it — is what students spend the semester working on. The merge handles noise. The filter removes a non-decision. Students spend their time on the part that's actually judgment.
