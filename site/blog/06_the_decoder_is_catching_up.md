---
title: "44% Accuracy Without a Single Training Example"
layout: default
parent: Blog
nav_order: 6
---

# 44% Accuracy Without a Single Training Example

I gave Opus 4.6 a list of 113 CFPB complaint categories and asked it to classify consumer complaints. No labeled examples in the prompt. No fine-tuning.

44% accuracy. Random chance is 0.9%.

The CFPB complaint database is public. Opus has read it, along with every financial regulation and consumer protection statute on the internet. It hasn't seen the labeled pairs. But it has absorbed the domain as a side effect of pretraining on the open web.

The fine-tuned encoder gets 56.6% after three epochs on 58,000 labeled pairs. Thirty minutes on a free Kaggle T4. That's a 13-point lead.

## The prompt is the only lever, and it has a low ceiling

Without training data, prompt engineering is all you have. I tested five prompting strategies across four models. The full range of prompt variation on Opus was 8 percentage points. Switching to GPT-5.4-Pro at 20x the cost gained zero.

One thing moved the needle: spelling out the institutional distinctions that the category names don't convey. "Incorrect information on your report" means the data is wrong. "Improper use of your report" means unauthorized access. That disambiguation guide took Opus from 36% to 44%. Chain-of-thought reasoning, few-shot examples beyond 2 per class, and batch classification all made things worse.

The pattern in the errors tells you why. Opus gets 100% on "Fraud or scam" and 0% on "Deposits and withdrawals." Categories that mean what they say are easy. Categories that are bureaucratic artifacts of the CFPB's internal taxonomy are impossible from the name alone. The encoder handles both because fourteen thousand labeled examples of "Incorrect information on your report" teach conventions that no amount of general knowledge can replace.

## Thirteen points

The encoder wins today. 56.6% vs 44.0%, 800x cheaper per prediction. Clear recommendation.

But 44% on a 113-class problem without labels or a GPU is not a static number. If the next frontier model picks up another 15 points from scale alone, the encoder's lead disappears. And that model won't need your 58,000 labels, your training pipeline, or your thirty minutes of GPU time to get there. It may not even need you.
