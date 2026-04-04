# When Your "Ready to Use" Dataset Has the Same Category Listed Twice

`determined-ai/consumer_complaints_medium` is a good teaching dataset. Pre-split, business-realistic, multiclass, compute-friendly. 64,000 training examples. 153 issue categories.

Except it doesn't have 153 categories. It has about 111 — plus 42 duplicates wearing different hats.

## What happened

The Consumer Financial Protection Bureau updated their complaint submission form in April 2017. They rewrote category names: first-person became second-person ("my" → "your"), abbreviations were expanded ("Cont'd" → full words), some categories were restructured. Historical complaints kept their original labels. The dataset captured both.

So "Incorrect information on credit report" and "Incorrect information on your report" sit side by side in the label column. Same concept. Different strings. 7,607 examples of one, 7,208 of the other.

The model's top confusion pair — by a wide margin — is between these two categories. The model isn't confused. The labels are.

## Why this matters for a course

This is for a class where students fine-tune one model across six weeks. They'll train on this data, analyze errors, apply compression, write memos about what's working and what isn't. If the label set is broken, every downstream analysis is contaminated.

But there's a question about where the fix belongs.

**Option 1: Clean it up before students arrive.** Distribute a fixed dataset. Students never see the problem. Pro: they focus on modeling, not data janitorial work. Con: they miss a lesson that's arguably more important than any modeling technique — that real datasets have problems you have to find before you can model.

**Option 2: Let students discover it.** Hand them the raw 153-class dataset. Watch them struggle with a model that can't distinguish between identical categories with different names. Hope they figure it out. Con: that's a lot of wasted GPU time and frustration for a lesson that could be taught in 20 minutes.

**Option 3: Split the difference.** Provide the merge mapping as a course artifact — a dictionary in the notebook that collapses 153 labels down to 111. Students see it applied. They understand why. They don't have to reverse-engineer the CFPB's form change history. Three lines of code, not a research project.

Then the real exercise starts: what do you do with 111 classes when 50 of them have fewer than 100 training examples?

## The interesting question

Once the duplicates are merged, the dataset still has an extreme long tail. Some classes have 14,000 examples. Others have 5. The data audit question — what to keep, what to drop, where to draw the line — is genuinely interesting in a way that debugging label schema drift is not.

Give students a business scenario: "This model routes complaints to the right department. Departments exist for categories that receive meaningful volume." Now the filtering has a rationale. A class with 14,000 examples stays. A class with 3 examples — do you keep it? Group it? Drop it? The answer depends on the scenario, and the student has to justify their choice.

This is the kind of decision that working data scientists make constantly and rarely get taught explicitly. Most courses hand students a clean dataset and skip straight to `model.fit()`. The messy middle — is this label set actually what I want to model? — is where a lot of real-world ML time goes.

The merge handles the part that's just noise. The filtering handles the part that's actually judgment. Students do the judgment part in class (it's thinking work, not GPU work), then go home and train on their chosen dataset.

## What this changes downstream

The class count a student picks affects everything that follows:

- **Week 2:** Class weighting with 111 classes and weights ranging from 0.08x to 17.5x collapsed the model to near-random. With fewer classes or a higher minimum count, weighting becomes tractable.
- **Week 4:** The rare-vs-common slice analysis is more dramatic with a long tail. Filtering too aggressively makes this slice boring. Not filtering enough makes the model's macro-F1 hard to interpret.
- **Week 5:** The operating point memo asks "would you ship this?" The class count is part of that question — did the student make a defensible choice about what to model?

The memo at the end of Week 1 now includes: "Here's my data audit. Here's what I chose to keep and why." That's the first line of a technical argument that runs through the rest of the course.

The course runs April and May 2026 at CEU Vienna. Previous post: [Model Cards Won't Tell You If Your Course Will Work](#).
