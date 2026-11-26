#import "../uet-thesis.typ": *

= Conclusion and future work <ch:conclusion>

== Conclusion <sec:conclusion>

This thesis asked how far speculative decoding between mismatched models can be improved by working on the drafter's side alone, leaving the large target model untouched. It built three things to study the question: a vocabulary bridge that translates the drafter's guesses into the target's tokens and proposes repeated text from a cache; two alignment adapters, one on the drafter's output layer and one spread through all its layers, that teach the drafter to imitate the target and are then folded into it at no run-time cost; and an oracle replay method that measures any drafter against the target's recorded answers without running the target again.

On the primary pair, a small Llama drafter with the Qwen2.5-7B-Instruct target, the LoRA adapter made the largest difference, raising acceptance from under one in five to almost one in three on the validation prompts. The output adapter, of almost the same size, achieved roughly a third of that gain. The vocabulary bridge helped an untrained drafter, but once the drafter was trained it mainly replaced drafter calls rather than adding accepted tokens.

Measured against the research question, these results do not support the hope that training-free or very lightweight adaptation recovers most of the gain: for acceptance on this pair, gradient training throughout the drafter matters most. The conclusion is limited to acceptance on validation prompts of one pair. Held-out test results, live speed measurements and an adaptive draft length are still to come, and the threats listed in @sec:threats apply.

== Future work <sec:future>

The remaining work, in the planned order, is as follows.

+ *More pairs.* Train and evaluate the same-tokenizer pair P1 and the larger-drafter pair P4 with the settings chosen for P2.
+ *Held-out test evaluation.* Freeze the chosen adapters and evaluate them once on the 400 untouched test prompts per pair, with bootstrap confidence intervals.
+ *Adaptive draft length.* Let the drafter decide how many tokens to propose in each round, first with a simple confidence threshold and then with a small learned model using cheap signals such as the drafter's uncertainty and whether the cache was hit, chosen to minimise cost per token rather than to maximise acceptance.
+ *Live speed measurements.* Run every configuration with the real target on the core test prompts to measure the actual speed-up, the bridge's overhead against its budget, and how often the answer matches the target's own.
+ *Ablations.* Measure how the results change with less training data, with one task left out of training, with larger seed caches, and without caches.
+ *Further targets.* Add pairs with a Llama-3.1-8B target if access and budget allow.
+ *Sampling-based decoding.* Implement the verification rule for sampled rather than greedy answers.
+ *A wider search.* Try larger learning rates and longer training, since the best settings found lie at the edge of the grid.
