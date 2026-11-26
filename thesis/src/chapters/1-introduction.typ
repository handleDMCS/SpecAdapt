#import "../uet-thesis.typ": *

= Introduction <ch:intro>

== Motivation <sec:motivation>

A large language model writes its answer one token at a time. Each new token needs a full forward pass through the model, so a long answer means a long chain of expensive passes, one after another.

Speculative decoding (SpD) shortens this chain by letting a much smaller model do the guessing. The small _draft_ model quickly proposes the next few tokens. The large _target_ model then checks all of them in a single pass and keeps the longest run of guesses that agrees with what it would have written itself, plus one token of its own. When the guesses are good, one expensive pass produces several tokens. When they are poor, the pass produces only one, and the time spent drafting is wasted.

Everything therefore depends on how often the drafter's guesses are accepted. That is easiest when the two models are close relatives: trained on similar data and splitting text into the same tokens. In practice the most convenient small model is often not a relative of the large one. Its vocabulary can differ, so its guesses must first be translated into the target's tokens, and its sense of what comes next can differ, so even well-translated guesses are often rejected.

This project gives a concrete example. A 68-million-parameter Llama drafter paired with the Qwen2.5-7B-Instruct target, which uses a different tokenizer, has fewer than one in five of its proposed tokens accepted (@tab:alignment-headline). Most of the drafting work is thrown away. The question this thesis takes up is how much of that loss can be won back by changing only the drafter's side.

== Problem statement and research question <sec:problem>

The setting is draft-side adaptation. The target model is treated as fixed: its weights, its tokenizer and its outputs are never changed, and every proposal is still checked by it, so the final text is the target's own. All changes happen on the small model's side, in the drafter itself and in the code that translates and caches its proposals. Because that code runs in every drafting round, it must also be cheap; the project set itself a budget of at most a tenth of the time of one draft step for all adapter overhead.

The research question is:

#quote(block: true)[_How much of the acceptance-rate and speedup improvement of adaptive speculative decoding can be recovered using training-free or very lightweight draft-side adaptation?_]

The project started from the hypothesis that the training-free and lightweight options would recover most of the gain. The thesis does not assume this; it measures it. Two clarifications shape the answer. First, of the options studied, only the vocabulary bridge with its caches is truly training-free; the lightweight output adapter is small and touches only the drafter's final layer, but it is still trained with gradients offline. Second, the speed-up half of the question requires timing a live target with the trained drafters, which has not been done yet, so the answer given in @sec:answer concerns acceptance.

== Contributions <sec:contributions>

The thesis makes four contributions.

+ *Oracle replay for evaluating drafters.* The target writes its answer to every prompt once, and the answer is stored. Any drafter can then be scored against the stored answer without running the target again. Because the stored answer decides what is committed, the output text is identical in every replay and only the acceptance counts change. This makes comparing many drafters cheap and exact.
+ *A vocabulary bridge with continuation caches* for pairs whose tokenizers differ. It translates proposals through text without disturbing tokens already committed, remembers translations it has already made, and proposes likely continuations straight from the prompt and from verified output, skipping the drafter when it can.
+ *Draft-side logit alignment in two sizes.* A small output adapter on the drafter's final layer and a LoRA adapter spread over all of its layers are trained to imitate the target's own answers, then merged into the drafter so that they cost nothing extra at run time.
+ *A reproducible, low-cost experimental pipeline.* Every model, dataset and package version is pinned, every result records where it came from, and the whole study runs on free or low-cost cloud GPUs. The code is available in the project repository @specadapt-repo.

== Thesis overview <sec:overview>

@ch:literature describes the background the method builds on: how greedy speculative decoding works, and why draft and target models with different vocabularies are hard to combine.

@ch:method presents the method and the experimental design: the vocabulary bridge, the two alignment adapters, the model pairs and data, the oracle replay, and the way configurations are trained, selected and measured.

@ch:results-alignment reports the measurements for the untrained drafters and for the two alignment adapters. @ch:results-bridge reports the vocabulary bridge, alone and together with the best adapter, as well as an earlier demonstration in a different setup. These two chapters report what was measured without interpreting it.

@ch:discussion interprets the results, answers the research question as far as the evidence allows, and lists the threats to validity. @ch:conclusion summarises the findings and the work that remains. The appendices give the run register, the full tables, the software environment, and the prompt formats and data splits.
