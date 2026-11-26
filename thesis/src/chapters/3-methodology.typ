#import "../uet-thesis.typ": *
#import "../diagrams.typ": *

= Methodology <ch:method>

This chapter has two parts. @sec:method describes the method itself: how proposals are produced, translated, cached and checked, and how the drafter is taught to imitate the target. @sec:design describes how the method is evaluated. @sec:tiers first gives the overall picture.

== Overview and tiers <sec:tiers>

SpecAdapt works entirely on the drafter's side of speculative decoding and leaves the target untouched. It has two working modules.

- *M1, the vocabulary bridge*, deals with pairs whose tokenizers differ. It translates the drafter's guesses into target tokens, remembers translations it has already made, and, where the text so far makes the continuation predictable, proposes tokens from a cache without asking the drafter at all.
- *M2, logit alignment*, deals with the drafter predicting differently from the target. It trains a small adapter so that the drafter's first choices agree with the target's more often, and then folds the adapter into the drafter's weights.

A third module, which would choose the number of guesses per round adaptively, is planned and is described as future work in @sec:future.

The experiments compare a ladder of configurations, summarised in @tab:tiers. Each rung adds one ingredient, so the effect of each can be read off separately.

#figure(
  table(columns: (auto, 1fr), align: (left, left),
    table.header[*Run*][*What it is*],
    [B0], [The target alone, one token per forward pass. The reference for speed and for the correct output.],
    [B1], [Plain speculative decoding with the untrained drafter and three guesses per round. For pairs with different tokenizers, guesses are translated with the simple text bridge of @sec:b1-bridge.],
    [B1 + M1], [As B1, with the full vocabulary bridge and continuation caches of @sec:m1.],
    [T1], [As B1, with the drafter's output layer adapted (@sec:m2).],
    [T2], [As B1, with LoRA adapters throughout the drafter (@sec:m2).],
    [T2 + M1], [The LoRA drafter combined with the full vocabulary bridge.],
  ),
  caption: [The configurations compared in this thesis.],
) <tab:tiers>

@fig:system shows how the pieces fit together in one round.

#figure(fig-round-flow, caption: [One SpecAdapt round. When a cache recognises the current ending, the drafter is skipped; when a guess has been translated before in the same context, the stored translation is reused. Every proposal, however it was produced, is checked by the target.]) <fig:system>

== The SpecAdapt method <sec:method>

=== Notation and prompt formats <sec:notation>

Throughout, $K$ is the largest number of tokens proposed in one round and $N$ the largest number of tokens in an answer. Each verification by the target is a _round_.

The two models see the same conversation in different clothes. The target sees the user's prompt wrapped in its own chat template, exactly as it would in normal use. Vicuna's tokenizer has no chat template, so for Vicuna the standard Vicuna v1.1 system prompt is used instead. The drafter sees something simpler: the plain prompt, a blank line, and then the text the target has committed so far. That text is decoded from the target's tokens and re-encoded with the drafter's tokenizer in every round, so the drafter always continues exactly the text the target has accepted, whatever the two tokenizers make of it. The prompt formats for each dataset are listed in @app:prompts.

=== Baseline: naive text bridge (B1) <sec:b1-bridge>

The simplest way to move a guess from one vocabulary to another is to go through text: turn the drafter's tokens into characters, then cut those characters into the target's tokens. Two details make this work reliably.

First, the guess is decoded together with the text before it and only the new characters are kept. Decoding the guessed tokens on their own would drop a leading space.

Second, the new characters are re-encoded together with the last few committed target tokens. This lets a piece that straddles the boundary, such as the end of a word the target has started, be split the way the target would split it. If the committed tokens come back unchanged, the tokens after them form the proposal; if not, the new characters are encoded on their own, so committed tokens are never rewritten. At most $K$ target tokens are kept. If nothing sensible comes out, the proposal is empty and the target simply writes one token of its own that round.

@fig:text-bridge summarises these steps.

#figure(fig-text-bridge, caption: [The simple text bridge used by the baseline B1. The committed target tokens are never rewritten.]) <fig:text-bridge>

For the pair that shares a tokenizer, no translation is needed: the drafter's tokens are passed to the target unchanged.

=== M1: vocabulary bridge and continuation caches <sec:m1>

The full bridge adds memory to the baseline in three ways, and its settings are collected in @tab:m1-settings.

*Remembering translations.* Each translation is stored, keyed by the guessed draft tokens together with the last few draft and target tokens around them. When exactly the same guess appears again in the same surroundings, the stored translation is reused instead of going through text again. Including the surroundings in the key matters: the same draft tokens can translate differently after a space than inside a word.

*Recognising continuations.* Text often repeats itself: a number from a maths question may reappear in the working, a variable name may recur throughout a piece of code, a phrase from an article may return in its summary. The continuation cache exploits this. It records, for every short run of target tokens it has seen in the prompt and in the committed answer, which tokens came next. At the start of each round it looks for the longest recorded run that matches the end of the committed text. If it finds one, the tokens that followed last time become the proposal and the drafter is not called. Only committed tokens are recorded, never rejected guesses, and the target still checks every cached proposal like any other. @fig:cache shows the idea on a small example.

#figure(fig-cache, caption: [The continuation cache at work, with illustrative tokens. The end of the answer matches a run of tokens in the prompt, and the token that followed it there becomes the proposal, without calling the drafter.]) <fig:cache>


*Checking direct overlaps.* Some plain ASCII pieces exist in both vocabularies. The bridge builds a map between such identical pieces. In the current implementation this map is used as a consistency check alongside the text translation, not as a shortcut that replaces it.

An optional fourth part, a read-only _seed cache_, holds continuations taken from the target's answers to separate training questions, so that useful continuations are available before the current answer has repeated anything. It was used in the early demonstration of @sec:october but not in the main experiments, because it is built from maths training answers drawn from the same dataset as the maths validation prompts.

#figure(
  table(columns: (1fr, auto), align: (left, left),
    table.header[*Setting*][*Value*],
    [Context in the translation key], [last 16 target tokens and last 4 draft tokens],
    [Capacity of the translation memory], [4,096 entries, cleared when full],
    [Length of the endings recorded by the continuation cache], [2 to 8 target tokens],
    [Tokens proposed from a cache hit], [up to $K$],
    [Seed cache endings / continuation], [3 to 8 tokens / the most frequent full continuation],
    [Seed cache used over the request cache], [only for a strictly longer match],
  ),
  caption: [Settings of the vocabulary bridge.],
) <tab:m1-settings>

@alg:round puts these parts together in the order they are tried.

#algorithm(caption: [One SpecAdapt round with the vocabulary bridge (greedy, at most $K$ guesses).])[
  #set par(justify: false, first-line-indent: 0pt)
  #set text(size: 11pt)
  *Input:* the prompt; the committed target tokens $y$; the request cache; the translation memory; an optional seed cache; the number of tokens still allowed $r$. \
  *Output:* the newly committed target tokens $u$, at least one and at most $min(K + 1, r)$.
  #v(4pt)
  ```text
   1  s := longest match for the end of y in the request cache
   2  if the seed cache has a strictly longer match:
   3      s := the seed cache's continuation
   4  if s is empty:                              # no cached continuation
   5      c := draft encoding of (prompt + blank line + text of y)
   6      d := up to K greedy draft tokens after c
   7      key := (last 16 target tokens, last 4 tokens of c, d)
   8      if key is in the translation memory:
   9          s := the stored translation
  10      else:
  11          s := text translation of d (Section 3.2.2)
  12          store s under key
  13  s := first r tokens of s
  14  p := the target's greedy choices after y, for each position of s and one more
  15  a := number of leading tokens of s that equal p
  16  u := first a tokens of s, followed by p[a] if room remains
  17  return u, cut after the first end-of-answer token
  ```
] <alg:round>

After each round the new tokens are appended to the answer and to the request cache. Generation ends when the target writes its end-of-answer token or the length limit is reached. Without M1, steps 1 to 3 and the translation memory are skipped and the drafter is called in every round.

=== M2: logit alignment <sec:m2>

An untrained drafter fails mostly because it would have written something different from the target. Logit alignment teaches it to write like the target.

*What it learns from.* The teacher is the target itself. For every training prompt, the target's own greedy answer is recorded once, and the drafter is trained to predict that answer token by token, not the reference answer that comes with the dataset. This is exactly the skill that speculative decoding rewards: agreeing with the target's first choice. @fig:training-data shows how one training example is put together. The training text is the drafter's view of the prompt followed by the target's answer, and the loss counts only the answer tokens, not the prompt. When the target ended its answer on its own, the drafter also learns to end there; when the target was cut off by the length limit, it does not.

#figure(fig-training-data, caption: [How a training example is built. The target answers each training prompt once; the drafter then learns to predict that answer, token by token, after its own view of the prompt.]) <fig:training-data>


*Two adapters.* Two ways of adapting the drafter are compared. They are of almost the same size, so that the comparison is about _where_ the adaptation sits, not how big it is (@tab:adapters). @fig:adapters shows the two placements in the drafter, which has two blocks.

- The *output adapter (T1)* leaves the whole drafter frozen and changes only its final step, where the hidden state $h$ is turned into a score for every word in the vocabulary. It adds a learned bias $b$ for each vocabulary entry and a low-rank correction to the output weights $W$:

$ z = W h + U (D h) + b. $ <eq:t1>

  Here $D$ compresses the hidden state to a handful of numbers (the rank) and $U$ expands them back to the vocabulary.

- The *LoRA adapter (T2)* instead places a small low-rank correction inside every linear layer of every block of the drafter: the attention projections and the feed-forward projections alike. Each weight matrix $W$ becomes

$ W' = W + U D, $ <eq:lora>

  with $U$ starting at zero, so training begins from the untrained drafter.

#figure(fig-adapters, caption: [Where the two adapters act in the 68M drafter. (a) The output adapter changes only the final layer that scores the vocabulary. (b) LoRA adds a small correction to each of the seven linear projections in both blocks.]) <fig:adapters>

*Free at run time.* After training, both adapters are folded into the drafter's own weights: the correction $U D$ is added to $W$ once and for all. The adapted drafter then has exactly the same shape as the original and runs at the same speed. As a safety check, the training job compares the drafter's outputs before and after folding and stops if they differ by more than a tiny tolerance; the measured differences were far below it (@tab:adapters).

#figure(
  table(columns: (1fr, 1fr, 1fr), align: (left, left, left),
    table.header[][*T1, output adapter*][*T2, LoRA*],
    [Where it acts], [final output layer only], [all seven linear projections in every block],
    [Rank], [8 or 16 (selected: 16)], [16],
    [Trainable parameters (68M drafter)], [556,288], [565,248],
    [Weight tensors], [3], [28],
    [Folded into the drafter], [yes], [yes],
    [Largest output change when folded], [$3.8 times 10^(-5)$], [$4.2 times 10^(-5)$],
  ),
  caption: [The two alignment adapters on the 68M drafter of pair P2. The folding check stops a run if the change reaches $10^(-3)$.],
) <tab:adapters>

Both adapters are trained with the same recipe, summarised in @tab:training.

#figure(
  table(columns: (1fr, 1fr), align: (left, left),
    table.header[*Setting*][*Value*],
    [Optimiser], [AdamW, no weight decay],
    [Learning-rate schedule], [linear warm-up over the first 5% of steps, then cosine decay],
    [Precision], [32-bit for training, 16-bit for replay],
    [Batches], [sorted by length; at most 16 sequences and 16,000 tokens],
    [Random seed], [42],
  ),
  caption: [Training settings shared by both adapters.],
) <tab:training>

=== Run-time cost of the adapters <sec:adapter-cost>

Because both adapters are folded into the drafter, they add no run-time cost: the adapted drafter does exactly the same amount of work per step as the original.

The vocabulary bridge is a different matter. It is ordinary program logic that runs between the drafter and the target, and its cost is real. The only measurement of it so far comes from the early demonstration (@tab:october), in a different setup, where the bridge took well over the intended tenth of a draft step. Keeping the bridge within budget therefore remains an open engineering problem.

== Experimental design <sec:design>

@fig:pipeline gives the whole experimental pipeline at a glance. The sections below follow it in order: the model pairs and data, the one-off recording of the targets' answers, training and selection, and the measures reported.

#figure(fig-pipeline, caption: [The experimental pipeline. The targets' answers are recorded once; everything after that runs on the drafter alone. The test set is held back for the final evaluation.]) <fig:pipeline>

=== Model pairs <sec:pairs>

Three draft–target pairs are studied (@tab:pairs). They are chosen so that each comparison changes one thing. P1 and P2 use the same drafter with two different targets, one sharing its tokenizer and one not. P2 and P4 use the same target with two drafter sizes. P2, the cross-tokenizer pair with the smaller drafter, is the primary pair and the only one trained so far.

#figure(
  table(columns: (auto, 1.6fr, 1fr, 1fr), align: left,
    table.header[*Pair*][*Drafter → target*][*Purpose*][*State*],
    [P1], [Llama-68M → Vicuna-7B v1.5], [same tokenizer; control], [answers recorded],
    [P2], [Llama-68M → Qwen2.5-7B-Instruct], [different tokenizers; primary], [trained and evaluated],
    [P4], [Llama-160M → Qwen2.5-7B-Instruct], [larger drafter], [answers recorded],
  ),
  caption: [Draft–target pairs. The exact model versions are listed in @app:env.],
) <tab:pairs>

Two further pairs with a Llama-3.1-8B target are planned if model access and budget allow; they are not evaluated in this thesis.

=== Datasets and splits <sec:data>

The prompts cover four kinds of task, so that results are not tied to one style of text: grade-school maths word problems (GSM8K), programming (HumanEval and MBPP), general instructions (Alpaca) and one-sentence news summaries (XSum).

The prompts are divided into separate sets with distinct jobs (@tab:splits). A training set teaches the adapters. A validation set chooses between adapter settings and is the set on which all results in this thesis are reported. A test set is held back untouched for the final evaluation, and a small smoke set is used for quick live checks.

#figure(
  table(columns: (auto, 1fr, auto), align: (left, left, right),
    table.header[*Set*][*Used for*][*Prompts per target*],
    [Training], [training the adapters], [1,440 (360 per task)],
    [Validation], [choosing settings; all results reported here], [320 (80 per task)],
    [Test], [final held-out evaluation (not yet used)], [400 (100 per task)],
    [Smoke], [quick live checks], [40],
    [Development], [the early demonstration only], [12],
  ),
  caption: [Prompt sets. The dataset slices behind each set are in @app:prompts.],
) <tab:splits>

The sets are built by a fixed, deterministic procedure. Exact duplicates are removed first. The test prompts are chosen next, before anything else, and any other prompt that is too similar to a test prompt of the same task (a word-overlap Jaccard similarity of 0.8 or more) is discarded, so that near-copies of test questions cannot leak into training. The remaining prompts are shuffled with a fixed seed and sliced into the other sets. When the training job loads its data, it also checks that no prompt appears in both the training and the validation set.

=== Oracle recording and replay <sec:oracle>

Measuring acceptance normally means running both models together, and the 7B target is by far the expensive part. Yet in greedy speculative decoding the target's answer never depends on the drafter (@sec:lit-spd). That observation makes a much cheaper method possible, shown in @fig:oracle.

*Record once.* Each target writes its greedy answer to every prompt once, in 16-bit precision without quantisation, and the answer is stored together with everything needed to check later that nothing has changed: the exact model and tokenizer versions, the prompt's tokens, the software versions and the hardware.

*Replay many times.* To evaluate a drafter, only the drafter is loaded. It runs speculative decoding as usual, but each proposal is checked against the stored answer instead of a live target: a guess is accepted exactly when it matches the stored token at that position. Because the stored answer decides what is committed, the output of every replay is the stored answer itself; the replay stops with an error if it ever is not. Only the counts of proposed and accepted tokens differ from one drafter to another, which is precisely what is being measured.

#figure(fig-oracle, caption: [Oracle record-then-replay. The expensive target runs once per prompt; any number of drafters are then evaluated against its stored answers without loading it.]) <fig:oracle>

This makes the comparison both cheap and exact: every drafter is judged against the very same answers. It has one clear limit. Replay counts tokens; it does not time anything. Speed-ups, tokens per second and a check that a live target really produces the same text all require running the target live, and are kept strictly apart from replay results. Where recording and replay ran is summarised in @tab:compute.

#figure(
  table(columns: (auto, 1fr, 1fr), align: left,
    table.header[*Step*][*Hardware*][*Setting*],
    [Recording, test prompts], [NVIDIA A10 (Modal)], [16-bit, one prompt at a time],
    [Recording, training and validation prompts], [NVIDIA A10G (Modal)], [16-bit, batches of 16],
    [Replay and training], [NVIDIA T4 (Kaggle)], [16-bit replay, one prompt at a time],
  ),
  caption: [Where each step ran. All answers are greedy and at most 256 tokens long.],
) <tab:compute>

=== Training and selection protocol <sec:protocol>

Each adapter was trained with a small grid of settings (@tab:grid), and the best setting was chosen by how well the drafter imitates the target on the validation prompts.

The selection score is _top-1 agreement_: feed the drafter the target's validation answers one token at a time and count how often the drafter's first choice is the target's actual next token. This is a quick, direct measure of the skill acceptance depends on. The chosen adapter is then folded into the drafter and replayed on the same validation prompts to measure acceptance.

#figure(
  table(columns: (auto, auto, auto, auto, auto), align: left,
    table.header[*Adapter*][*Rank*][*Learning rates tried*][*Training length*][*Scored after*],
    [T2, LoRA], [16], [$10^(-4)$, $3 times 10^(-4)$, $10^(-3)$], [4 epochs], [epochs 2 and 4],
    [T1, output], [8, 16], [$10^(-3)$, $3 times 10^(-3)$, $10^(-2)$], [2 epochs], [epoch 2],
  ),
  caption: [Settings tried for pair P2: six scored candidates per adapter.],
) <tab:grid>

One consequence should be stated plainly: the same validation prompts are used both to choose the setting and to report the result, which may make the reported numbers somewhat optimistic. The test prompts were set aside for exactly this reason and have not been looked at by any training or selection step.

=== Metrics <sec:metrics>

Four measures are used. Counts are always added up over all prompts before dividing, so long answers weigh more than short ones. Write $A$ for accepted tokens, $P$ for proposed tokens, $R$ for rounds (each one target call), $G$ for generated tokens and $S$ for drafter steps.

*Acceptance* is the share of proposed tokens the target accepted:

$ "acceptance" = A / P. $ <eq:acceptance>

*Tokens per target call* is how many tokens each expensive target pass produces on average. Target-only decoding scores exactly one; anything above one is the saving speculative decoding is after:

$ "tokens per target call" = G / R. $ <eq:tpc>

Acceptance alone can mislead, because a cache hit or a shorter proposal changes how many tokens are proposed. Tokens per target call does not have this problem, which is why both are reported.

*Modeled cost per token* also charges for the drafter's work. One drafter step costs a fraction $c$ of a target step, where $c$ was measured on live runs for each pair (@tab:stage1):

$ "modeled cost per token" = (R + c dot S) / G. $ <eq:cost>

It is expressed in target steps per generated token, so target-only decoding costs one and lower is better. It is built from counts and a separately measured ratio, not from a timing of the run itself.

*Top-1 agreement*, the selection score, is the share of the target's answer tokens that the drafter ranks first when given the correct preceding text:

$ "top-1 agreement" = M / T, $ <eq:top1>

where $T$ is the number of answer tokens and $M$ the number the drafter ranks first.

=== Infrastructure and reproducibility <sec:infra>

The study was designed to run on modest, mostly free resources. Training and all replay ran on a single NVIDIA T4 on Kaggle; only the recording of the 7B targets' answers ran on rented A10-class GPUs on Modal, at a recorded cost of 3.52 US dollars. Every model, dataset and software package is pinned to an exact version, and every result records the code version, the model and dataset versions, the tokenizer fingerprint and the hardware it was recorded and replayed on (@app:env). The automated test suite passed in full on the code version of the last run. The code, the experiment definitions and the scripts that produce every figure in this thesis are available in the project repository @specadapt-repo.
