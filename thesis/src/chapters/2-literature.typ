#import "../uet-thesis.typ": *
#import "../diagrams.typ": *

= Literature review <ch:literature>

This chapter sets out the two ideas the method builds on: how greedy speculative decoding works, and why it becomes difficult when the draft and target models do not share a vocabulary. The descriptions follow the implementation used throughout the thesis.

== Speculative decoding <sec:lit-spd>

In ordinary greedy decoding the target model produces one token per forward pass: it reads everything written so far and picks its most likely next token. Speculative decoding keeps that rule for what is finally written but changes how the target's passes are spent. @fig:timeline contrasts the two schemes: instead of one target pass per token, a few cheap drafter steps are followed by one target pass that can write several tokens at once.

#figure(fig-timeline, caption: [Target-only decoding versus speculative decoding. Schematic with illustrative counts; not to scale.]) <fig:timeline>

One round works as follows.

+ *Draft.* The small model guesses the next few tokens, up to a fixed number $K$, one at a time, each time taking its own most likely choice. It stops early if it believes the answer is finished.
+ *Verify.* The target reads the committed text followed by all the guesses in a single forward pass. Because a transformer predicts every position at once, this one pass tells what the target would have chosen at each guessed position, and one position beyond.
+ *Accept.* Guesses are kept from the left for as long as they agree with the target's choices. The first disagreement ends the run.
+ *Correct.* The target's own choice at the first disagreement is written as well, or, if every guess was accepted, the target's choice for the next position. Every round therefore writes at least one token.
+ *Roll back.* Both models cache intermediate results for the text they have read. Entries for rejected guesses are dropped, so both caches again end exactly at the committed text.

@fig:round-example walks through one round with made-up tokens. The drafter's first two guesses agree with the target and are kept; the third differs, so the target's own token replaces it, and the round writes three tokens for a single target pass.

#figure(fig-round-example, caption: [One greedy speculative round, worked example with illustrative tokens. Light grey: committed text; dark grey: the target's correction.]) <fig:round-example>

Since only tokens that match the target's own choices are kept, and the target supplies the token at every disagreement, the committed text is the target's own greedy answer, whatever the drafter proposes. The drafter cannot change the answer; it can only change how many target passes are needed to produce it. A good drafter lets one pass write several tokens. A poor one makes every pass write a single token while still paying for the drafting.

In practice a speculative run can still end up with a slightly different answer from a plain target-only run on a small share of prompts. @sec:replay-validity reports how often this happened here; its cause was not investigated.

== Heterogeneous pairs: tokenizer and distribution mismatch <sec:lit-hetero>

A draft model and a target model can differ in two ways.

They can disagree about the next token while sharing a vocabulary. The drafter is then simply a weaker predictor, and the problem is one of distribution mismatch: its first choices do not match the target's often enough. This thesis addresses it by training the drafter to imitate the target (@sec:m2).

They can also split text into different pieces. Every model reads text as a sequence of tokens drawn from its own vocabulary, numbered in its own way. The 68-million-parameter Llama drafter used here has a SentencePiece vocabulary of 32,000 pieces, while the Qwen target uses a byte-level BPE vocabulary with its own pieces and its own numbering. Even a single space is written differently: Llama marks a word-initial space with "▁", while Qwen uses "Ġ", and "Ċ" for a newline. A drafter's guess, expressed as Llama token numbers, therefore means nothing to the Qwen target. It has to be translated into the target's tokens before it can be checked. @fig:tokenizers illustrates the problem on a short piece of text.

#figure(fig-tokenizers, caption: [The same text split by two different vocabularies. Illustrative split; the space markers are those used by the two tokenizers.]) <fig:tokenizers>



The translation is not trivial. Text that one tokenizer splits into two pieces, such as "st" and "amps", may be a single piece, "stamps", for the other, and the right split of a guess can depend on the last few characters already committed. A careless translation can also rewrite tokens the target has already committed, which would break the verification step. @sec:b1-bridge and @sec:m1 describe how this thesis handles both issues.

The pairs studied here cover both situations: one pair shares a vocabulary and differs only in its predictions, and the others differ in both (@tab:pairs).
