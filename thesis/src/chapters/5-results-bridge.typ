#import "../uet-thesis.typ": *

= Results: vocabulary bridge and decoding cost <ch:results-bridge>

This chapter reports the measurements of the vocabulary bridge (M1), first in the current setup and then in an earlier demonstration that used a different setup. As in @ch:results-alignment, it reports and leaves the interpretation to @ch:discussion.

== Combining the vocabulary bridge with LoRA (P2) <sec:m1-combo>

The question here is whether the bridge and the trained drafter help in the same way or in different ways. To find out, four combinations were replayed on the same 80 validation prompts of pair P2, the first 20 of each task: the untrained drafter and the LoRA drafter, each with the simple text bridge and with the full bridge and its caches. No seed cache was used. Because replay always commits the stored answer, all four produced exactly the same 14,976 tokens, so their counts can be compared directly. @tab:m1-arms gives the results and @fig:m1-arms shows the two most telling columns.

#figure(
  table(inset: 4pt, columns: (1.5cm, 2.5cm, 1.9cm, 1.7cm, 2.0cm, 2.0cm, 2.4cm, 2.0cm), align: (left, right, right, right, right, right, right, right),
    table.header[*Drafter*][*Accepted / proposed*][*Accept.*][*Rounds*][*Tokens per target call*][*Drafter steps*][*Rounds served by cache*][*Modeled cost per token*],
    [B1], [5,092 / 27,186], [18.73%], [9,905], [1.512], [29,686], [0], [0.785],
    [B1 + M1], [5,421 / 26,729], [20.28%], [9,578], [1.564], [23,397], [1,773 (18.5%)], [0.737],
    [T2], [6,793 / 21,929], [30.98%], [8,209], [1.824], [24,514], [0], [0.650],
    [T2 + M1], [6,776 / 22,411], [30.24%], [8,224], [1.821], [19,662], [1,636 (19.9%)], [0.631],
  ),
  caption: [The bridge (M1) with the untrained (B1) and LoRA (T2) drafters on pair P2, 80 validation prompts, three guesses per round, no seed cache. Modeled cost uses the P2 step ratio of @tab:stage1.],
) <tab:m1-arms>

#figure(
  image("../../figures/m1_arms.svg", width: 100%),
  caption: [The four combinations of @tab:m1-arms: (a) tokens produced per target call; (b) drafter steps over the 80 prompts. Hatched bars use the full bridge.],
) <fig:m1-arms>

The two drafters respond differently to the bridge.

With the *untrained drafter*, the bridge helps on both fronts. More of the proposed tokens are accepted, each target call produces a few more tokens, and the drafter is called noticeably less often, because almost a fifth of the rounds are answered from the cache instead.

With the *LoRA drafter*, the cache answers a similar share of rounds, and the drafter is again called about a fifth less often. But the number of tokens per target call does not move, and acceptance even dips slightly. Here the bridge saves drafter work without saving target work.

In both cases the modeled cost, which charges for target calls and drafter steps together, goes down with the bridge. Training the drafter, however, lowers it considerably more than the bridge does. The same four combinations broken down by task are in @app:tables.

Three limits apply to this experiment. It covers 80 prompts, and no confidence interval has been computed yet. It uses no seed cache. And the modeled cost combines counts with a step ratio measured on a different GPU; it is not a measured time.

== Early M1 demonstration (October setup) <sec:october>

Before the current pipeline existed, the bridge was demonstrated in a smaller, different setup. It is reported separately because nearly everything differs from the main experiments: 12 maths test prompts, a target compressed to 4-bit precision, a T4 GPU, answers of at most 128 tokens, and a seed cache built from 16 training questions. Unlike the main experiments, both modes ran with the live target, so this is the only experiment in the project with a measured speed (@tab:october).

#figure(
  table(columns: (1.4fr, 1fr, 1fr), align: (left, right, right),
    table.header[*Measure*][*Simple bridge*][*Full bridge with seed cache*],
    [Accepted / proposed], [580 / 2,651], [623 / 2,615],
    [Acceptance], [21.88%], [23.82%],
    [Generation speed], [8.94 tokens per second], [9.36 tokens per second],
    [Rounds served by cache], [0], [317],
    [Bridge time as a share of one draft step], [41%], [28%],
    [Answers identical to the target alone], [12 / 12], [12 / 12],
  ),
  caption: [Early demonstration on pair P2: 12 maths test prompts, 4-bit target, live decoding on a T4, three guesses per round, at most 128 tokens.],
) <tab:october>

The full bridge raised acceptance on these prompts, with a paired 95% bootstrap confidence interval for the gain of 0.73 to 3.15 percentage points, and generation became about 5% faster. Every answer matched the target's own. The bridge itself was expensive, taking a large fraction of a draft step's time in both modes.

The same 12 prompts were also used to check oracle replay against live decoding, with answers of up to 512 tokens. Two replays, loading only the drafter, reproduced every one of the 3,423 recorded target tokens, and replay acceptance (27.89%) was almost identical to live acceptance (27.87%). On one of the 12 prompts the live run with the full bridge ended with a different answer from the target alone.
