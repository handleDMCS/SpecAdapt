#import "../uet-thesis.typ": *

= Results: baselines and logit alignment <ch:results-alignment>

This chapter reports what was measured for the untrained drafters and for the two alignment adapters. It describes what the tables and figures show and leaves the interpretation to @ch:discussion. Unless stated otherwise, every acceptance figure is an oracle replay on validation prompts, with greedy decoding, at most three guesses per round and answers of up to 256 tokens. None is a test-set result and none is a measured speed.

== Replay validity and cost ratios (stage 1) <sec:replay-validity>

Before any training, all three pairs were run live, with the real target, on the 40 smoke prompts. This check served three purposes: to confirm that the untrained drafters are usable at all, to measure how expensive a drafter step is compared with a target step, and to see whether speculative decoding reproduces the target's own answers in practice. @tab:stage1 gives the results.

#figure(
  table(columns: (auto, 1.2fr, 1fr, 1.2fr), align: (left, right, right, right),
    table.header[*Pair*][*Live acceptance, untrained drafter*][*Drafter step ÷ target step*][*Prompts whose answer matched the target alone*],
    [P1], [25.19% (2,582 / 10,250)], [0.058], [39 / 40],
    [P2], [20.20% (2,454 / 12,146)], [0.062], [34 / 40],
    [P4], [26.35% (2,878 / 10,921)], [0.231], [36 / 40],
  ),
  caption: [Live check on 40 smoke prompts, 16-bit targets on an NVIDIA A10, three guesses per round. The step ratio includes the bridge's time.],
) <tab:stage1>

A drafter step is cheap relative to a target step for all three pairs, and markedly more so for the 68M drafter than for the 160M one. For reference, the targets themselves produced about 28 tokens per second when answering one prompt at a time on this GPU (@tab:target-speed).

#figure(
  table(columns: (auto, 1fr, 1fr), align: (left, right, right),
    table.header[*Target*][*One prompt at a time*][*Batch of 32 prompts*],
    [Qwen2.5-7B-Instruct], [28.3 tokens per second], [199.5 tokens per second],
    [Vicuna-7B v1.5], [28.7 tokens per second], [79.0 tokens per second],
  ),
  caption: [Target-only generation speed on an NVIDIA A10, 16-bit.],
) <tab:target-speed>

Speculative decoding did not reproduce the target-only answer on every prompt: a few prompts per pair ended with a different answer (last column of @tab:stage1). The project's own requirement for the later live evaluation is agreement on at least 99% of prompts, so this is an open issue, taken up in @sec:threats.

The same prompts were then replayed against the recorded answers, and the project log records that replay acceptance agreed with live acceptance to within half a percentage point for all three pairs, the tolerance set in advance. This is the evidence that replay measures what a live run would.

== Untrained baseline (B1) <sec:b1>

The untrained drafter of the primary pair P2 has about one in five of its proposals accepted, with some variation between tasks: maths is the easiest and code the hardest (@tab:b1-task).

#figure(
  table(columns: (auto, 1fr, auto), align: (left, right, right),
    table.header[*Task*][*Accepted / proposed*][*Acceptance*],
    [Maths], [7,000 / 31,948], [21.91%],
    [Code], [6,006 / 36,951], [16.25%],
    [Instruction], [4,118 / 22,468], [18.33%],
    [Summarization], [1,192 / 6,950], [17.15%],
    [*All*], [*18,316 / 98,317*], [*18.63%*],
  ),
  caption: [Untrained drafter (B1) on pair P2, 320 validation prompts.],
) <tab:b1-task>

The other two pairs have so far been replayed untrained only on a 40-prompt validation sample (@tab:b1-40). On that sample the same-tokenizer pair P1 has the highest acceptance and P2 the lowest, with the larger drafter of P4 in between.

#figure(
  table(columns: (auto, 1fr, auto), align: (left, right, right),
    table.header[*Pair*][*Accepted / proposed*][*Acceptance*],
    [P1, same tokenizer], [3,133 / 11,110], [28.20%],
    [P2, different tokenizers], [2,610 / 13,938], [18.73%],
    [P4, larger drafter], [3,061 / 12,579], [24.33%],
  ),
  caption: [Untrained drafters on a 40-prompt validation sample.],
) <tab:b1-40>

The P2 count on this sample was reproduced exactly, token for token, by the training pipeline before training started, which confirms that the training job's replay agrees with the earlier one.

== Logit alignment on P2 (T1, T2) <sec:alignment>

Both adapters raised acceptance on the primary pair, and the LoRA adapter raised it much more than the output adapter (@tab:alignment-headline). LoRA lifted acceptance from below one in five to almost one in three. It also changed how many tokens the drafter proposed in total: since every replay produces the same answers, a drafter that is right more often proposes fewer tokens to get there.

#figure(
  table(columns: (auto, 1.3fr, 1fr, auto, auto), align: (left, left, right, right, right),
    table.header[*Drafter*][*Chosen setting*][*Accepted / proposed*][*Acceptance*][*Gain*],
    [B1, untrained], [—], [18,316 / 98,317], [18.63%], [—],
    [T1, output adapter], [rank 16, learning rate $10^(-2)$, 2 epochs], [20,836 / 89,120], [23.38%], [+4.75 pp],
    [T2, LoRA], [rank 16, learning rate $10^(-3)$, 4 epochs], [24,772 / 78,735], [31.46%], [+12.83 pp],
  ),
  caption: [Acceptance on pair P2, 320 validation prompts, simple text bridge. Gains are in percentage points (pp) over the untrained drafter.],
) <tab:alignment-headline>

The improvement holds in every task, but its size varies a great deal (@fig:acceptance-task, @tab:alignment-task). Maths and code improve most, instruction following and summarization much less.

#figure(
  image("../../figures/acceptance_per_task.svg", width: 100%),
  caption: [Acceptance per task on pair P2 for the untrained drafter (B1), the output adapter (T1) and LoRA (T2); 320 validation prompts.],
) <fig:acceptance-task>

#figure(
  table(columns: (auto, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right),
    table.header[*Drafter*][*Maths*][*Code*][*Instruction*][*Summarization*],
    [B1], [21.91% \ (7,000 / 31,948)], [16.25% \ (6,006 / 36,951)], [18.33% \ (4,118 / 22,468)], [17.15% \ (1,192 / 6,950)],
    [T1], [29.24% \ (8,173 / 27,956)], [22.22% \ (7,257 / 32,659)], [19.17% \ (4,162 / 21,707)], [18.30% \ (1,244 / 6,798)],
    [T2], [39.85% \ (9,634 / 24,173)], [32.91% \ (9,152 / 27,806)], [22.39% \ (4,580 / 20,452)], [22.30% \ (1,406 / 6,304)],
  ),
  caption: [Acceptance per task on pair P2, 80 validation prompts per task, with accepted / proposed tokens.],
) <tab:alignment-task>

=== Selection sweep <sec:sweep>

@fig:top1 shows how well each candidate setting learned to imitate the target, measured by top-1 agreement on the validation answers. The pattern is regular. For LoRA, every increase in learning rate helped, and so did training longer. For the output adapter, a higher learning rate also helped at both ranks, and the larger rank was better at every learning rate. LoRA's weakest setting still imitated the target better than the output adapter's best. The exact values are in @tab:sweep, and the underlying token counts in @app:tables.

#figure(
  image("../../figures/top1_sweep.svg", width: 100%),
  caption: [Top-1 agreement with the target's validation answers for each candidate setting on pair P2: (a) LoRA after 2 and 4 epochs; (b) the output adapter at ranks 8 and 16.],
) <fig:top1>

#figure(
  table(columns: (auto, auto, 1fr, 1fr, 1fr), align: (left, left, right, right, right),
    table.header[*Adapter*][*Variant*][*lr low*][*lr middle*][*lr high*],
    [T2, LoRA], [epoch 2], [0.5049], [0.5503], [0.5889],
    [], [epoch 4], [0.5186], [0.5638], [*0.5995*],
    [T1, output], [rank 8], [0.4462], [0.4695], [0.4874],
    [], [rank 16], [0.4546], [0.4802], [*0.4995*],
  ),
  caption: [Top-1 agreement on 62,972 validation answer tokens. Learning rates are $10^(-4)$, $3 times 10^(-4)$, $10^(-3)$ for LoRA and $10^(-3)$, $3 times 10^(-3)$, $10^(-2)$ for the output adapter. The chosen settings are in bold.],
) <tab:sweep>

=== Run time <sec:runtime>

The entire P2 experiment, three replays of the full validation set and the training runs behind all twelve candidate settings, took under half an hour on one free Kaggle T4 (@tab:runtime). None of it required the 7B target. These times are development costs, not decoding speeds.

#figure(
  table(columns: (1fr, auto), align: (left, right),
    table.header[*Phase*][*Minutes*],
    [Preparation and checks], [1.3],
    [Replay, untrained drafter], [6.6],
    [Training, LoRA (three learning rates, 4 epochs each)], [6.7],
    [Replay, LoRA drafter], [5.3],
    [Training, output adapter (six settings, 2 epochs each)], [3.7],
    [Replay, output-adapter drafter], [6.2],
    [*Total*], [*29.8*],
  ),
  caption: [Time per phase of the P2 experiment on one NVIDIA T4.],
) <tab:runtime>
