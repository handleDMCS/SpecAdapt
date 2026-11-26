#import "../uet-thesis.typ": *

= Discussion <ch:discussion>

This chapter interprets the results of the two previous chapters. Each point refers back to the table or figure it rests on. Where a cause has not been tested, none is offered.

== Answer to the research question <sec:answer>

The research question asked how much of the benefit of adaptive speculative decoding can be recovered by training-free or very lightweight changes on the drafter's side. For acceptance on the primary pair, the measurements give a clear, if partial, answer: _less than hoped_. The gradient-trained LoRA adapter does most of the work, the lightweight output adapter recovers roughly a third of what LoRA achieves, and the training-free bridge adds only a small improvement, and only to an untrained drafter (@tab:alignment-headline, @tab:m1-arms).

The comparison between the two adapters is fairer than it might first appear. They have almost exactly the same number of trainable parameters (@tab:adapters), so LoRA's advantage is not a matter of size. What differs is where the adaptation sits, spread through every layer or confined to the last one, and, to a lesser extent, how long each was trained. The experiments do not separate these two factors, so it cannot yet be said which one matters more.

The project's starting hypothesis was that the training-free and lightweight options would recover most of the gain. On this evidence, for this pair, they do not. Three things limit how far the conclusion reaches. The speed-up half of the question has not been measured for trained drafters. The results are on validation prompts that also served for selection, not on the held-out test set. And only one pair has been trained.

== What the acceptance results show <sec:reading>

*Where the gains come from.* Both adapters improve maths and code far more than instruction following and summarization, and they rank the four tasks in the same order (@tab:gains). Why the tasks differ so much was not tested, so no explanation is offered here.

#figure(
  table(columns: (auto, 1fr, 1fr, 1fr, 1fr), align: (left, right, right, right, right),
    table.header[*Adapter*][*Maths*][*Code*][*Instruction*][*Summarization*],
    [T1, output adapter], [+7.32], [+5.97], [+0.85], [+1.15],
    [T2, LoRA], [+17.94], [+16.66], [+4.07], [+5.15],
  ),
  caption: [Gain in acceptance over the untrained drafter, in percentage points, per task (pair P2, validation prompts; differences of the rates in @tab:alignment-task).],
) <tab:gains>

*What the sweep says.* Every step up in learning rate improved both adapters, and LoRA kept improving between its two checkpoints (@fig:top1). The chosen learning rate was the largest one tried for both adapters, so the search never found the point where a larger value starts to hurt. Better settings may well exist just beyond the grid. The LoRA result is therefore the best of the settings tried, not the best achievable.

*Agreement predicts acceptance.* The adapter that imitates the target better under teacher forcing also has the higher acceptance in replay (@tab:sweep, @tab:alignment-headline). This supports top-1 agreement as a cheap stand-in for acceptance when choosing settings, although with only two adapters compared it is a consistency check, not a demonstrated correlation.

== Accepted tokens, drafter calls and modeled cost <sec:cost-reading>

The bridge experiment separates two kinds of saving that acceptance alone hides: saving target calls and saving drafter calls (@tab:m1-arms, @fig:m1-arms).

One reading of the table, offered as a hypothesis, is that a trained drafter and the continuation cache compete for the same easy tokens. When the drafter is untrained, the cache knows things the drafter does not, such as numbers and names already seen in the prompt, so its proposals add accepted tokens and each target call produces more. Once the drafter has been trained to imitate the target, it already gets most of those tokens right on its own. The cache still answers about a fifth of the rounds, but now it mostly replaces a guess the drafter would also have got right. The saving moves from the expensive target to the cheap drafter. This hypothesis is consistent with the table but was not tested directly, for example by checking which tokens each source gets right.

Because a drafter step is cheap for this pair, saving a fifth of the drafter's steps lowers the modeled cost only a little, much less than training does. The balance would shift with a more expensive drafter. The 160M drafter of pair P4 costs almost four times as much per step relative to its target as the 68M drafter does for P2 (@tab:stage1), so the same reduction in drafter steps would be worth more there. That is a property of the cost formula, not a measurement: no P4 experiment with the bridge has been run.

The modeled cost also leaves something out: the time spent on cache lookups and on the full bridge's extra work is not counted. The only measurement of bridge time, from the early demonstration (@tab:october), found it to be a large fraction of a draft step, far above the project's budget. Until the bridge has been timed in the current setup, the modeled savings from the bridge should be treated as optimistic.

== Threats to validity <sec:threats>

The following limits apply to the current evidence.

+ *Selection and reporting share prompts.* The validation prompts were used both to choose settings and to report results, which may make the results somewhat optimistic. The held-out test set has not been used yet.
+ *Same task families.* Training and validation prompts come from the same four datasets, so how well the adapters transfer to new kinds of text is unknown.
+ *One trained pair.* Only P2 has been trained; whether the findings hold for other pairs is untested.
+ *Acceptance is not speed.* Acceptance and tokens per target call are proxies. No live timing exists for the trained drafters.
+ *Replay follows one recorded answer.* Replay follows the answer the target gave on one kind of GPU. A live target on other hardware may answer differently; even the untrained live runs disagreed with the target-only answer on a few prompts per pair (@tab:stage1).
+ *Greedy decoding only.* Sampling-based decoding and its verification rule were not studied.
+ *One random seed.* Each setting was trained once, so the run-to-run variation is unknown.
+ *No confidence intervals.* None has been computed yet for the main results.
+ *Split check incomplete.* An automated test that the prompt sets share no prompts has not been added; only the separation of training and validation prompts is checked at load time.
