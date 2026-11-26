Large language models write one token per expensive forward pass. Speculative decoding speeds this up by letting a small draft model guess the next few tokens, which the large target model then checks all at once. The idea works only when the guesses are good, and it works poorly when the two models are strangers: when they split text into different tokens, or simply would not write the same thing.

This thesis asks how much can be gained by improving only the small model's side, leaving the large model untouched, and whether cheap or training-free changes are enough. Three components are built. A _vocabulary bridge_ translates the drafter's guesses into the target's tokens and, when the text so far repeats something already seen, proposes the continuation straight from a cache without asking the drafter. Two _alignment adapters_ teach the drafter to imitate the target's own answers: a small one on the drafter's output layer only, and a LoRA adapter spread through every layer. Both are folded into the drafter afterwards, so they cost nothing at run time. Finally, an _oracle replay_ method records each target's answers once and then scores any number of drafters against them without running the large model again.

On a small Llama drafter paired with Qwen2.5-7B-Instruct, the main findings, measured on validation prompts, are:

#align(center, block(width: 92%, {
  set par(first-line-indent: 0pt, justify: false, leading: 0.5em)
  set text(size: 11pt)
  table(columns: (1.1fr, 1.4fr), align: (left, left), stroke: 0.5pt, inset: 5pt,
    table.header[*Change to the draft side*][*Effect*],
    table.cell(colspan: 2)[_Share of guessed tokens the target accepts (320 prompts)_],
    [None (untrained drafter)], [18.6%],
    [Output-layer adapter], [23.4%],
    [LoRA adapter], [31.5%],
    table.cell(colspan: 2)[_Vocabulary bridge added (80-prompt subset)_],
    [With the untrained drafter], [slightly more tokens per target pass (1.51 → 1.56)],
    [With the LoRA drafter], [same tokens per target pass (1.82)],
    [With either drafter], [about a fifth fewer drafter steps],
  )
}))

The two adapters are almost the same size, yet the one spread through the whole drafter helps far more. The bridge helps an untrained drafter, but once the drafter has learned to imitate the target, the bridge mostly saves drafter work rather than target work. On this evidence, cheap draft-side tricks do not recover most of the benefit; training the drafter throughout matters most. Held-out test results and live speed measurements come next.
