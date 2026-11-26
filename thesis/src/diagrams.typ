// Diagrams for the thesis, drawn with CeTZ. Greyscale-safe: black strokes, white and grey fills,
// dashed outlines for optional or held-back parts. Examples marked "illustrative" are not data.
#import "@preview/cetz:0.4.2": canvas, draw

#let grey = luma(215)
#let dark = luma(160)

#let nd(c, w, h, body, fill: white, dash: none, size: 9.5pt, weight: "regular") = {
  let (x, y) = c
  draw.rect((x - w / 2, y - h / 2), (x + w / 2, y + h / 2), fill: fill, radius: 0.08,
       stroke: (paint: black, thickness: 0.6pt, dash: dash))
  draw.content((x, y), box(width: (w - 0.2) * 1cm, {
    set par(justify: false, first-line-indent: 0pt, leading: 0.4em)
    set text(size: size, weight: weight, hyphenate: false)
    align(center, body)
  }))
}

#let dm(c, w, h, body, size: 9pt) = {
  let (x, y) = c
  draw.line((x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y), close: true,
       fill: white, stroke: 0.6pt)
  draw.content((x, y), box(width: (w * 0.62) * 1cm, {
    set par(justify: false, first-line-indent: 0pt, leading: 0.4em)
    set text(size: size, hyphenate: false)
    align(center, body)
  }))
}

#let ar(..pts, dash: none) = draw.line(..pts, stroke: (paint: black, thickness: 0.6pt, dash: dash),
  mark: (end: ">", fill: black, scale: 0.7))
#let ln(..pts, dash: none) = draw.line(..pts, stroke: (paint: black, thickness: 0.6pt, dash: dash))
#let lb(c, body, anchor: "center", size: 8.5pt, style: "italic", angle: 0deg) = draw.content(
  c, {
    set par(justify: false, first-line-indent: 0pt, leading: 0.4em)
    text(size: size, style: style, hyphenate: false, body)
  }, anchor: anchor, angle: angle)
#let tag(c, body, anchor: "west") = draw.content(c, text(size: 9.5pt, weight: "bold", body), anchor: anchor)

// Figure: target-only decoding versus speculative decoding (schematic).
#let fig-timeline = canvas({
  import draw: *
  tag((0, 2.2), [Target alone])
  for i in range(6) { nd((3.3 + i * 1.2, 2.2), 1.0, 0.6, [T], fill: dark) }
  lb((10.4, 2.2), [6 target passes → 6 tokens], anchor: "west", style: "normal", size: 9pt)
  tag((0, 0.7), [Speculative])
  let gains = ([+3 tokens], [+1 token], [+2 tokens])
  for r in range(3) {
    let x0 = 2.85 + r * 2.5
    for j in range(3) { nd((x0 + j * 0.42, 0.7), 0.36, 0.6, [d], size: 8pt) }
    nd((x0 + 1.75, 0.7), 1.0, 0.6, [T], fill: dark)
    lb((x0 + 1.0, -0.05), gains.at(r))
  }
  lb((10.4, 0.7), [3 target passes → 6 tokens], anchor: "west", style: "normal", size: 9pt)
  nd((3.0, -1.0), 0.6, 0.45, [T], fill: dark, size: 8pt)
  lb((3.45, -1.0), [one target forward pass (expensive)], anchor: "west", style: "normal")
  nd((8.6, -1.0), 0.36, 0.45, [d], size: 8pt)
  lb((8.95, -1.0), [one drafter step (cheap)], anchor: "west", style: "normal")
})

// Figure: one greedy speculative round, worked example (illustrative tokens).
#let fig-round-example = canvas({
  import draw: *
  let col(i) = 7.0 + i * 1.55
  tag((0, 2.4), [Committed text])
  for (i, t) in ([Tom], [has], [12]).enumerate() { nd((3.6 + i * 1.2, 2.4), 1.05, 0.6, t, fill: grey) }
  tag((0, 1.3), [Drafter guesses])
  for (i, t) in ([apples], [and], [3]).enumerate() { nd((col(i), 1.3), 1.35, 0.6, t) }
  tag((0, 0.2), [Target's choices])
  for (i, t) in ([apples], [and], [5]).enumerate() { nd((col(i), 0.2), 1.35, 0.6, t) }
  nd((col(3), 0.2), 1.35, 0.6, [ate], dash: "dashed")
  for (i, t) in ([agree], [agree], [differ]).enumerate() { lb((col(i), -0.55), t) }
  tag((0, -1.4), [Written this round])
  nd((col(0), -1.4), 1.35, 0.6, [apples], fill: grey)
  nd((col(1), -1.4), 1.35, 0.6, [and], fill: grey)
  nd((col(2), -1.4), 1.35, 0.6, [5], fill: dark)
  lb((col(2) + 0.85, -1.4), [target's correction], anchor: "west")
  lb((col(3), 0.75), [next position], size: 8pt)
  lb((col(3) - 0.55, -0.55), [used only if all agree], anchor: "west", size: 8pt)
})

// Figure: the same text in two vocabularies (illustrative split).
#let fig-tokenizers = canvas({
  import draw: *
  tag((0, 2.0), [Text])
  lb((5.4, 2.0), ["the stamps"], anchor: "west", style: "normal", size: 10pt)
  tag((0, 0.9), [Llama drafter (SentencePiece)])
  nd((6.0, 0.9), 1.3, 0.6, [▁the])
  nd((7.45, 0.9), 1.3, 0.6, [▁st])
  nd((8.9, 0.9), 1.3, 0.6, [amps])
  tag((0, -0.3), [Qwen target (byte-level BPE)])
  nd((6.0, -0.3), 1.3, 0.6, [Ġthe], fill: grey)
  nd((8.2, -0.3), 2.7, 0.6, [Ġstamps], fill: grey)
  lb((10.0, 0.3), [different pieces, \ different numbering], anchor: "west")
})

// Figure: the simple text bridge.
#let fig-text-bridge = canvas({
  import draw: *
  nd((1.2, 0), 2.2, 1.6, [Drafter's guess (draft tokens)])
  nd((4.1, 0), 2.6, 1.6, [Decode with the text before it; keep only the new characters])
  nd((7.25, 0), 2.7, 1.6, [Append to the last 16 committed target tokens and re-encode])
  dm((10.5, 0), 2.8, 1.9, [Committed tokens unchanged?])
  nd((13.75, 0), 2.3, 1.6, [Proposal: new target tokens (at most $K$)], fill: grey)
  nd((10.5, -2.4), 2.6, 0.9, [Encode the new characters alone])
  ar((2.3, 0), (2.8, 0))
  ar((5.4, 0), (5.9, 0))
  ar((8.6, 0), (9.1, 0))
  ar((11.9, 0), (12.6, 0))
  lb((12.25, 0.3), [yes])
  ar((10.5, -0.95), (10.5, -1.95))
  lb((10.8, -1.45), [no])
  ar((11.8, -2.4), (13.75, -2.4), (13.75, -0.8))
})

// Figure: the continuation cache (illustrative tokens).
#let fig-cache = canvas({
  import draw: *
  let x(i) = 3.1 + i * 1.3
  tag((0, 1.7), [Prompt])
  for (i, t) in ([Tom], [has], [12], [apples], [and], [buys], [5], [more]).enumerate() {
    nd((x(i), 1.7), 1.15, 0.55, t, fill: if i in (0, 1, 2) { grey } else { white })
  }
  tag((0, 0), [Answer so far])
  for (i, t) in ([So], [Tom], [has], [12]).enumerate() {
    nd((x(i), 0), 1.15, 0.55, t, fill: if i in (1, 2, 3) { grey } else { white })
  }
  nd((x(4), 0), 1.15, 0.55, [?], dash: "dashed")
  ar((x(2), 0.3), (x(1), 1.4))
  lb((x(1) + 0.2, 0.85), [(1) find the longest \ matching ending], anchor: "east")
  ar((x(3), 1.4), (x(4), 0.3))
  lb((x(4) - 0.1, 0.85), [(2) propose what came next: "apples"], anchor: "west")
})

// Figure: one SpecAdapt round as a flowchart.
#let fig-round-flow = canvas({
  import draw: *
  let c = 4
  let w = 4.4
  let h = 0.95
  nd((c, 11.0), w, h, [Text committed so far], fill: grey)
  dm((c, 9.55), 4.0, 1.6, [Cache knows this ending?])
  nd((8.9, 9.55), 3.3, h, [Propose the cached continuation])
  nd((c, 8.0), w, h, [Drafter guesses up to $K$ tokens])
  dm((c, 6.5), 4.0, 1.6, [Same guess seen in same context?])
  nd((8.9, 6.5), 3.3, h, [Reuse the stored translation])
  nd((c, 5.0), w, h, [Translate through text; store it])
  nd((c, 3.6), w, h, [Proposal of target tokens], fill: grey)
  nd((c, 2.3), w, h, [Target checks it in one pass])
  nd((c, 1.0), w, h, [Keep the agreeing prefix plus one target token])
  nd((c, -0.3), w, h, [Add the new tokens to the answer and the cache])
  nd((8.9, -0.3), 3.3, h, [Stop at end of answer or length limit], dash: "dashed")
  ar((c, 10.52), (c, 10.35))
  ar((c, 8.75), (c, 8.48)); lb((c + 0.3, 8.62), [no])
  ar((c + 2.0, 9.55), (7.25, 9.55)); lb((6.6, 9.8), [yes])
  ar((c, 7.52), (c, 7.3))
  ar((c, 5.7), (c, 5.48)); lb((c + 0.3, 5.6), [no])
  ar((c + 2.0, 6.5), (7.25, 6.5)); lb((6.6, 6.75), [yes])
  ar((c, 4.52), (c, 4.08))
  ln((8.9, 6.02), (8.9, 3.6))
  ar((10.55, 9.55), (11.2, 9.55), (11.2, 3.6), (6.2, 3.6))
  ar((c, 3.12), (c, 2.78))
  ar((c, 1.82), (c, 1.48))
  ar((c, 0.52), (c, 0.18))
  ar((6.2, -0.3), (7.25, -0.3)); lb((6.7, -0.05), [done])
  ar((1.8, -0.3), (0.6, -0.3), (0.6, 11.0), (1.8, 11.0))
  lb((0.3, 5.3), [next round], angle: 90deg)
})

// Figure: where the two adapters sit in the drafter.
#let drafter-stack(ox, title, head-fill, proj-fill, head-text) = {
  import draw: *
  let cx = ox + 3.2
  nd((cx, 0), 5.0, 0.7, [Token embeddings])
  for (b, y) in ((1, 1.75), (2, 3.85)) {
    rect((cx - 3.1, y - 0.85), (cx + 3.1, y + 0.85), radius: 0.08, stroke: 0.6pt)
    content((cx, y + 0.55), text(size: 8.5pt)[Block #b: attention and feed-forward])
    for (i, p) in ([q], [k], [v], [o], [gate], [up], [down]).enumerate() {
      nd((cx - 2.7 + i * 0.9, y - 0.2), 0.8, 0.55, p, fill: proj-fill, size: 8pt)
    }
  }
  nd((cx, 5.55), 5.0, 0.8, head-text, fill: head-fill)
  lb((cx, 6.55), [scores over the 32,000-piece vocabulary], style: "normal")
  ar((cx, 0.35), (cx, 0.9))
  ar((cx, 2.6), (cx, 3.0))
  ar((cx, 4.7), (cx, 5.15))
  ar((cx, 5.95), (cx, 6.35))
  content((cx, -0.85), text(size: 9.5pt, weight: "bold", title))
}
#let fig-adapters = canvas({
  import draw: *
  drafter-stack(0, [(a) T1: output adapter], dark, white, [Output layer $+ U D h + b$])
  drafter-stack(7.8, [(b) T2: LoRA], white, dark, [Output layer])
  nd((4.6, -1.7), 0.6, 0.4, [], fill: dark)
  lb((5.05, -1.7), [adapted part, folded into the drafter after training], anchor: "west", style: "normal")
})

// Figure: how a training example is built.
#let fig-training-data = canvas({
  import draw: *
  nd((1.5, 2.3), 2.6, 0.9, [Training prompt])
  nd((5.3, 2.3), 3.2, 0.9, [Target, greedy, recorded once])
  nd((9.6, 2.3), 3.0, 0.9, [The target's own answer], fill: grey)
  ar((2.8, 2.3), (3.7, 2.3))
  ar((6.9, 2.3), (8.1, 2.3))
  nd((2.6, 0), 4.6, 0.75, [Prompt and blank line (drafter's view)], fill: white)
  nd((8.3, 0), 6.4, 0.75, [The target's answer, re-encoded for the drafter], fill: grey)
  nd((12.35, 0), 1.3, 0.75, [EOS], dash: "dashed")
  ar((9.6, 1.85), (9.6, 0.4))
  ar((1.5, 1.85), (1.5, 0.4))
  lb((2.6, -0.75), [not scored])
  lb((8.3, -0.75), [scored: predict each next token])
  lb((12.35, -0.95), [only if the target \ ended by itself])
})

// Figure: oracle record-then-replay.
#let fig-oracle = canvas({
  import draw: *
  let h = 1.05
  tag((0, 3.4), [Record once per prompt (rented GPU)])
  nd((1.4, 2.3), 2.6, h, [Prompt])
  nd((5.2, 2.3), 3.2, h, [Target (7B) writes its greedy answer], fill: dark)
  nd((9.6, 2.3), 3.6, h, [Stored answer, with model and software versions], fill: grey)
  ar((2.7, 2.3), (3.6, 2.3))
  ar((6.8, 2.3), (7.8, 2.3))
  tag((0, 0.95), [Replay once per drafter (target not loaded)])
  nd((1.6, -0.2), 3.0, h, [Drafter and bridge propose])
  nd((5.9, -0.2), 3.4, h, [Check each guess against the stored token])
  nd((10.4, -0.2), 3.4, h, [Counts: proposed and accepted tokens, rounds], fill: grey)
  ar((3.1, -0.2), (4.2, -0.2))
  ar((7.6, -0.2), (8.7, -0.2))
  ar((9.6, 1.77), (9.6, 1.45), (6.9, 1.45), (6.9, 0.33))
  ar((5.9, -0.73), (5.9, -1.15), (1.6, -1.15), (1.6, -0.73))
  lb((3.75, -1.4), [the stored token is committed; next round])
})

// Figure: the experimental pipeline.
#let fig-pipeline = canvas({
  import draw: *
  nd((1.4, 1.8), 2.4, 1.1, [Four datasets])
  nd((4.85, 1.8), 3.1, 1.1, [Deduplicate, fix the test set, split])
  nd((8.45, 1.8), 2.9, 1.1, [Record target answers once], fill: dark)
  nd((12.1, 1.8), 3.0, 1.1, [Train adapters on the training set])
  nd((12.1, 0), 3.0, 1.1, [Choose settings by top-1 agreement])
  nd((8.45, 0), 2.9, 1.1, [Fold the adapter into the drafter])
  nd((4.85, 0), 3.1, 1.1, [Replay on the validation prompts])
  nd((1.4, 0), 2.4, 1.1, [Results (Chapters 4–5)], fill: grey)
  nd((4.85, 3.6), 3.1, 0.9, [Test set, held back], dash: "dashed")
  ar((2.6, 1.8), (3.3, 1.8))
  ar((6.4, 1.8), (7.0, 1.8))
  ar((9.9, 1.8), (10.6, 1.8))
  ar((12.1, 1.25), (12.1, 0.55))
  ar((10.6, 0), (9.9, 0))
  ar((7.0, 0), (6.4, 0))
  ar((3.3, 0), (2.6, 0))
  ar((4.85, 2.35), (4.85, 3.15), dash: "dashed")
})
