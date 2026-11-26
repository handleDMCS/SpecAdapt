// Starter for thesis/src/main.typ. Replace every [TO FILL ...] from thesis/skeleton.md.
#import "uet-thesis.typ": *

#show: thesis.with(
  // Leave a field out to keep its [TO FILL] placeholder; never guess a name or a title.
  keywords: (),
  abbreviations: (
    ("SpD", "Speculative decoding"),
    ("LoRA", "Low-rank adaptation"),
  ),
  list-of-algorithms: true,
)

= Introduction
== Motivation
Body text goes here. Mention @fig:example and @tab:example before they appear.

#figure(
  rect(width: 6cm, height: 3cm, stroke: 0.5pt)[placeholder],
  caption: [Caption below the figure.],
) <fig:example>

#figure(
  table(columns: 3, table.header[*Tier*][*Accepted / proposed*][*Acceptance*], [B1], [], [], [T2], [], []),
  caption: [Caption above the table.],
) <tab:example>

#algorithm(caption: [One speculative round.])[
  + Propose up to $K$ tokens.
  + Verify and commit the accepted prefix.
] <alg:example>

$ "acceptance" = "accepted" / "proposed" $ <eq:acceptance>

== Contributions and thesis overview
#pending[write from skeleton §1.3 and §1.4]

= Literature review

= Methodology
== Overview and tiers
== The SpecAdapt method
== Experimental design

= Results: baselines and logit alignment

= Results: vocabulary bridge and decoding cost

= Discussion

= Conclusion and future work
== Conclusion
== Future work

// #bibliography("refs.bib", title: [References], style: "ieee")

#show: appendix
= Reproducibility
