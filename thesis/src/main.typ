// SpecAdapt thesis. Content source: thesis/skeleton.md. Layout: uet-thesis.typ.
#import "uet-thesis.typ": *

#show: thesis.with(
  // Name, title, supervisor: thesis/personal_info/info.md. Major: UET K66 class list (K66C-CLC, Cong nghe thong tin).
  // Degree name inferred from the major; confirm. Year: defense expected 12/2026.
  title: [Improving Speculative Decoding via Draft Adapters],
  author: [Trương Tuấn Dũng],
  supervisor: [Dr. Nguyễn Văn Vinh],
  major: [Information Technology],
  degree: [Bachelor of Information Technology],
  place-year: [Ha Noi - 2026],
  acknowledgement: include "chapters/0-acknowledgement.typ",
  logo: image("../assets/logo.png", width: 2.6cm),
  abstract: include "chapters/0-abstract.typ",
  keywords: ("speculative decoding", "cross-tokenizer drafting", "low-rank adaptation", "oracle replay"),
  abbreviations: (
    ("SpD", "Speculative decoding"),
    ("LoRA", "Low-rank adaptation"),
    ("LLM", "Large language model"),
    ("LM", "Language model"),
    ("KV", "Key-value (attention cache)"),
    ("EOS", "End-of-sequence token"),
    ("FP16", "16-bit floating point"),
    ("FP32", "32-bit floating point"),
    ("GPU", "Graphics processing unit"),
    ("pp", "Percentage points"),
    ("B1", "Untrained drafter with the baseline text bridge, fixed K = 3"),
    ("T1", "Tier 1 adapter: output logit adapter on the drafter's LM head"),
    ("T2", "Tier 2 adapter: LoRA on all drafter projections"),
    ("M1", "Module 1: vocabulary bridge with continuation caches"),
    ("M2", "Module 2: draft-side logit alignment"),
    ("M3", "Module 3: adaptive draft length"),
  ),
  list-of-algorithms: true,
)

#include "chapters/1-introduction.typ"
#include "chapters/2-literature.typ"
#include "chapters/3-methodology.typ"
#include "chapters/4-results-alignment.typ"
#include "chapters/5-results-bridge.typ"
#include "chapters/6-discussion.typ"
#include "chapters/7-conclusion.typ"

#bibliography("refs.bib", title: [References], style: "ieee")

#show: appendix
#include "chapters/appendix.typ"
