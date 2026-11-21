# SpecAdapt: Lightweight Draft-Side Adaptation for Speculative Decoding

## Core Idea

**SpecAdapt** aims to improve the **acceptance rate of speculative decoding (SpD)** when the draft model and target model are heterogeneous, especially when they use **different tokenizers or have mismatched output distributions**.

In standard speculative decoding, a small draft model proposes `K` tokens and a larger target model verifies them in parallel. The practical speedup therefore depends heavily on how many draft tokens are accepted. When the draft and target models have different vocabularies/tokenizers or substantially different probability distributions, the acceptance rate can collapse, potentially making speculative decoding slower than simply running the target model.

SpecAdapt introduces a **lightweight adapter on the draft side only**, leaving the target model unchanged. :chatgpt-content-reference{index="0"}

### Proposed Adapter

The adapter contains three components:

1. **M1 — Vocabulary Bridge**
   - Maps the draft vocabulary into the target vocabulary.
   - Handles direct vocabulary overlap.
   - Uses an **n-gram cache/trie** to merge sequences of draft sub-tokens into target tokens.
   - Example:  
     `st + amps → stamps`

2. **M2 — Logit Alignment**
   - Adjusts the draft model's probability distribution so that it more closely matches the target model.
   - Candidate methods include:
     - static logit bias,
     - low-rank logit correction head,
     - LoRA + knowledge distillation.

3. **M3 — Adaptive Draft Length K**
   - Dynamically determines how many tokens the draft model should propose.
   - A lightweight version uses an **entropy threshold**.
   - A stronger version uses logistic regression with inexpensive confidence features such as entropy, top-1 probability, top-1/top-2 margin, and cache hits.

The central design constraint is that the adapter must remain extremely cheap: its latency overhead should be **≤10% of one draft-model decoding step**. :chatgpt-content-reference{index="1"}

---

## Research Hypothesis

A significant fraction of the benefit of adaptive cross-vocabulary speculative decoding may be achievable **without expensive online gradient-based training**.

The project therefore studies three levels of adaptation:

- **Tier 0 — Training-free:** vocabulary bridge + n-gram cache + entropy-based adaptive K.
- **Tier 1 — Lightweight learned adaptation:** static logit bias + low-rank logit head + logistic-regression-based adaptive K.
- **Tier 2 — Gradient-based upper bound:** LoRA + knowledge distillation.

The main research question can therefore be summarized as:

> **How much of the acceptance-rate and speedup improvement of adaptive speculative decoding can be recovered using training-free or very lightweight draft-side adaptation?**

This Tier 0/Tier 1 emphasis is intended to distinguish SpecAdapt from prior work that relies primarily on online gradient-based adaptation. :chatgpt-content-reference{index="2"}

---

## Expected Contribution

The intended output is a reusable adapter framework that:

- works across heterogeneous `(draft, target)` model pairs;
- improves acceptance rate while keeping adaptation overhead very small;
- requires no modification of the target model;
- supports cross-tokenizer speculative decoding;
- dynamically adjusts draft length;
- provides reproducible benchmarks across multiple models and tasks;
- potentially evaluates Vietnamese as an additional cross-lingual/tokenization-stress setting.

Primary evaluation metrics are:

1. acceptance rate;
2. average accepted tokens per verification round;
3. actual inference speedup;
4. adapter overhead in ms/token;
5. output quality relative to direct target-model decoding. :chatgpt-content-reference{index="3"}

---

# Main Reference

## OmniDraft

**Ramakrishnan et al. — _OmniDraft: A Cross-vocabulary, Online Adaptive Drafter_**  
arXiv:2507.02659, Qualcomm AI Research.

This is the **closest and most important reference** for SpecAdapt.

OmniDraft addresses essentially the same general problem:

> improving speculative decoding when the draft and target models have heterogeneous vocabularies and distributions.

It is particularly relevant because it combines ideas closely related to all three SpecAdapt modules:

- cross-vocabulary handling;
- n-gram/cache-based mechanisms;
- draft–target distribution alignment;
- adaptive draft length.

SpecAdapt uses OmniDraft both as:

1. the primary experimental/reference baseline for expected acceptance-rate improvements; and
2. the main point from which the proposed research gap is defined.

The proposed distinction is that **OmniDraft's major gains are associated with online gradient-based adaptation**, whereas SpecAdapt investigates how much of those gains can be recovered through **training-free and lightweight parameter-efficient adaptation (Tier 0 and Tier 1)**. :chatgpt-content-reference{index="4"}

---

## One-Sentence Thesis Summary

> **SpecAdapt investigates whether a lightweight draft-side adapter combining cross-tokenizer vocabulary bridging, inexpensive logit alignment, and confidence-based adaptive draft length can recover most of the acceptance-rate benefits of online adaptive speculative decoding without requiring continuous gradient-based training.**