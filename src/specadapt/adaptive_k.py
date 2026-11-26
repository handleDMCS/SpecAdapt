"""Adaptive draft length: after each drafted token, decide whether to draft another.

A policy only shortens or lengthens the proposal. The target (or the oracle) still verifies every
proposed token, so replay stays lossless whatever the policy decides.

Features come from the drafter's next-token distribution that produced the token just drafted,
with the logits cast to FP32 (FP16 softmax tails distort entropy):
entropy, top1_prob and top1_top2_margin = top1_prob - top2_prob.
"""

import json
import math

FEATURES = ("entropy", "top1_prob", "top1_top2_margin")
K_MAX = 8


def token_features(logits):
    """Features of one next-token logit vector (any dtype, any device), as plain floats."""
    import torch

    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum()
    top = probs.topk(2).values
    values = torch.stack([entropy, top[0], top[0] - top[1]]).tolist()
    return dict(zip(FEATURES, values))


class _Policy:
    def __init__(self, threshold, k_max=K_MAX):
        if type(k_max) is not int or k_max <= 0:
            raise ValueError("k_max must be a positive integer.")
        if not math.isfinite(threshold):
            raise ValueError("The threshold must be finite.")
        self.threshold, self.k_max = float(threshold), k_max

    def score(self, features):
        raise NotImplementedError

    def should_continue(self, features, drafted):
        """After `drafted` tokens, the last with these features: draft one more?

        At least one token is always drafted (the caller asks only after one) and never more than k_max.
        The token that triggers the stop is still proposed: its draft step is already paid.
        """
        return drafted < self.k_max and self.keeps(self.score(features))


class EntropyKPolicy(_Policy):
    """Keep drafting while the entropy of the token just drafted is at most the threshold."""

    name = "entropy"

    def score(self, features):
        return features["entropy"]

    def keeps(self, score):
        return score <= self.threshold

    def describe(self):
        return dict(kind=self.name, threshold=self.threshold, k_max=self.k_max)


class LogisticKPolicy(_Policy):
    """Keep drafting while the modeled P(accept) of the token just drafted is at least the threshold."""

    name = "logistic"

    def __init__(self, threshold, k_max=K_MAX, weights=None):
        super().__init__(threshold, k_max)
        if weights is None or tuple(weights["features"]) != FEATURES:
            raise ValueError(f"Logistic weights must cover the features {FEATURES}.")
        self.weights = weights

    def score(self, features):
        w = self.weights
        z = w["intercept"] + sum(coef * (features[name] - mean) / scale for name, coef, mean, scale
                                 in zip(FEATURES, w["coef"], w["mean"], w["scale"]))
        return _sigmoid(z)

    def keeps(self, score):
        return score >= self.threshold

    def describe(self):
        return dict(kind=self.name, threshold=self.threshold, k_max=self.k_max, weights=self.weights)

    @staticmethod
    def fit(rows, labels, l2=1.0, iterations=50):
        """L2 logistic regression on standardized features, by Newton steps, on CPU.

        rows: sequence of feature dicts (or of tuples in FEATURES order); labels: 0/1.
        The intercept is not penalized. Returns the weights dict that LogisticKPolicy takes.
        """
        import numpy as np

        x = np.asarray([[row[name] for name in FEATURES] if isinstance(row, dict) else row for row in rows],
                       dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or len(y) != len(x) or len(x) == 0:
            raise ValueError("fit() needs one feature row per label.")
        if not (0 < y.mean() < 1):
            raise ValueError("fit() needs both accepted and rejected labels.")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        design = np.hstack([np.ones((len(x), 1)), (x - mean) / scale])
        penalty = np.diag([0.0] + [l2] * len(FEATURES))
        w = np.zeros(design.shape[1])
        for _ in range(iterations):
            p = 1 / (1 + np.exp(-design @ w))
            gradient = design.T @ (p - y) + penalty @ w
            hessian = (design * (p * (1 - p))[:, None]).T @ design + penalty
            step = np.linalg.solve(hessian, gradient)
            w -= step
            if np.abs(step).max() < 1e-10:
                break
        if not np.isfinite(w).all():
            raise ValueError("The logistic fit diverged.")
        return dict(features=list(FEATURES), mean=mean.tolist(), scale=scale.tolist(),
                    intercept=float(w[0]), coef=w[1:].tolist(), l2=l2, samples=len(y),
                    label_rate=float(y.mean()))


def _sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def save_policy(policy, path):
    path.write_text(json.dumps(policy.describe(), indent=2) + "\n", encoding="utf-8")


def load_policy(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["kind"] == "entropy":
        return EntropyKPolicy(data["threshold"], data["k_max"])
    if data["kind"] == "logistic":
        return LogisticKPolicy(data["threshold"], data["k_max"], data["weights"])
    raise ValueError(f"Unknown policy kind {data['kind']}.")


def auc(scores, labels):
    """Area under the ROC curve (Mann-Whitney, ties count half). None if one class is missing."""
    pairs = sorted(zip(scores, labels))
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum, index = 0.0, 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label)
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile() of no values.")
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def text_labels(draft_lengths, accepted_length, verified_length=None):
    """Label drafted tokens from text lengths; works across tokenizers.

    draft_lengths[i] is the length of the decoded text of drafted tokens 1..i+1; accepted_length is the
    length of the decoded text of the target tokens accepted in that round. Token i is accepted when its
    cumulative text is no longer than the accepted text. Labels run up to and including the first
    rejected token. When the round's whole proposal was accepted but cut short (the generation limit),
    drafted text beyond verified_length was never verified and gets no label.
    """
    labels = []
    for length in draft_lengths:
        if length <= accepted_length:
            labels.append(1)
            continue
        if verified_length is not None and accepted_length >= verified_length:
            break
        labels.append(0)
        break
    return labels
