"""Adaptive draft length on one pair: gates, labels, a fitted stopping policy, sweeps and seven arms.

Everything pair-specific (pair, cost ratio, T2 weight hash, gate counts, tokenizer sharing) comes from
the job TOML, so the same script runs on every pair unchanged. The drafter is frozen; the policy is a small
logistic model on the drafter's own confidence, fitted on CPU inside the job.

Stages, results written as each one ends:
  a. Gates on all validation prompts at fixed K: B1 must reproduce its training-run counts exactly and
     T2 its acceptance within a tolerance.
  b. Labels: T2, mode baseline, fixed K = k_max, trace on, over the first fit prompts of each task and
     all validation prompts. B1 labels on a few fit prompts give B1's own entropy percentiles.
     Fit the logistic policy on the T2 fit labels; report AUC and label acceptance on fit and validation.
  c. Ceiling from the validation K = k_max trace: stop right after the first rejected token of every round.
  d. Threshold sweeps on the validation subset, thresholds at percentiles of each policy's score on the
     fit labels; each picked by modeled cost.
  e. Seven arms on all validation prompts (the two gate arms are the fixed-K drafter-only arms).
"""

import gzip
from importlib.metadata import version
import json
from pathlib import Path

from demo import kaggle_runtime as runtime
from specadapt.adaptive_k import (
    FEATURES, EntropyKPolicy, LogisticKPolicy, auc, percentile, save_policy,
)
from specadapt.training_inputs import MAX_NEW_TOKENS, TASKS
from training.kaggle.m1_combo import arm_summary, load_assets, replay_arm, select_subset, with_oracles
from training.kaggle.train_pair import Progress, deadline, submission_provenance, write_json

ARTIFACTS = Path("artifacts")
# name: (drafter, mode, policy key or None). The policy arms run at k = k_max.
ARMS = {
    "B1-k3": ("B1", "baseline", None),
    "T0-b1-m1-entropy": ("B1", "m1", "entropy-b1-m1"),
    "T2-k3": ("T2", "baseline", None),
    "T2-k3-m1": ("T2", "m1", None),
    "T2-entropy": ("T2", "baseline", "entropy-t2"),
    "T2-logistic": ("T2", "baseline", "logistic-t2"),
    "T2-logistic-m1": ("T2", "m1", "logistic-t2"),
}
# The gate arms first, then the arms the decision compares, so a late failure still leaves them.
ARM_ORDER = ("B1-k3", "T2-k3", "T2-k3-m1", "T2-logistic-m1", "T2-logistic", "T2-entropy", "T0-b1-m1-entropy")
# Sweep key: (drafter, mode, policy kind, labels its thresholds come from).
SWEEPS = {
    "entropy-b1-m1": ("B1", "m1", "entropy", "b1-fit"),
    "entropy-t2": ("T2", "baseline", "entropy", "fit"),
    "logistic-t2": ("T2", "baseline", "logistic", "fit"),
}


def check_job(job):
    if not 0 < job["preflight_minutes"] <= min(5, job["max_minutes"]):
        raise ValueError("Set a positive preflight budget of at most five minutes.")
    if job["seed"] != 42 or job["k_max"] <= job["k"] or job["cache_k"] <= 0:
        raise ValueError("Use seed 42, k_max above the fixed k and a positive cache_k.")
    if sorted(job["percentiles"]) != job["percentiles"] or not all(0 < q < 100 for q in job["percentiles"]):
        raise ValueError("percentiles must be increasing and strictly between 0 and 100.")
    if set(ARM_ORDER) != set(ARMS):
        raise ValueError("ARM_ORDER must list every arm once.")


def make_policy(kind, threshold, k_max, weights):
    if kind == "entropy":
        return EntropyKPolicy(threshold, k_max)
    return LogisticKPolicy(threshold, k_max, weights)


def options(job, mode, policy=None):
    """generate() options of an arm: fixed k for drafter-only arms, k_max (and cache_k) with a policy."""
    if policy is None:
        return dict(mode=mode, k=job["k"], same_tokenizer=job["same_tokenizer"])
    extra = dict(cache_k=job["cache_k"]) if mode == "m1" else {}
    return dict(mode=mode, k=policy.k_max, policy=policy, same_tokenizer=job["same_tokenizer"], **extra)


class Labels:
    """Compact per-token labels and per-round counts collected from K = k_max traces."""

    def __init__(self):
        self.prompts = dict(prompt_id=[], task=[], rounds=[], generated_tokens=[])
        self.rounds = dict(prompt=[], steps=[], drafted=[], proposed=[], accepted=[], labeled=[], ends_rejected=[])
        self.tokens = dict(prompt=[], round=[], position=[], label=[], **{name: [] for name in FEATURES})

    def add(self, row, data):
        index = len(self.prompts["prompt_id"])
        for key, value in (("prompt_id", row["prompt_id"]), ("task", row["task"]), ("rounds", data["rounds"]),
                           ("generated_tokens", data["generated_tokens"])):
            self.prompts[key].append(value)
        for number, record in enumerate(data["trace"]):
            if record["source"] != "draft":
                continue
            for key in ("steps", "drafted", "proposed", "accepted"):
                self.rounds[key].append(record[key])
            self.rounds["prompt"].append(index)
            for position, token in enumerate(record["tokens"], 1):
                self.tokens["prompt"].append(index)
                self.tokens["round"].append(number)
                self.tokens["position"].append(position)
                self.tokens["label"].append(token["label"])
                for name in FEATURES:
                    self.tokens[name].append(token[name])
            self.rounds["labeled"].append(len(record["tokens"]))
            self.rounds["ends_rejected"].append(int(bool(record["tokens"]) and record["tokens"][-1]["label"] == 0))

    def rows(self, task=None):
        """Feature tuples and labels, optionally for one task."""
        tasks = self.prompts["task"]
        keep = [i for i, prompt in enumerate(self.tokens["prompt"]) if task is None or tasks[prompt] == task]
        features = [tuple(self.tokens[name][i] for name in FEATURES) for i in keep]
        return features, [self.tokens["label"][i] for i in keep]

    def save(self, path):
        def compact(values):
            return [round(value, 6) if isinstance(value, float) else value for value in values]

        data = dict(features=list(FEATURES), prompts=self.prompts,
                    rounds={key: compact(values) for key, values in self.rounds.items()},
                    tokens={key: compact(values) for key, values in self.tokens.items()})
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))


def label_report(labels, weights):
    """AUC of the logistic policy (and of low entropy alone) and the label acceptance, overall and per task."""
    policy = LogisticKPolicy(0.5, 8, weights)

    def report(task):
        features, targets = labels.rows(task)
        named = [dict(zip(FEATURES, row)) for row in features]
        return dict(labels=len(targets), label_acceptance=sum(targets) / len(targets) if targets else 0.0,
                    logistic_auc=auc([policy.score(row) for row in named], targets),
                    entropy_auc=auc([-row["entropy"] for row in named], targets))

    return dict(**report(None), per_task={task: report(task) for task in TASKS})


def ceiling(labels, cost_ratio):
    """Modeled cost had drafting stopped right after the first rejected token of every round.

    The accepted prefix of every round is unchanged, so the rounds and committed tokens are the same;
    only the draft steps after the first rejection are saved.
    """
    def cost(task):
        prompts = [i for i, value in enumerate(labels.prompts["task"]) if task is None or value == task]
        chosen = set(prompts)
        rounds = sum(labels.prompts["rounds"][i] for i in prompts)
        generated = sum(labels.prompts["generated_tokens"][i] for i in prompts)
        steps = stopped = 0
        for i, prompt in enumerate(labels.rounds["prompt"]):
            if prompt in chosen:
                steps += labels.rounds["steps"][i]
                rejected = labels.rounds["ends_rejected"][i]
                stopped += labels.rounds["labeled"][i] if rejected else labels.rounds["steps"][i]
        return dict(fixed_k_max_cost_per_token=(rounds + cost_ratio * steps) / generated,
                    ceiling_cost_per_token=(rounds + cost_ratio * stopped) / generated,
                    draft_steps_saved=1 - stopped / steps if steps else 0.0)

    return dict(**cost(None), per_task={task: cost(task) for task in TASKS})


def cache_table(per_prompt):
    """Cache rounds by match length {rounds, proposed, accepted, acceptance}, and the draft rounds' acceptance."""
    table = {}
    for item in per_prompt:
        for length, counts in item.get("cache_by_match", {}).items():
            total = table.setdefault(length, dict(rounds=0, proposed=0, accepted=0))
            for key in total:
                total[key] += counts[key]
    for counts in table.values():
        counts["acceptance"] = counts["accepted"] / counts["proposed"] if counts["proposed"] else 0.0
    cache_proposed = sum(counts["proposed"] for counts in table.values())
    cache_accepted = sum(counts["accepted"] for counts in table.values())
    proposed = sum(item["proposed"] for item in per_prompt) - cache_proposed
    accepted = sum(item["accepted"] for item in per_prompt) - cache_accepted
    return dict(by_match_length=dict(sorted(table.items(), key=lambda pair: int(pair[0]))),
                draft_round_acceptance=accepted / proposed if proposed else 0.0)


class Job:
    def __init__(self, job, summary, progress, assets):
        self.job, self.summary, self.progress = job, summary, progress
        self.models, self.draft, self.target, record, fit, validation = assets
        self.validation = with_oracles(validation, record, self.target)
        self.subset = with_oracles(select_subset(validation, job["sweep_prompts_per_task"]), record, self.target)
        self.fit = with_oracles(select_subset(fit, job["fit_prompts_per_task"]), record, self.target)
        self.b1_fit = self.fit_subset(job["b1_fit_prompts_per_task"])
        self.per_prompt = {}
        summary.update(prompts=dict(validation=len(self.validation), sweep=len(self.subset), fit=len(self.fit),
                                    b1_fit=len(self.b1_fit)))
        ARTIFACTS.mkdir(exist_ok=True)

    def fit_subset(self, per_task):
        seen, chosen = {task: 0 for task in TASKS}, []
        for row, oracle in self.fit:
            if seen[row["task"]] < per_task:
                chosen.append((row, oracle))
                seen[row["task"]] += 1
        return chosen

    def save(self):
        write_json(Path("summary.json"), self.summary)

    def replay(self, drafter, prompts, name, on_prompt=None, **generate_options):
        return replay_arm(self.models[drafter], self.draft, self.target, prompts, name, self.progress,
                          on_prompt=on_prompt, **generate_options)

    def arm(self, name, policy=None):
        drafter, mode, _ = ARMS[name]
        per_prompt = self.replay(drafter, self.validation, name, **options(self.job, mode, policy))
        result = arm_summary(per_prompt, self.job["cost_ratio"])
        if policy is not None:
            result["policy"] = dict(kind=policy.name, threshold=policy.threshold, k_max=policy.k_max)
        if mode == "m1":
            result["cache"] = cache_table(per_prompt)
        self.per_prompt[name] = per_prompt
        self.summary["arms"][name] = result
        write_json(ARTIFACTS / f"arm-{name}.json", per_prompt)
        self.progress.once("result", arm=name, acceptance=result["acceptance"], mean_k=result["mean_k"],
                           tokens_per_round=result["tokens_per_round"], cost=result["modeled_cost_per_token"],
                           cache_share=result["cache_round_share"])
        self.save()
        return result

    def gates(self):
        gate = self.job["gate"]
        b1 = self.arm("B1-k3")
        if (b1["accepted"], b1["proposed"]) != (gate["b1_accepted"], gate["b1_proposed"]):
            runtime.abort(f"B1 gate: {b1['accepted']} of {b1['proposed']} accepted, expected "
                          f"{gate['b1_accepted']} of {gate['b1_proposed']}.")
        t2 = self.arm("T2-k3")
        gap_pp = 100 * abs(t2["acceptance"] - gate["t2_acceptance"])
        self.summary["gates"] = dict(b1_exact=True, t2_gap_pp=gap_pp)
        if not gap_pp <= gate["t2_tolerance_pp"]:
            runtime.abort(f"T2 gate: acceptance {t2['acceptance']:.4%} is {gap_pp:.3f} points from "
                          f"{gate['t2_acceptance']:.4%}.")
        self.save()

    def collect(self, drafter, prompts, name):
        labels = Labels()
        per_prompt = self.replay(drafter, prompts, name, on_prompt=labels.add, mode="baseline",
                                 k=self.job["k_max"], trace=True, same_tokenizer=self.job["same_tokenizer"])
        labels.save(ARTIFACTS / f"labels-{name}.json.gz")
        return labels, arm_summary(per_prompt, self.job["cost_ratio"])

    def label_and_fit(self):
        k_max = self.job["k_max"]
        self.labels = {}
        reference = self.summary["labels"] = {}
        for key, drafter, prompts in (("fit", "T2", self.fit), ("val", "T2", self.validation),
                                      ("b1-fit", "B1", self.b1_fit)):
            self.labels[key], reference[f"{drafter}-k{k_max}-{key}"] = self.collect(drafter, prompts, key)
            self.save()
        features, targets = self.labels["fit"].rows()
        self.weights = LogisticKPolicy.fit(features, targets, l2=self.job["l2"])
        write_json(ARTIFACTS / "logistic-weights.json", self.weights)
        fit_report = label_report(self.labels["fit"], self.weights)
        val_report = label_report(self.labels["val"], self.weights)
        self.summary["policy_fit"] = dict(
            weights=self.weights, fit=fit_report, val=val_report,
            gap=dict(label_acceptance_pp=100 * (fit_report["label_acceptance"] - val_report["label_acceptance"]),
                     logistic_auc=None if None in (fit_report["logistic_auc"], val_report["logistic_auc"])
                     else fit_report["logistic_auc"] - val_report["logistic_auc"]))
        self.progress.once("policy-fit", fit_auc=fit_report["logistic_auc"], val_auc=val_report["logistic_auc"],
                           fit_label_acceptance=fit_report["label_acceptance"],
                           val_label_acceptance=val_report["label_acceptance"])
        self.summary["ceiling"] = ceiling(self.labels["val"], self.job["cost_ratio"])
        self.progress.once("ceiling", cost=self.summary["ceiling"]["ceiling_cost_per_token"],
                           fixed_k_max_cost=self.summary["ceiling"]["fixed_k_max_cost_per_token"])
        self.save()

    def thresholds(self, kind, labels_key):
        features, _ = self.labels[labels_key].rows()
        if kind == "entropy":
            scores = [row[0] for row in features]
        else:
            policy = LogisticKPolicy(0.5, self.job["k_max"], self.weights)
            scores = [policy.score(dict(zip(FEATURES, row))) for row in features]
        return [percentile(scores, q) for q in self.job["percentiles"]]

    def sweep(self, key):
        drafter, mode, kind, labels_key = SWEEPS[key]
        points = []
        self.summary["sweeps"][key] = dict(drafter=drafter, mode=mode, kind=kind, thresholds_from=labels_key,
                                           points=points)
        for q, threshold in zip(self.job["percentiles"], self.thresholds(kind, labels_key)):
            policy = make_policy(kind, threshold, self.job["k_max"], self.weights)
            name = f"sweep-{key}-p{q}"
            per_prompt = self.replay(drafter, self.subset, name, **options(self.job, mode, policy))
            result = arm_summary(per_prompt, self.job["cost_ratio"])
            points.append(dict(percentile=q, threshold=threshold, acceptance=result["acceptance"],
                               mean_k=result["mean_k"], tokens_per_round=result["tokens_per_round"],
                               modeled_cost_per_token=result["modeled_cost_per_token"]))
            self.progress.once("sweep", policy=key, percentile=q, threshold=threshold,
                               cost=result["modeled_cost_per_token"], mean_k=result["mean_k"])
            self.save()
        best = min(points, key=lambda point: point["modeled_cost_per_token"])
        self.summary["sweeps"][key]["chosen"] = best
        policy = make_policy(kind, best["threshold"], self.job["k_max"], self.weights)
        save_policy(policy, ARTIFACTS / f"policy-{key}.json")
        self.save()
        return policy

    def finish(self):
        arms = self.summary["arms"]
        base = arms["B1-k3"]
        for result in arms.values():
            result["cost_vs_b1_k3"] = result["modeled_cost_per_token"] / base["modeled_cost_per_token"]
            for task, data in result["per_task"].items():
                data["cost_vs_b1_k3"] = (data["modeled_cost_per_token"]
                                         / base["per_task"][task]["modeled_cost_per_token"])
        self.summary["arms"] = {name: arms[name] for name in ARMS}
        k3 = arms["T2-k3"]["modeled_cost_per_token"]
        self.summary["decision"] = dict(
            logistic_m1_cost_cut_vs_k3_m1=1 - arms["T2-logistic-m1"]["modeled_cost_per_token"]
            / arms["T2-k3-m1"]["modeled_cost_per_token"],
            logistic_cost_cut_vs_k3=1 - arms["T2-logistic"]["modeled_cost_per_token"] / k3,
            entropy_cost_cut_vs_k3=1 - arms["T2-entropy"]["modeled_cost_per_token"] / k3,
            ceiling_cost_cut_vs_k3=1 - self.summary["ceiling"]["ceiling_cost_per_token"] / k3,
            logistic_beats_entropy=arms["T2-logistic"]["modeled_cost_per_token"]
            < arms["T2-entropy"]["modeled_cost_per_token"])
        self.save()


def run(job, summary, progress, assets):
    check_job(job)
    summary.setdefault("arms", {})
    summary.setdefault("sweeps", {})
    work = Job(job, summary, progress, assets)
    work.save()
    work.gates()
    work.label_and_fit()
    policies = {key: work.sweep(key) for key in SWEEPS}
    for name in ARM_ORDER:
        if name not in summary["arms"]:
            work.arm(name, policies.get(ARMS[name][2]))
    work.finish()


def record_config(job, summary):
    import torch

    summary["gpu_config"] = dict(
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, device=0,
        attention="sdpa", replay_precision="float16", feature_precision="float32", seed=job["seed"],
        batch=1, greedy=True, max_new_tokens=MAX_NEW_TOKENS,
        packages={name: version(name) for name in ("transformers", "accelerate", "sentencepiece", "safetensors",
                                                   "tokenizers", "huggingface-hub", "numpy")})


def main():
    job = runtime.job_config()
    check_job(job)
    summary = dict(verdict="running", pair=job["pair"], k=job["k"], k_max=job["k_max"],
                   cost_ratio=job["cost_ratio"], max_new_tokens=MAX_NEW_TOKENS, seed_cache=False,
                   provenance=submission_provenance(), job=job, arms={}, sweeps={})
    try:
        with deadline(job["max_minutes"] * 60, "Job exceeded max_minutes."), Progress() as progress:
            with deadline(job["preflight_minutes"] * 60, "Preflight exceeded its budget."):
                assets = load_assets(job, summary, progress)
                record_config(job, summary)
            run(job, summary, progress, assets)
    except Exception as error:
        summary.update(verdict="fail", error=f"{type(error).__name__}: {error}")
        write_json(Path("summary.json"), summary)
        raise
    summary["verdict"] = "pass"
    write_json(Path("summary.json"), summary)


if __name__ == "__main__":
    main()
