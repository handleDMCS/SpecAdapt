"""Draft-only adapters and answer-token supervision for greedy oracle alignment."""

from dataclasses import dataclass
import math
from random import Random

import torch
from torch import nn
from torch.nn import functional as F

from specadapt.decoding import draft_prompt


IGNORE_LABEL = -100


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, *, bias=False):
        super().__init__()
        self.base = base.requires_grad_(False)
        factory = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.down = nn.Linear(base.in_features, rank, bias=False, **factory)
        self.up = nn.Linear(rank, base.out_features, bias=False, **factory)
        nn.init.zeros_(self.up.weight)
        self.bias = nn.Parameter(torch.zeros(base.out_features, **factory)) if bias else None

    def forward(self, hidden):
        logits = self.base(hidden) + self.up(self.down(hidden))
        return logits if self.bias is None else logits + self.bias

    @torch.no_grad()
    def merged_linear(self):
        base = self.base
        weight = base.weight + self.up.weight @ self.down.weight
        bias = base.bias
        if self.bias is not None:
            bias = self.bias if bias is None else bias + self.bias
        # A new weight preserves the original input embeddings if the LM head was tied.
        merged = nn.Linear(base.in_features, base.out_features, bias=bias is not None, device="meta")
        merged.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            merged.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        return merged.train(self.training)


class OutputLogitAdapter(LowRankLinear):
    """T1 adds a learned vocabulary bias and a low-rank delta to the LM head."""

    def __init__(self, base, rank):
        super().__init__(base, rank, bias=True)

    @classmethod
    def attach(cls, model, rank):
        model.requires_grad_(False)
        adapter = cls(model.get_output_embeddings(), rank)
        model.set_output_embeddings(adapter)
        return adapter


class LoRAAdapter:
    """T2 attaches LoRA to each Llama attention and MLP projection."""

    PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def __init__(self, model, rank=16):
        self.model = model
        self.paths = [name for name, module in model.named_modules()
                      if name.rsplit(".", 1)[-1] in self.PROJECTIONS and isinstance(module, nn.Linear)]
        if not self.paths or len(self.paths) != model.config.num_hidden_layers * len(self.PROJECTIONS):
            raise ValueError("Expected seven Llama projections per layer for LoRA.")
        model.requires_grad_(False)
        for name in self.paths:
            self._set_projection(name, LowRankLinear(model.get_submodule(name), rank))

    def _set_projection(self, name, module):
        parent, attribute = name.rsplit(".", 1)
        setattr(self.model.get_submodule(parent), attribute, module)

    def merge_and_unload(self):
        for name in self.paths:
            self._set_projection(name, self.model.get_submodule(name).merged_linear())
        return self.model


def adapter_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


@dataclass(frozen=True)
class TrainingExample:
    input_ids: list[int]
    labels: list[int]


def encode_example(tokenizer, row, max_length):
    prefix = draft_prompt(row["prompt"])
    encoded = tokenizer(prefix + row["text"], add_special_tokens=True, truncation=False,
                        return_offsets_mapping=True, verbose=False)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    eos = tokenizer.eos_token_id
    if ids and ids[-1] == eos and tuple(offsets[-1]) == (0, 0):
        ids.pop()
        offsets.pop()
    # A token crossing the text boundary is supervised if it contains answer text.
    labels = [token if end > len(prefix) and end > start else IGNORE_LABEL
              for token, (start, end) in zip(ids, offsets, strict=True)]
    if row["stop_reason"] == "eos":
        if eos is None:
            raise ValueError("The drafter tokenizer must define EOS.")
        ids.append(eos)
        labels.append(eos)
    elif row["stop_reason"] != "length":
        raise ValueError(f"Unknown stop reason for {row['prompt_id']}.")
    if len(ids) > max_length:
        raise ValueError(f"Draft context exceeded for {row['prompt_id']}: {len(ids)} > {max_length}.")
    if not any(label != IGNORE_LABEL for label in labels[1:]):
        raise ValueError(f"No answer tokens to supervise for {row['prompt_id']}.")
    return TrainingExample(ids, labels)


def pack_batches(examples, max_sequences, max_tokens):
    """Sort by length and count padded tokens against the batch budget."""
    batches, batch = [], []
    for example in sorted(examples, key=lambda row: len(row.input_ids)):
        length = len(example.input_ids)
        if length > max_tokens:
            raise ValueError(f"A sequence of {length} tokens exceeds the batch limit of {max_tokens}.")
        if batch and (len(batch) >= max_sequences or (len(batch) + 1) * length > max_tokens):
            batches.append(batch)
            batch = []
        batch.append(example)
    if batch:
        batches.append(batch)
    return batches


def collate(examples, pad_id, device):
    length = max(len(row.input_ids) for row in examples)
    padding = [length - len(row.input_ids) for row in examples]
    ids = [row.input_ids + [pad_id] * extra for row, extra in zip(examples, padding)]
    labels = [row.labels + [IGNORE_LABEL] * extra for row, extra in zip(examples, padding)]
    mask = [[1] * len(row.input_ids) + [0] * extra for row, extra in zip(examples, padding)]
    return tuple(torch.tensor(values, device=device) for values in (ids, labels, mask))


class M2Trainer:
    """Train the adapters in FP32 and score token-weighted teacher-forced top-1 agreement."""

    def __init__(self, model, pad_id, max_sequences, max_tokens, seed, on_oom, *,
                 weight_decay=0.0, warmup_fraction=0.05):
        self.model, self.pad_id, self.seed = model, pad_id, seed
        self.max_sequences, self.max_tokens = max_sequences, max_tokens
        self.on_oom = on_oom
        self.weight_decay, self.warmup_fraction = weight_decay, warmup_fraction
        self.reduced = False

    def logits_and_labels(self, batch):
        ids, labels, mask = collate(batch, self.pad_id, self.model.device)
        labels = labels[:, 1:]
        selected = labels.ne(IGNORE_LABEL)
        hidden = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        # Apply the expensive LM head only where the next token is supervised.
        return self.model.get_output_embeddings()(hidden[:, :-1][selected]), labels[selected]

    def _halve_limits(self):
        if self.reduced:
            raise RuntimeError("CUDA out of memory after halving the batch limits.")
        self.reduced = True
        self.max_sequences = max(1, self.max_sequences // 2)
        self.max_tokens = max(1, self.max_tokens // 2)
        self.on_oom(self.max_sequences, self.max_tokens)

    def _backward(self, batch):
        token_count = sum(sum(label != IGNORE_LABEL for label in row.labels[1:]) for row in batch)
        loss_total = 0.0
        for part in pack_batches(batch, self.max_sequences, self.max_tokens):
            logits, labels = self.logits_and_labels(part)
            loss = F.cross_entropy(logits, labels, reduction="sum") / token_count
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite training loss: {value}")
            loss.backward()
            loss_total += value
            del loss, logits, labels
        return loss_total

    def _train_step(self, batch, optimizer):
        optimizer.zero_grad(set_to_none=True)
        try:
            return self._backward(batch)
        except torch.cuda.OutOfMemoryError:
            # Clear the traceback before retrying so it cannot retain GPU tensors.
            if self.reduced:
                raise
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self._halve_limits()
        return self._backward(batch)

    def fit(self, examples, config, *, on_step, on_epoch):
        epochs, lr = config["epochs"], config["lr"]
        batches = pack_batches(examples, self.max_sequences, self.max_tokens)
        total_steps = len(batches) * epochs
        warmup = math.ceil(self.warmup_fraction * total_steps)
        optimizer = torch.optim.AdamW(adapter_parameters(self.model).values(), lr=lr,
                                      weight_decay=self.weight_decay, foreach=False)
        rng = Random(self.seed)
        step = 0
        on_step(0, total_steps)
        for epoch in range(1, epochs + 1):
            self.model.train()
            rng.shuffle(batches)
            for batch in batches:
                step += 1
                if step <= warmup:
                    factor = step / warmup
                else:
                    progress = (step - warmup) / (total_steps - warmup)
                    factor = (1 + math.cos(math.pi * progress)) / 2
                optimizer.param_groups[0]["lr"] = lr * factor
                loss = self._train_step(batch, optimizer)
                optimizer.step()
                on_step(step, total_steps, loss=loss)
            optimizer.zero_grad(set_to_none=True)
            on_epoch(epoch)

    def _score(self, batch):
        logits, labels = self.logits_and_labels(batch)
        return (logits.argmax(dim=-1) == labels).sum().item(), labels.numel()

    @torch.inference_mode()
    def evaluate(self, examples, on_step):
        self.model.eval()
        batches = pack_batches(examples, self.max_sequences, self.max_tokens)
        on_step(0, len(examples))
        matched = tokens = seen = 0
        while batches:
            batch = batches.pop(0)
            try:
                correct, answer_tokens = self._score(batch)
            except torch.cuda.OutOfMemoryError:
                if self.reduced:
                    raise
            else:
                matched += correct
                tokens += answer_tokens
                seen += len(batch)
                on_step(seen, len(examples), top1_agreement=matched / tokens)
                continue
            torch.cuda.empty_cache()
            self._halve_limits()
            batches = pack_batches([*batch, *(row for part in batches for row in part)],
                                   self.max_sequences, self.max_tokens)
        return dict(top1_agreement=matched / tokens, matched_answer_tokens=matched, answer_tokens=tokens)
