"""Is a Kaggle training run going as expected right now? One flag per check, then a verdict.

    uv run python -m training.training_red_flag <timestamp>
    uv run python -m training.training_red_flag <timestamp> --visual
    uv run python -m training.training_red_flag <timestamp> --log saved.log

Reads the run's heartbeat lines: from the collected result (results/submission-<ts>.json) once the
run has been collected, otherwise from a snapshot of the live Kaggle log. Each check prints
green-flag (as expected), neutral (too early to tell, or mildly off) or red-flag (act now).
Read-only: it never changes or cancels the run. Run it as often as you like.
--visual also draws the run (loss, top-1 agreement, replay, results, flags) to
build/red-flag/<timestamp>.png and opens it.

Exit code: 0 green-flag, 3 neutral, 2 red-flag, 1 the check itself could not run.

The expectations live in EXPECT below. Later iterations update them there as runs measure them.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("math", "code", "instruction", "summarization")
FINAL = {"complete", "error", "cancelled", "canceled", "platform_error"}

EXPECT = {
    # Untrained (B1) replay acceptance at K = 3. P2: smoke 2026-11-24 (40 prompts, exact gate);
    # P1 and P4: stage 1 Kaggle replay on 40 validation prompts.
    "b1_acceptance": {"P1": 0.282, "P2": 0.187, "P4": 0.243},
    "b1_green": 0.03,
    "b1_red": 0.06,
    # Teacher-forced top-1 agreement of a barely trained drafter (P2 smoke, 40 prompts, 10 steps).
    "untrained_top1": {"P2": 0.433},
    "top1_green_pp": 1.0,
    "top1_red_pp": -3.0,
    # Training loss, first to last reading of one fit. Readings are single batches, so they are noisy.
    "loss_drop_green": 0.02,
    "loss_rise_red": 0.15,
    # Replay gain over B1 in percentage points. The plan's bar for "the recipe works" is 2.
    "gain_green_pp": 2.0,
    # Seconds since the last heartbeat. The trainer's keepalive prints at least every 30 s.
    "silence_neutral_s": 180,
    "silence_red_s": 600,
    # Share of the job's max_minutes used while it is still running.
    "budget_neutral": 0.75,
    "budget_red": 0.9,
}

GREEN, NEUTRAL, RED = "green-flag", "neutral", "red-flag"
RANK = {GREEN: 0, NEUTRAL: 1, RED: 2}
VERDICT = {
    GREEN: "On track. Nothing to do.",
    NEUTRAL: "Nothing is wrong yet, but something is mildly off. Check again in 10 minutes.",
    RED: "Act now. See 'If something goes wrong' in training_notes/babysitter.md.",
}
EXIT = {GREEN: 0, NEUTRAL: 3, RED: 2}
HEARTBEAT = re.compile(r"\[heartbeat (\d\d):(\d\d):(\d\d)Z\] (.*)")


@dataclass
class Beat:
    clock: int  # seconds since midnight UTC of the first line's day, unwrapped across midnight
    phase: str
    step: int | None
    total: int | None
    fields: dict = field(default_factory=dict)

    @property
    def done(self):
        return bool(self.total) and self.step == self.total


@dataclass
class Check:
    flag: str
    name: str
    detail: str
    pending: bool = False  # no signal yet; shown, but left out of the verdict


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_beats(lines):
    beats, offset, previous = [], 0, None
    for line in lines:
        match = HEARTBEAT.search(line)
        if not match:
            continue
        clock = int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) + offset
        if previous is not None and clock < previous - 43200:
            offset += 86400
            clock += 86400
        previous = clock
        fields, key = {}, None
        for token in match[4].split(" "):
            name, separator, value = token.partition("=")
            if separator and re.fullmatch(r"[A-Za-z_][\w.-]*", name):
                key, fields[name] = name, value
            elif key:
                fields[key] += " " + token  # values with spaces, such as an abort reason
        step = total = None
        if "step" in fields:
            done, _, out_of = fields.pop("step").partition("/")
            step, total = int(done), int(out_of) if out_of else None
        beats.append(Beat(clock, fields.pop("phase", "?"), step, total, fields))
    return beats


def fit_label(job, tier, index):
    try:
        if tier == "T2":
            return f"rank {job['t2']['rank']}, lr {job['t2']['learning_rates'][index - 1]:g}"
        t1 = job["t1"]
        rank, lr = [(r, lr) for r in t1["ranks"] for lr in t1["learning_rates"]][index - 1]
        return f"rank {rank}, lr {lr:g}"
    except (KeyError, IndexError, TypeError):
        return ""


def worst(*flags):
    return max(flags, key=RANK.__getitem__)


def status_check(status):
    if status == "queued":
        return Check(NEUTRAL, "kaggle", "queued: waiting for a GPU. Queue time does not count against "
                                        "max_minutes.", pending=True)
    if status in {"running", "complete"}:
        return Check(GREEN, "kaggle", status)
    if status in FINAL:
        return Check(RED, "kaggle", f"{status}: collect with --resume, then prompt 3.b.")
    return Check(NEUTRAL, "kaggle", status, pending=True)


def replay_check(tier, beats, pair, baseline=()):
    last = beats[-1]
    acceptance = number(last.fields.get("acceptance"))
    name = f"{tier} replay"
    if not last.done:
        progress = f"{last.step}/{last.total} prompts" if last.step is not None else "loading the drafter"
        detail = progress + (f", running acceptance {acceptance:.4f}" if acceptance is not None else "")
        if tier == "B1" and last.total:
            quarter = last.total // 4
            detail += (f". Tasks run in order math, code, instruction, summarization, so the running value "
                       f"drops during code (prompts {quarter + 1}-{2 * quarter}).")
        # B1 at about the same prompt is the fair comparison: the prompts run in task order.
        earlier = [beat for beat in baseline if beat.step and last.step and beat.step <= last.step
                   and number(beat.fields.get("acceptance")) is not None]
        if tier != "B1" and acceptance is not None and earlier:
            reference = earlier[-1]
            delta = 100 * (acceptance - number(reference.fields["acceptance"]))
            detail += (f", B1 was {number(reference.fields['acceptance']):.4f} at prompt {reference.step}"
                       f" ({delta:+.1f} pp so far)")
            return Check(GREEN if delta >= EXPECT["gain_green_pp"] else NEUTRAL, name, detail)
        return Check(NEUTRAL, name, detail, pending=True)
    if tier != "B1":
        return Check(GREEN, name, f"finished {last.total} prompts, acceptance {acceptance:.4f}")
    expected = EXPECT["b1_acceptance"].get(pair)
    detail = f"acceptance {acceptance:.4f} on {last.total} prompts"
    if expected is None:
        return Check(GREEN, name, detail + " (no stored expectation for this pair)")
    off = abs(acceptance - expected)
    detail += f", expected {expected:.3f} +/- {EXPECT['b1_green']:.2f}"
    if off <= EXPECT["b1_green"]:
        return Check(GREEN, name, detail)
    if off <= EXPECT["b1_red"]:
        return Check(NEUTRAL, name, detail + ". A little off; the untrained drafter should not move.")
    return Check(RED, name, detail + ". Far off: the data or the decoding changed. Prompt 3.b.")


def fit_check(tier, index, beats, job):
    last = beats[-1]
    name = f"{tier} fit {index}"
    label = fit_label(job, tier, index)
    prefix = f"{label}: " if label else ""
    progress = f"step {last.step}/{last.total}" if last.step is not None else "loading the drafter"
    losses = [number(beat.fields["loss"]) for beat in beats if "loss" in beat.fields and beat.step]
    losses = [loss for loss in losses if loss is not None]
    if len(losses) < 2:
        state = "finished" if last.done else progress
        return Check(NEUTRAL, name, f"{prefix}{state}, {len(losses)} loss reading(s). Too few to judge; "
                                    "the score line decides.", pending=True)
    first, end = losses[0], losses[-1]
    change = (end - first) / first
    detail = f"{prefix}{'finished' if last.done else progress}, loss {first:.3f} -> {end:.3f} ({change:+.1%})"
    if change <= -EXPECT["loss_drop_green"]:
        return Check(GREEN, name, detail)
    if change > EXPECT["loss_rise_red"]:
        return Check(RED, name, detail + ". Rising: learning rate too high or diverging.")
    return Check(NEUTRAL, name, detail + ". Flat so far; single-batch readings are noisy.")


def score_check(tier, index, epoch, beats, pair, smoke, history):
    last = beats[-1]
    name = f"{tier} score {index} ep {epoch}"
    if not last.done:
        progress = f"{last.step}/{last.total}" if last.step is not None else "starting"
        return Check(NEUTRAL, name, f"scoring validation answers, {progress}", pending=True)
    top1 = number(last.fields.get("top1_agreement"))
    previous = history.get((tier, index))
    history[(tier, index)] = top1
    parts = [f"top-1 agreement {top1:.4f}"]
    if smoke:
        return Check(GREEN, name, parts[0] + " (smoke: recorded only)")
    base = EXPECT["untrained_top1"].get(pair)
    flag, pending = GREEN, False
    if base is not None:
        delta = 100 * (top1 - base)
        parts.append(f"{delta:+.1f} pt vs untrained {base:.3f}")
        if delta < EXPECT["top1_red_pp"]:
            flag = RED
            parts.append("worse than untrained: training is hurting")
        elif delta < EXPECT["top1_green_pp"]:
            flag = NEUTRAL
            parts.append("no better than untrained yet")
    elif previous is None:
        flag, pending = NEUTRAL, True
        parts.append("no untrained value stored for this pair; compare with the next checkpoint")
    if previous is not None:
        if top1 < previous - 0.01:
            flag = worst(flag, NEUTRAL)
            parts.append(f"down from {previous:.4f} at the earlier checkpoint: past its best, "
                         "selection keeps the better one")
        else:
            parts.append(f"from {previous:.4f} at the earlier checkpoint")
    return Check(flag, name, ", ".join(parts), pending=pending)


def result_check(beat, smoke):
    if "tier" not in beat.fields or "gain_pp" not in beat.fields:
        # Other jobs (replay-only arms) report their own fields; show them as recorded.
        label = beat.fields.get("arm", "result")
        values = ", ".join(f"{key} {value}" for key, value in beat.fields.items() if key != "arm")
        return Check(GREEN, f"{label} result", values or "recorded")
    tier = beat.fields.get("tier", "?")
    gain = number(beat.fields.get("gain_pp"))
    acceptance, b1 = number(beat.fields.get("acceptance")), number(beat.fields.get("b1"))
    tasks = ", ".join(f"{task} {number(beat.fields[task]):.3f}" for task in TASKS if task in beat.fields)
    detail = f"acceptance {acceptance:.4f} vs B1 {b1:.4f}, gain {gain:+.2f} pp ({tasks})"
    name = f"{tier} result"
    if smoke:
        flag = GREEN if gain > -0.5 else NEUTRAL
        return Check(flag, name, detail + ". Smoke checks the plumbing, not the gain.")
    if gain >= EXPECT["gain_green_pp"]:
        return Check(GREEN, name, detail + ". Clears the +2 pp bar.")
    if gain >= 0 or tier != "T2":
        return Check(NEUTRAL, name, detail + ". Below the +2 pp bar" +
                     (": prompt 4.b after collecting." if tier == "T2" else "."))
    return Check(RED, name, detail + ". Below B1: prompt 4.b after collecting.")


def evaluate(beats, job, status=None, now=None, errors=()):
    """All checks in the order the run reached them, and a one-line view of where it is now."""
    pair, smoke = job.get("pair"), bool(job.get("smoke"))
    checks = [status_check(status)] if status else []
    grouped = {}
    for beat in beats:
        # Each tier's result line sits where the run reached it, not with the first result.
        label = beat.fields.get("tier") or beat.fields.get("arm")
        key = f"result:{label}" if beat.phase == "result" else beat.phase
        grouped.setdefault(key, []).append(beat)
    history = {}
    for phase, group in grouped.items():
        if phase == "abort":
            checks.extend(Check(RED, "abort", beat.fields.get("reason", "no reason given")) for beat in group)
        elif phase.startswith("result:"):
            checks.extend(result_check(beat, smoke) for beat in group)
        elif phase.endswith("-oom-retry"):
            last = group[-1].fields
            checks.append(Check(NEUTRAL, "out of memory", f"{phase}: batch halved to "
                                f"{last.get('max_sequences')} sequences / {last.get('max_tokens')} tokens. "
                                "A second OOM aborts."))
        elif match := re.fullmatch(r"(B1|T\d)(?:-baseline|-m1)?-replay", phase):
            # Under m1 the drafter is the same but the yardstick is not: label it apart from baseline B1.
            label = f"{match[1]} m1" if phase.endswith("-m1-replay") else match[1]
            baseline = grouped.get("B1-replay") or grouped.get("B1-baseline-replay") or ()
            checks.append(replay_check(label, group, pair, baseline))
        elif match := re.fullmatch(r"(T\d)-fit-(\d+)", phase):
            checks.append(fit_check(match[1], int(match[2]), group, job))
        elif match := re.fullmatch(r"(T\d)-score-(\d+)-epoch-(\d+)", phase):
            checks.append(score_check(match[1], int(match[2]), int(match[3]), group, pair, smoke, history))
    # A finished fit's loss readings are single batches; its validation score is the better judge.
    for check in checks:
        match = re.fullmatch(r"(T\d) fit (\d+)", check.name)
        if not match or check.flag != NEUTRAL or check.pending or "finished" not in check.detail:
            continue
        scores = [other for other in checks if other.name.startswith(f"{match[1]} score {match[2]} ")
                  and not other.pending]
        if scores and all(other.flag == GREEN for other in scores):
            check.flag = GREEN
            check.detail = check.detail.replace(". Flat so far; single-batch readings are noisy.",
                                                ". Readings are noisy single batches; its score line is green.")
    if errors and not grouped.get("abort"):
        checks.append(Check(RED, "error", errors[0][:200]))
    running = status == "running" or (status is None and now is not None)
    if beats and now is not None and running:
        now_clock = now.hour * 3600 + now.minute * 60 + now.second + beats[-1].clock // 86400 * 86400
        if now_clock < beats[-1].clock - 43200:
            now_clock += 86400
        silence = max(0, now_clock - beats[-1].clock)
        detail = f"last heartbeat {silence} s ago"
        if silence > EXPECT["silence_red_s"]:
            checks.append(Check(RED, "heartbeat", detail + ". Silent too long: open the kernel page."))
        elif silence > EXPECT["silence_neutral_s"]:
            checks.append(Check(NEUTRAL, "heartbeat", detail + ". Slower than the 30 s keepalive."))
        else:
            checks.append(Check(GREEN, "heartbeat", detail))
        end = now_clock
    else:
        end = beats[-1].clock if beats else 0
    if status == "running" and not beats:
        checks.append(Check(NEUTRAL, "heartbeat", "running, no heartbeat yet (installing packages)", pending=True))
    max_minutes = job.get("max_minutes")
    if beats and max_minutes:
        used = (end - beats[0].clock) / 60
        detail = f"{used:.0f} of {max_minutes} min used"
        share = used / max_minutes
        if status in FINAL or not running:
            checks.append(Check(GREEN, "time", detail))
        elif share > EXPECT["budget_red"]:
            checks.append(Check(RED, "time", detail + ". About to hit max_minutes and abort."))
        elif share > EXPECT["budget_neutral"]:
            checks.append(Check(NEUTRAL, "time", detail + ". Most of the budget is gone."))
        else:
            checks.append(Check(GREEN, "time", detail))
    if status in FINAL and not beats:
        checks.append(Check(NEUTRAL, "log", "the job has ended and its live log is gone. Collect it with "
                                            "`uv run python -m demo.kaggle_run --resume <timestamp>`, then run this again."))
    elif status == "complete":
        reported = {key.removeprefix("result:") for key in grouped if key.startswith("result:")}
        missing = [tier for tier in job.get("tier_order", ["T1", "T2"]) if tier not in reported]
        if missing:
            checks.append(Check(RED, "results", f"finished without a result line for {', '.join(missing)}"))
    return checks, describe_now(beats, status)


def describe_now(beats, status):
    if not beats:
        return f"Kaggle: {status or 'unknown'}, no heartbeat yet"
    last = beats[-1]
    parts = [last.phase]
    if last.step is not None:
        parts.append(f"step {last.step}/{last.total}" if last.total else f"step {last.step}")
    parts.extend(f"{key}={value}" for key, value in last.fields.items())
    hours, rest = divmod(last.clock % 86400, 3600)
    return f"{' '.join(parts)}  (at {hours:02d}:{rest // 60:02d}:{rest % 60:02d}Z)"


def verdict(checks):
    active = [check.flag for check in checks if not check.pending]
    return worst(*active) if active else NEUTRAL


def report(title, now_line, checks):
    lines = [title, f"Now: {now_line}", ""]
    width = max([len(check.name) for check in checks] + [4])
    for check in checks:
        flag = NEUTRAL + " (wait)" if check.pending else check.flag
        lines.append(f"{flag:<15} {check.name:<{width}}  {check.detail}")
    final = verdict(checks)
    lines += ["", f"Overall: {final.upper()}. {VERDICT[final]}"]
    return "\n".join(lines), EXIT[final]


def live_lines(api, kernel, idle_seconds=8, max_seconds=90):
    """Every log line so far: the live stream replays the session, then goes quiet between heartbeats."""
    from demo.kaggle_watch import parse_event, stream_events

    owner, slug = kernel.split("/")
    lines, started = [], time.monotonic()
    try:
        for line in stream_events(api, owner, slug, read_timeout=idle_seconds):
            event = parse_event(line)
            if event is not None:
                lines.extend(event["data"].splitlines())
            if time.monotonic() - started > max_seconds:
                break
    except Exception as error:  # the read timeout after the replay ends the snapshot
        if not re.search(r"timeout|timed out|connection|chunked", f"{type(error).__name__} {error}", re.I):
            raise
    return lines


def load_run(timestamp, log=None):
    stage = PROJECT_ROOT / "build/submissions" / timestamp
    submission_path = stage / "submission.json"
    if not submission_path.is_file():
        raise ValueError(f"No staged submission {timestamp}; use the timestamp from the run card.")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    kernel = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))["id"].lower()
    job = {}
    if submission.get("job"):
        job_path = PROJECT_ROOT / submission["job"]
        if job_path.is_file():
            job = tomllib.loads(job_path.read_text(encoding="utf-8"))
    collected = PROJECT_ROOT / "results" / f"submission-{timestamp}.json"
    if log:
        return kernel, job, Path(log).read_text(encoding="utf-8", errors="replace").splitlines(), None, "log file"
    if collected.is_file():
        result = json.loads(collected.read_text(encoding="utf-8"))
        status = "complete" if result.get("status") == "complete" else result.get("status", "error")
        return kernel, job, result.get("run_log", "").splitlines(), status, "collected result"
    from demo.kaggle_watch import get_api, status_of

    api = get_api()
    status = status_of(api, kernel)
    lines = live_lines(api, kernel) if status != "queued" else []
    return kernel, job, lines, status, "live Kaggle log"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("timestamp", help="The timestamp printed on the run card")
    parser.add_argument("--log", help="Judge a saved log file instead of fetching the run's log")
    parser.add_argument("--visual", action="store_true", help="Also draw the run to a PNG and open it")
    parser.add_argument("--no-open", action="store_true", help="With --visual, write the PNG without opening it")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{8}T\d{12}Z", args.timestamp):
        parser.error("Expected the UTC timestamp from the run card.")
    try:
        kernel, job, lines, status, source = load_run(args.timestamp, args.log)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    beats = parse_beats(lines)
    errors = [line for line in lines if line.startswith("Traceback") or re.match(r"\w+Error: ", line)]
    live = source == "live Kaggle log"
    checks, now_line = evaluate(beats, job, status=status, now=datetime.now(timezone.utc) if live else None,
                                errors=errors)
    mode = "smoke" if job.get("smoke") else "full"
    title = f"{args.timestamp} | {job.get('pair', '?')} {mode} | {kernel} | from the {source}"
    text, code = report(title, now_line, checks)
    print(text)
    if args.visual:
        from training.red_flag_figure import draw

        summary_path = PROJECT_ROOT / "results/runs" / args.timestamp / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        path = PROJECT_ROOT / "build/red-flag" / f"{args.timestamp}.png"
        draw(path, title, now_line, checks, verdict(checks), beats, job, summary)
        print(f"Figure: {path}")
        if not args.no_open and hasattr(os, "startfile"):
            os.startfile(path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
