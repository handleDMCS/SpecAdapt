"""The --visual view of training_red_flag: one PNG per run, light theme, regenerated on every call."""

import re

from training.training_red_flag import EXPECT, GREEN, NEUTRAL, RED, TASKS, fit_label, number

# Reference palette (dataviz skill), light mode. Categorical slots in fixed order: B1, T2, T1.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
TIER_COLOR = {"B1": "#2a78d6", "T2": "#eb6834", "T1": "#1baf7a"}
STATUS = {GREEN: "#0ca30c", NEUTRAL: "#fab219", RED: "#d03b3b"}
WAIT = "#b9b8b2"
BAND = "#f0efec"


def shade(color, amount):
    """Mix a hex color toward the surface: 0 keeps it, 1 is the surface. An ordinal ramp for learning rates."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    white = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (w - c) * amount):02x}" for c, w in zip(rgb, white))


def collect(beats):
    fits, scores, replays, results = {}, {}, {}, {}
    for beat in beats:
        if match := re.fullmatch(r"(T\d)-fit-(\d+)", beat.phase):
            if beat.step and number(beat.fields.get("loss")) is not None:
                fits.setdefault((match[1], int(match[2])), []).append((beat.step, number(beat.fields["loss"])))
        elif match := re.fullmatch(r"(T\d)-score-(\d+)-epoch-(\d+)", beat.phase):
            if beat.done:
                scores.setdefault((match[1], int(match[2])), []).append(
                    (int(match[3]), number(beat.fields.get("top1_agreement"))))
        elif match := re.fullmatch(r"(B1|T\d)-replay", beat.phase):
            if beat.step and number(beat.fields.get("acceptance")) is not None:
                replays.setdefault(match[1], {"total": beat.total, "points": []})["points"].append(
                    (beat.step, number(beat.fields["acceptance"])))
        elif beat.phase == "result":
            results[beat.fields.get("tier")] = beat.fields
    return fits, scores, replays, results


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def empty(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def schedule_epochs(job, tier):
    try:
        if tier == "T2":
            checkpoints = job["t2"]["checkpoints"]
            return min(checkpoints) if job.get("smoke") else max(checkpoints)
        return job["t1"]["epochs"]
    except (KeyError, TypeError):
        return None


def draw_loss(ax, tier, fits, job, beats):
    style(ax, f"{tier} training loss, per configuration")
    series = {key: points for key, points in fits.items() if key[0] == tier}
    if not series:
        empty(ax, f"No {tier} loss readings yet")
        return
    count = max(index for _, index in series)
    totals = [beat.total for beat in beats if beat.phase.startswith(f"{tier}-fit-") and beat.total]
    epochs = schedule_epochs(job, tier)
    if totals and epochs:
        for epoch in range(1, epochs):
            x = totals[-1] * epoch / epochs
            ax.axvline(x, color=GRID, linewidth=1.2, zorder=0)
            ax.text(x, 1.0, f" ep {epoch + 1}", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8,
                    va="top")
    rates = sorted({cfg["lr"] for cfg in config_list(job, tier)}) if config_list(job, tier) else []
    for (_, index), points in sorted(series.items()):
        label = fit_label(job, tier, index) or f"fit {index}"
        lr = number(label.rsplit("lr ", 1)[-1]) if "lr " in label else None
        position = rates.index(lr) if lr in rates else index - 1
        amount = 0.55 * (1 - position / max(1, len(rates) - 1)) if len(rates) > 1 else 0
        dashed = tier == "T1" and "rank 16" in label
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=shade(TIER_COLOR[tier], amount), linewidth=2, linestyle="--" if dashed else "-",
                marker="o", markersize=4, label=label)
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=INK_2, va="center")
    ax.set_xlabel("optimizer step", color=INK_2, fontsize=9)
    ax.set_ylabel("loss (single batch)", color=INK_2, fontsize=9)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95, fontsize=8, labelcolor=INK_2,
              ncol=2 if count > 3 else 1, loc="best")


def config_list(job, tier):
    try:
        if tier == "T2":
            return [dict(lr=lr) for lr in job["t2"]["learning_rates"]]
        return [dict(rank=r, lr=lr) for r in job["t1"]["ranks"] for lr in job["t1"]["learning_rates"]]
    except (KeyError, TypeError):
        return []


def draw_top1(ax, scores, job):
    style(ax, "Top-1 agreement on validation answers, per checkpoint")
    if not scores:
        empty(ax, "No checkpoint scored yet")
        return
    rows = sorted(scores, key=lambda key: (key[0] != "T2", key[1]))
    best = max(rows, key=lambda key: max(top1 for _, top1 in scores[key]))
    for y, key in enumerate(rows):
        tier, index = key
        points = sorted(scores[key])
        xs = [top1 for _, top1 in points]
        ax.plot(xs, [y] * len(xs), color=TIER_COLOR[tier], linewidth=2, zorder=2)
        for epoch, top1 in points:
            ax.scatter([top1], [y], s=64, color=TIER_COLOR[tier], edgecolor=SURFACE, linewidth=2, zorder=3,
                       marker="o" if epoch == points[-1][0] else "D")
            ax.annotate(f"ep {epoch}", (top1, y), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.5, color=MUTED)
        if key == best:
            ax.annotate("highest", (xs[-1], y), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [f"{tier}  {fit_label(job, tier, index)}" for tier, index in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()
    base = EXPECT["untrained_top1"].get(job.get("pair"))
    if base is not None:
        ax.axvline(base, color=INK_2, linewidth=1.2, linestyle=":", zorder=1)
        ax.text(base, 0.01, f" untrained {base:.3f}", transform=ax.get_xaxis_transform(), fontsize=8,
                color=INK_2, va="bottom")
    ax.set_xlabel("share of target tokens the drafter ranks first", color=INK_2, fontsize=9)


def draw_replay(ax, replays):
    style(ax, "Replay acceptance, running total over validation prompts")
    if not replays:
        empty(ax, "No replay yet")
        return
    total = max(data["total"] or 0 for data in replays.values())
    if total:
        quarter = total / 4
        for i, task in enumerate(TASKS):
            if i % 2 == 0:
                ax.axvspan(i * quarter, (i + 1) * quarter, color=BAND, zorder=0, linewidth=0)
            ax.text((i + 0.5) * quarter, 1.0, task, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color=MUTED)
        ax.set_xlim(0, total)
    for tier in ("B1", "T2", "T1"):
        if tier not in replays:
            continue
        xs, ys = zip(*replays[tier]["points"])
        ax.plot(xs, ys, color=TIER_COLOR[tier], linewidth=2, marker="o", markersize=4, label=tier)
        ax.annotate(f"{tier} {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center")
    ax.set_xlabel("validation prompt (tasks run in this order)", color=INK_2, fontsize=9)
    ax.set_ylabel("accepted / proposed tokens", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")


def draw_results(ax, results, summary):
    style(ax, "Final acceptance per task (K = 3)")
    b1_tasks = (summary or {}).get("tiers", {}).get("B1", {}).get("per_task")
    tiers = [tier for tier in ("T2", "T1") if tier in results]
    if not tiers:
        empty(ax, "No result line yet")
        return
    groups = ("overall",) + TASKS
    shown = (["B1"] if b1_tasks else []) + tiers
    width = 0.8 / len(shown)
    for offset, tier in enumerate(shown):
        if tier == "B1":
            values = [summary["tiers"]["B1"]["acceptance"]] + [b1_tasks[task]["acceptance"] for task in TASKS]
        else:
            fields = results[tier]
            values = [number(fields.get("acceptance"))] + [number(fields.get(task)) for task in TASKS]
        xs = [i + (offset - (len(shown) - 1) / 2) * width for i in range(len(groups))]
        bars = ax.bar(xs, values, width=width, color=TIER_COLOR[tier], edgecolor=SURFACE, linewidth=2, label=tier)
        if tier == "T2":
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
    if not b1_tasks:
        b1 = number(results[tiers[0]].get("b1"))
        ax.axhline(b1, color=TIER_COLOR["B1"], linewidth=2, linestyle="--", label=f"B1 overall {b1:.3f}")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("accepted / proposed tokens", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")


def draw_verdict(ax, final, results, now_line):
    ax.axis("off")
    ax.add_patch(__import__("matplotlib.patches", fromlist=["FancyBboxPatch"]).FancyBboxPatch(
        (0.02, 0.62), 0.07, 0.28, boxstyle="round,pad=0.01", transform=ax.transAxes, color=STATUS[final]))
    ax.text(0.13, 0.76, final.upper(), transform=ax.transAxes, fontsize=24, fontweight="bold", color=INK,
            va="center")
    lines = []
    for tier in ("T2", "T1"):
        if tier in results:
            lines.append(f"{tier} gain over B1: {number(results[tier].get('gain_pp')):+.2f} pp "
                         f"(bar +{EXPECT['gain_green_pp']:g} pp)")
    if not lines:
        lines.append("No result line yet.")
    lines.append("Now: " + (now_line if len(now_line) < 80 else now_line[:77] + "..."))
    for i, line in enumerate(lines):
        ax.text(0.02, 0.46 - i * 0.14, line, transform=ax.transAxes, fontsize=10.5, color=INK if i < len(lines) - 1
                else INK_2, va="center")


def draw_flags(ax, checks):
    ax.axis("off")
    ax.set_title("Checks", loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    rows = len(checks)
    for i, check in enumerate(checks):
        y = 1 - (i + 0.5) / max(rows, 1)
        color = WAIT if check.pending else STATUS[check.flag]
        label = "neutral (wait)" if check.pending else check.flag
        ax.scatter([0.008], [y], s=70, color=color, marker="s", transform=ax.transAxes, clip_on=False)
        ax.text(0.02, y, label, transform=ax.transAxes, fontsize=9, color=INK, va="center", fontweight="bold")
        ax.text(0.12, y, check.name, transform=ax.transAxes, fontsize=9, color=INK, va="center")
        detail = check.detail if len(check.detail) <= 135 else check.detail[:132] + "..."
        ax.text(0.25, y, detail, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="center")


def draw(path, title, now_line, checks, final, beats, job, summary=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fits, scores, replays, results = collect(beats)
    flag_height = 0.32 * max(len(checks), 4)
    height = 13.5 + flag_height
    figure = plt.figure(figsize=(15, height), facecolor=SURFACE)
    grid = figure.add_gridspec(4, 2, height_ratios=[1, 1, 1, flag_height / 4.2], hspace=0.42, wspace=0.22,
                               top=1 - 0.75 / height, bottom=0.2 / height)
    figure.suptitle(title, x=0.06, ha="left", fontsize=13, color=INK, fontweight="bold", y=1 - 0.2 / height)
    draw_loss(figure.add_subplot(grid[0, 0]), "T2", fits, job, beats)
    draw_loss(figure.add_subplot(grid[0, 1]), "T1", fits, job, beats)
    draw_top1(figure.add_subplot(grid[1, 0]), scores, job)
    draw_replay(figure.add_subplot(grid[1, 1]), replays)
    draw_results(figure.add_subplot(grid[2, 0]), results, summary)
    draw_verdict(figure.add_subplot(grid[2, 1]), final, results, now_line)
    draw_flags(figure.add_subplot(grid[3, :]), checks)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path
