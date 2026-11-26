"""Acceptance per task for B1, T1 and T2 (P2, 320 validation prompts, oracle replay, K = 3)."""
from _style import P2_FULL, OUT, FILLS, HATCHES, INK, load, setup, tidy, plt

summary = load(P2_FULL)
tasks = ["math", "code", "instruction", "summarization"]
tiers = [("B1", "B1 (untrained)"), ("T1", "T1 (output adapter)"), ("T2", "T2 (LoRA)")]
setup()
fig, ax = plt.subplots(figsize=(6.0, 3.0))
width = 0.26
for index, (tier, label) in enumerate(tiers):
    values = [100 * summary["tiers"][tier]["per_task"][task]["acceptance"] for task in tasks]
    xs = [position + (index - 1) * width for position in range(len(tasks))]
    ax.bar(xs, values, width - 0.03, color=FILLS[index], edgecolor=INK, linewidth=0.6,
           hatch=HATCHES[index], label=label)
ax.set_xticks(range(len(tasks)), tasks)
ax.set_ylabel("Acceptance (%)")
ax.set_ylim(0, 45)
tidy(ax)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "acceptance_per_task.svg")
