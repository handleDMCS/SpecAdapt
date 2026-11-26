"""Tokens per target call and drafter steps for the four M1 x T2 arms (P2, 80 validation prompts)."""
from _style import P2_COMBO, OUT, FILLS, HATCHES, INK, load, setup, tidy, plt

arms = load(P2_COMBO)["arms"]
order = [("B1-baseline", "B1"), ("B1-m1", "B1 + M1"), ("T2-baseline", "T2"), ("T2-m1", "T2 + M1")]
fills = [FILLS[0], FILLS[1], FILLS[0], FILLS[1]]
hatches = ["", "////", "", "////"]
setup()
fig, (left, right) = plt.subplots(1, 2, figsize=(6.2, 2.8))
for ax, key, title, fmt in ((left, "tokens_per_round", "(a) Tokens per target call", "{:.3f}"),
                            (right, "draft_steps", "(b) Drafter steps", "{:,}")):
    values = [arms[name][key] for name, _ in order]
    bars = ax.bar(range(4), values, 0.62, color=fills, edgecolor=INK, linewidth=0.6, hatch=hatches)
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(4), [label for _, label in order], fontsize=9)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_ylim(0, max(values) * 1.15)
    tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "m1_arms.svg")
