"""Teacher-forced top-1 agreement per configuration (P2, 320 validation prompts)."""
from _style import P2_FULL, OUT, INK, MUTED, load, setup, tidy, plt

trials = load(P2_FULL)["trials"]
setup()
fig, (left, right) = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
series_t2 = [(2, "o", "--", "epoch 2"), (4, "s", "-", "epoch 4")]
for epoch, marker, style, label in series_t2:
    rows = sorted((t["config"]["lr"], t["top1_agreement"]) for t in trials["T2"] if t["config"]["epochs"] == epoch)
    left.plot([r[0] for r in rows], [r[1] for r in rows], marker=marker, linestyle=style, color=INK,
              markersize=6, markerfacecolor="white" if epoch == 2 else INK, linewidth=1.2)
    left.annotate(label, (rows[-1][0], rows[-1][1]), xytext=(6, -3), textcoords="offset points", fontsize=9, color=INK)
series_t1 = [(8, "^", "--", "rank 8"), (16, "D", "-", "rank 16")]
for rank, marker, style, label in series_t1:
    rows = sorted((t["config"]["lr"], t["top1_agreement"]) for t in trials["T1"] if t["config"]["rank"] == rank)
    right.plot([r[0] for r in rows], [r[1] for r in rows], marker=marker, linestyle=style, color=INK,
               markersize=6, markerfacecolor="white" if rank == 8 else INK, linewidth=1.2)
    right.annotate(label, (rows[-1][0], rows[-1][1]), xytext=(6, -3), textcoords="offset points", fontsize=9, color=INK)
for ax, title in ((left, "(a) T2, LoRA rank 16"), (right, "(b) T1, output adapter, 2 epochs")):
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_title(title, fontsize=10, color=INK)
    tidy(ax)
left.set_ylabel("Top-1 agreement")
left.set_xlim(6e-5, 3e-3)
right.set_xlim(6e-4, 3e-2)
fig.tight_layout()
fig.savefig(OUT / "top1_sweep.svg")
