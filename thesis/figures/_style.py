"""Shared print style: greyscale-safe fills, hatches and markers. Data come from result files only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
P2_FULL = ROOT / "results/runs/20261124T052050221490Z/summary.json"
P2_COMBO = ROOT / "results/runs/20261124T063633692969Z/summary.json"
OUT = Path(__file__).resolve().parent
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d0d0d0"
FILLS = ["#ffffff", "#a6a6a6", "#3a3a3a"]   # light, mid, dark: separable in greyscale
HATCHES = ["////", "", ""]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def setup():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "hatch.color": MUTED,
                         "hatch.linewidth": 0.6})

def tidy(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
