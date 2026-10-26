"""Slide figure: LightGBM APS vs tree complexity on both windows, with ridge.

    python presentation/make_lgbm_complexity.py   # -> presentation/lgbm_complexity.png

Reads results/<window>/lgb_*/record.json (learning rate 0.02, stop on loss,
mean of 3 seeds) and ridge_eqall (contract ridge on the same rows, dashed).
"""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "lgbm_complexity.png"

STEPS = [("lgb_depth1", "2\n(depth 1)"), ("lgb_depth2", "4\n(depth 2)"),
         ("lgb_depth3", "8\n(depth 3)"), ("lgb_leaves15", "15"), ("lgb_lr02_loss", "63")]
WINDOWS = [("v2_52w", "Window 1", "#1f4e9c"),       # dark blue
           ("v3_52w_jan24", "Window 2", "#a51d2d")]  # dark red
INK, INK2, GRID, SURFACE = "#1a1a1a", "#6b6b6b", "#ececec", "#ffffff"


def aps(window, run):
    return json.loads((REPO / "results" / window / run / "record.json").read_text()
                      )["report"]["windows"]["val"]["APS_mean_bps"]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)

    x = range(len(STEPS))
    for window, label, color in WINDOWS:
        ridge = aps(window, "ridge_eqall")
        end = len(STEPS) - 1 + 0.25  # dashed line stops before its label
        ax.plot([-0.3, end], [ridge, ridge], color=color, lw=1.5, ls=(0, (4, 4)), alpha=0.7,
                zorder=1)
        ax.text(end + 0.06, ridge, f"Ridge {ridge:.2f}", color=color, va="center", fontsize=11)
        ys = [mean(aps(window, f"{tag}_s{s}") for s in range(3)) for tag, _ in STEPS]
        ax.plot(x, ys, color=color, lw=2.6, marker="o", ms=8, mec=SURFACE, mew=2,
                zorder=3, label=label)

    ax.set_xticks(list(x), [lab for _, lab in STEPS], color=INK)
    ax.set_xlim(-0.3, len(STEPS) + 0.15)
    ax.set_ylim(1.45, 2.05)
    ax.set_ylabel("APS (bps)", color=INK2)
    ax.set_xlabel("Leaves per tree", color=INK2, labelpad=8)
    ax.set_title("LightGBM beats ridge once trees can combine features",
                 loc="left", color=INK, fontsize=15, fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
