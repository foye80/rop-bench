#!/usr/bin/env python
"""Generate the manuscript figures for ROP-Bench from verified numbers
(see notes/results.md). Outputs flat figure files into paper_overleaf/.

Fig1  RQ1  cross-site AUC heatmap (ROP presence, DINOv2@518 linear probe)
Fig2  Visual examples of high-confidence source-to-target failures
Fig3  RQ2  forest plot, DINOv2 - RETFound paired AUC diff
Fig4  RQ3  annotator-agreement envelope (FARFUM, 5 graders, plus disease)

Run:  env/bin/python scripts/make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def save(fig, stem):
    base = os.path.join(OUT, stem)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".eps", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".tif", dpi=600, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", os.path.normpath(base + ".pdf"))
    print("wrote", os.path.normpath(base + ".eps"))
    print("wrote", os.path.normpath(base + ".tif"))


def fig_crosssite():
    train_sites = ["Czech", "Shen"]
    test_sites = ["Czech", "Shen", "HVD"]
    # rows = train, cols = test. HVDROPDB is external test only.
    A = np.array([
        [0.926, 0.822, 0.739],
        [0.866, 0.999, 0.953],
    ])
    fig, ax = plt.subplots(figsize=(3.1, 2.15))
    cmap = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#f7f7f7", "#2166ac"])
    im = ax.imshow(A, cmap=cmap, vmin=0.70, vmax=1.00)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            diag = j < 2 and train_sites[i] == test_sites[j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="black", fontweight="bold" if diag else "normal")
            if diag:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           ec="black", lw=1.6))
    ax.set_xticks(range(len(test_sites))); ax.set_xticklabels(test_sites, rotation=20, ha="right")
    ax.set_yticks(range(len(train_sites))); ax.set_yticklabels(train_sites)
    ax.set_xlabel("Test center"); ax.set_ylabel("Train center")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("AUC", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, "Fig1")


def fig_forest():
    # direction, diff, lo, hi, significant  (DINOv2 - RETFound, AUC)
    rows = [
        ("Shen to HVD",            0.188, 0.13, 0.25, True),
        ("Czech to HVD",           0.101, 0.03, 0.18, True),
        ("Shen to Czech",          0.071, -0.05, 0.21, False),
        ("Shen (in)",               0.045, 0.01, 0.09, True),
        ("Czech to Shen",          0.034, -0.01, 0.08, False),
        ("Czech (in)",              0.033, -0.07, 0.13, False),
    ]
    fig, ax = plt.subplots(figsize=(3.1, 2.15))
    y = np.arange(len(rows))[::-1]
    for yi, (_, d, lo, hi, sig) in zip(y, rows):
        c = "#2166ac" if sig else "0.55"
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", ms=4.5,
                    color=c, ecolor=c, elinewidth=1.3, capsize=2.5)
    ax.axvline(0, color="0.3", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("AUC difference (DINOv2 minus RETFound)")
    ax.set_xlim(-0.12, 0.40)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2166ac", label="sig."),
                       Line2D([], [], marker="o", ls="", color="0.55", label="ns")],
              fontsize=7, loc="lower right", frameon=False)
    save(fig, "Fig3")


def fig_kappa():
    # (label, mean, min, max)
    rows = [
        ("Human vs human", 0.806, 0.671, 1.000),
        ("Model vs grader", 0.784, 0.494, 0.908),
    ]
    fig, ax = plt.subplots(figsize=(3.1, 1.55))
    y = [1, 0]
    colors = ["#4d9221", "#2166ac"]
    pale = ["#b8d8a2", "#a8c4df"]
    for yi, (lab, m, lo, hi), c, pc in zip(y, rows, colors, pale):
        ax.hlines(yi, lo, hi, color=pc, lw=6)
        ax.plot(m, yi, "o", color=c, ms=7)
        ax.text(hi + 0.01, yi, f"{m:.2f} [{lo:.2f},{hi:.2f}]", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, 1.6); ax.set_xlim(0.45, 1.18)
    ax.set_xlabel("Cohen kappa (binary plus disease)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "Fig4")


def fig_failures():
    root = os.environ.get("ROP_BENCH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    cases = [
        ("a", os.path.join(root, "data/raw/hvdropdb/ext/RetCam_ROP/47.png")),
        ("b", os.path.join(root, "data/raw/hvdropdb/ext/Neo_Normal/27.png")),
        ("c", os.path.join(root, "data/raw/czech/ext/images_stack/084_M_GA27_BW990_PA40_DG9_PF0_D2_S06_3.jpg")),
        ("d", os.path.join(root, "data/raw/czech/ext/images_stack/096_F_GA31_BW2000_PA36_DG0_PF0_D2_S02_3.jpg")),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.75))
    for ax, (letter, path) in zip(axes.ravel(), cases):
        im = Image.open(path).convert("RGB")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
        ax.text(0.035, 0.94, letter, transform=ax.transAxes,
                ha="left", va="top", color="white", fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="none", pad=2.5))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015,
                        wspace=0.04, hspace=0.08)
    save(fig, "Fig2")


if __name__ == "__main__":
    fig_crosssite()
    fig_failures()
    fig_forest()
    fig_kappa()
    print("done.")
