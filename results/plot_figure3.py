"""Render the camera-ready Figure 3 from figure3_data.json."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "figure3_data.json")))

COLORS = {"S2": "#56B4E9", "S3": "#009E73", "S4": "#E69F00",
          "S5": "#7A5195", "S6": "#D55E00"}
LABELS = {"S2": "Always-RAG", "S3": "Naive-router", "S4": "ReAct-all",
          "S5": "Gold-constrained", "S6": "Gold-hint/all"}
MARKERS = {"S2": "o", "S3": "s", "S4": "^", "S5": "D", "S6": "P"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.titlesize": 15,
    "axes.labelsize": 14, "legend.fontsize": 12, "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5, "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.8), constrained_layout=True)

# (a) Encode the agent setting explicitly rather than drawing an undifferentiated cloud.
a = d["scatter_3a"]
settings = [label.split(":", 1)[0] for label in a["labels"]]
for setting in ("S2", "S3", "S4", "S5", "S6"):
    idx = [i for i, value in enumerate(settings) if value == setting]
    ax[0].scatter([a["x_route_f1"][i] for i in idx],
                  [a["y_answer"][i] for i in idx], s=44,
                  color=COLORS[setting], marker=MARKERS[setting], alpha=0.78,
                  edgecolor="white", linewidth=0.45, label=LABELS[setting])
ax[0].set(xlabel="Route F1", ylabel="Answer score",
          title=rf"(a) Routing and answering ($\rho={a['spearman_rho']:.2f}$)")
ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.02)
ax[0].legend(loc="upper left", frameon=False, handletextpad=0.3)

# (b) Grouped bars with readable model abbreviations.
g = d["guidance_3b"]
short = {"gpt-4o-mini": "4o-mini", "deepseek-v4-pro": "DS-V4-Pro",
         "gemini-3.1-pro-preview": "Gemini-3.1",
         "gpt-5.5": "GPT-5.5"}
model_order = ["gpt-4o-mini", "deepseek-v4-pro",
               "gemini-3.1-pro-preview", "gpt-5.5"]
order = [g["models"].index(m) for m in model_order if m in g["models"]]
ordered_models = [g["models"][i] for i in order]
x = np.arange(len(ordered_models)); width = 0.19
series = [("S3_naive", "S3 Naive", COLORS["S3"], "//"),
          ("S4_react", "S4 ReAct", COLORS["S4"], ".."),
          ("S6_hint", "S6 Gold-hint/all", COLORS["S6"], "xx"),
          ("S5_constrained", "S5 Gold-constr.", COLORS["S5"], "")]
for offset, (key, label, color, hatch) in zip((-1.5, -.5, .5, 1.5), series):
    ax[1].bar(x + offset * width, [(g[key][i] or 0) for i in order], width,
              color=color, label=label, hatch=hatch, edgecolor="white", linewidth=0.5)
ax[1].set_xticks(x, [short.get(m, m) for m in ordered_models], rotation=18, ha="right")
ax[1].set(ylabel="Answer score", title="(b) Hint and tool restriction", ylim=(0, 0.9))
ax[1].set_yticks(np.arange(0, 0.71, 0.1))
ax[1].legend(loc="upper center", frameon=False, ncol=2,
             columnspacing=0.7, handletextpad=0.25)

# (c) Color and line style jointly encode the setting for grayscale robustness.
c = d["per_surface_3c"]
surface_labels = ["RAG", "Table", "Graph", "Cross"]
styles = {"S2": ":", "S3": "--", "S4": "-", "S5": "-.", "S6": (0, (3, 1, 1, 1))}
for setting in ("S2", "S3", "S4", "S5", "S6"):
    vals = c["lines"].get(setting)
    if vals:
        ax[2].plot(surface_labels, vals, marker=MARKERS[setting], markersize=7,
                   linewidth=2.2, linestyle=styles[setting], color=COLORS[setting],
                   label=LABELS[setting])
ax[2].set(ylabel="Answer score", title="(c) Answer by task type", ylim=(0, 1.18))
ax[2].set_yticks(np.arange(0, 1.01, 0.2))
ax[2].margins(x=0.04)
ax[2].legend(loc="upper center", frameon=False, ncol=2,
             columnspacing=0.9, handletextpad=0.3)

for axis in ax:
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)

out = os.path.join(HERE, "figure3.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
pdf_out = os.path.join(HERE, "figure3.pdf")
plt.savefig(pdf_out, bbox_inches="tight", facecolor="white")
print(f"wrote {out} and {pdf_out}")
