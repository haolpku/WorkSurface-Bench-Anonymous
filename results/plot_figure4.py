"""Render the camera-ready Figure 4 dataset distributions.

Four subplots showing the benchmark's cross-cutting distributions, so a
reviewer can see the set is balanced across the axes it claims to cover.
"""

import json
import os
import matplotlib.pyplot as plt

data = json.load(open(os.path.join(os.path.dirname(__file__), "quality_report.json")))
d = data["distribution"]
paths = data["path_counts"]

COLORS = {
    "blue": "#56B4E9", "green": "#009E73", "orange": "#E69F00",
    "purple": "#7A5195", "vermillion": "#D55E00", "slate": "#4C566A",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, ax = plt.subplots(1, 4, figsize=(14.2, 3.6), sharey=False,
                       constrained_layout=True)


def styled_bars(axis, labels, values, colors):
    bars = axis.bar(labels, values, color=colors, width=0.68,
                    edgecolor="white", linewidth=0.6)
    axis.bar_label(bars, labels=[str(v) for v in values], padding=3,
                   fontsize=10.5)
    axis.set_ylim(0, max(values) * 1.18)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    return bars

# (a) task_type
tt_order = ["rag_only", "cross_surface", "table_only", "graph_only"]
styled_bars(ax[0], ["RAG", "Cross", "Table", "Graph"],
            [d["task_type"][k] for k in tt_order],
            [COLORS["blue"], COLORS["purple"], COLORS["orange"], COLORS["green"]])
ax[0].set_title("(a) Task type")
ax[0].set_ylabel("Number of tasks")

# (b) persona
per_order = sorted(d["persona"], key=lambda k: -d["persona"][k])
persona_labels = {
    "Logistics Manager": "Logistics\nMgr", "Operations Manager": "Ops\nMgr",
    "Product Manager": "Product\nMgr", "Backend Developer": "Backend\nDev",
    "Researcher": "Research",
}
styled_bars(ax[1], [persona_labels[p] for p in per_order],
            [d["persona"][p] for p in per_order], COLORS["slate"])
ax[1].set_title("(b) Persona")
ax[1].tick_params(axis="x", labelsize=9.5)

# (c) answer_type
at_order = ["number", "list", "string", "abstain"]
styled_bars(ax[2], ["Number", "List", "String", "Abstain"],
            [d["answer_type"].get(k, 0) for k in at_order],
            [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["vermillion"]])
ax[2].set_title("(c) Answer type")

# (d) derivation path
p_labels = {"deterministic": "Deterministic",
            "llm_augmented": "LLM\nassisted",
            "graph_table_cross": "Graph+Table\nrules",
            "rag_graph_cross": "RAG+Graph\nrules"}
p_order = ["deterministic", "llm_augmented", "graph_table_cross",
           "rag_graph_cross"]
styled_bars(ax[3], [p_labels[k] for k in p_order], [paths[k] for k in p_order],
            [COLORS["green"], COLORS["blue"], COLORS["purple"],
             COLORS["orange"]])
ax[3].set_title("(d) Derivation path")
ax[3].tick_params(axis="x", labelsize=9.5)

plt.savefig(os.path.join(os.path.dirname(__file__), "figure4_distribution.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(os.path.dirname(__file__), "figure4_distribution.pdf"),
            bbox_inches="tight", facecolor="white")
print("wrote figure4_distribution.png and figure4_distribution.pdf")
