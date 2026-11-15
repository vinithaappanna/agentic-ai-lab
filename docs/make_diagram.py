#!/usr/bin/env python3
"""Regenerate docs/agent-loop.png - the README architecture diagram."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "agent-loop.png"

BOX = dict(boxstyle="round,pad=0.55", linewidth=1.4)


def box(ax, xy, text, face, edge="#334155", fontsize=11, weight="normal"):
    ax.add_patch(FancyBboxPatch((xy[0] - 0.5, xy[1] - 0.5), 1.0, 1.0,
                                mutation_scale=1, fc=face, ec=edge, **BOX))
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def arrow(ax, a, b, label="", rad=0.0, color="#475569", label_offset=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                                 color=color, linewidth=1.4,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mid = ((a[0] + b[0]) / 2 + label_offset[0], (a[1] + b[1]) / 2 + label_offset[1])
        ax.text(mid[0], mid[1], label, ha="center", fontsize=9, color=color,
                style="italic")


fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(-0.2, 10.2)
ax.set_ylim(-0.6, 5.4)
ax.axis("off")

task = (1.0, 4.2)
llm = (5.0, 4.2)
tool = (5.0, 1.2)
final = (9.0, 4.2)

box(ax, task, "Task\n(your prompt)", "#e0f2fe", weight="bold")
box(ax, llm, "LLM\none JSON object per turn", "#fef9c3", weight="bold")
box(ax, tool, "Tool registry\nvalidated call", "#dcfce7", weight="bold")
box(ax, final, "Final answer", "#fae8ff", weight="bold")

arrow(ax, (task[0] + 0.75, task[1]), (llm[0] - 1.05, llm[1]), "messages")
arrow(ax, (llm[0], llm[1] - 0.75), (tool[0], tool[1] + 0.75),
      '{"action": name, "action_input": {...}}', label_offset=(1.9, 0.1))
arrow(ax, (tool[0] - 0.6, tool[1] + 0.55), (llm[0] - 0.6, llm[1] - 0.75),
      "observation", rad=0.35, color="#16a34a", label_offset=(-0.8, -0.3))
arrow(ax, (llm[0] + 1.05, llm[1]), (final[0] - 0.8, final[1]), '{"final": "..."}')

ax.text(5.0, 0.05, "Loop repeats until \"final\" or max_steps; every step is recorded.",
        ha="center", fontsize=9, color="#64748b")

fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
