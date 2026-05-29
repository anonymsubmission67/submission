"""
Combined main figure: PB-T scatter (RQ1, left) + PB-A compass (right).

Run from repo root: ``python evaluation/rq2_pba_pbt/build_main_plot.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from rq1_no_persona.build_pbt_scatter import (
    load_me_diff,
    draw_pbt_scatter_on_ax,
    pbt_scatter_model_legend_handles,
    pbt_scatter_prompt_legend_handles,
)
from rq2_pba_pbt.build_pba_plot_single import draw_pba_single_on_ax, pba_modality_legend_handles

OUTPUT_PATH = Path("output/images/main_plot.png")

LEGEND_FONTSIZE = 10
LEGEND_TITLE_FONTSIZE = 10
PBT_MARKER_CIRCLE = 50
PBT_MARKER_RECT = 52
PBA_LIKERT_SIZE = 45
PBA_FT_SIZE = 120
PBA_PAD = 0.10
PBA_DIRECTION_MARGIN = 0.55

LEGEND_KW = dict(
    framealpha=1,
    facecolor="white",
    columnspacing=0.8,
    handletextpad=0.35,
    labelspacing=0.55,
    borderpad=0.35,
    fontsize=LEGEND_FONTSIZE,
    title_fontsize=LEGEND_TITLE_FONTSIZE,
)


def build_main_plot() -> Path:
    me_diff = load_me_diff()

    fig = plt.figure(figsize=(14, 5.2))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.26, 1.0, 1.0],
        wspace=0.18,
        left=0.07,
        right=0.98,
        top=0.98,
        bottom=0.04,
    )

    ax_models = fig.add_subplot(outer[0, 0])
    ax_models.axis("off")

    ax_pbt = fig.add_subplot(outer[0, 1])
    ax_pba = fig.add_subplot(outer[0, 2])

    draw_pbt_scatter_on_ax(
        ax_pbt,
        me_diff,
        marker_size_circle=PBT_MARKER_CIRCLE,
        marker_size_rect=PBT_MARKER_RECT,
    )
    draw_pba_single_on_ax(
        ax_pba,
        pba_pad=PBA_PAD,
        direction_margin_scale=PBA_DIRECTION_MARGIN,
        likert_size=PBA_LIKERT_SIZE,
        ft_size=PBA_FT_SIZE,
    )
    ax_pbt.set_aspect("equal", adjustable="box")

    ax_models.legend(
        handles=pbt_scatter_model_legend_handles(me_diff),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        title="Models",
        **LEGEND_KW,
    )

    ax_pbt.legend(
        handles=pbt_scatter_prompt_legend_handles(me_diff),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        title="prompt conditions",
        **LEGEND_KW,
    )

    ax_pba.legend(
        handles=pba_modality_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        title="response types",
        **LEGEND_KW,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Main plot saved to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_main_plot()
