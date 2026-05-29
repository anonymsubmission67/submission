"""
PB-A single compass panel: Likert (diamond) and free text (star) per model.

Combined figure in ``evaluation/build_main_plot.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

XLIM = (-0.5, 0.2)
YLIM = (-0.5, 0.2)
PBA_PAD = 0.0
TOP_LABEL = "Authoritarian/Communitarian"

LIKERT_MARKER = "D"
LIKERT_SIZE = 70
FT_MARKER = "*"
FT_SIZE = 120

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from utils import draw_eco_social_axis_labels, llm_color, target_llms

from .utils import (
    economic_social_merged_mapped,
    load_free_text_matrix,
    load_questionnaire_concat,
    merge_compass_coordinates,
)


def _compass_direction_labels(ax, top_label: str, *, margin_scale: float = 1.0):
    """Left/Right/Libertarian + top label (Authoritarian / Communitarian / combined)."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin
    oy_top = 0.04 * yr * margin_scale if yr > 0 else 0.06 * margin_scale
    oy_bottom = 0.07 * yr * margin_scale if yr > 0 else 0.1 * margin_scale
    ox = 0.075 * xr * margin_scale if xr > 0 else 0.1 * margin_scale

    def _clip_x(x):
        return np.clip(x, xmin + 1e-9 * max(xr, 1.0), xmax - 1e-9 * max(xr, 1.0))

    def _clip_y(y):
        return np.clip(y, ymin + 1e-9 * max(yr, 1.0), ymax - 1e-9 * max(yr, 1.0))

    xc = _clip_x(0.0) if xmin < 0 < xmax else (xmin + xmax) / 2.0
    yc = _clip_y(0.0) if ymin < 0 < ymax else (ymin + ymax) / 2.0

    fs = 12 if abs(xmax) <= 2 and abs(ymax) <= 2 else 11
    kw = dict(ha="center", va="center", color="black", fontsize=fs)
    kw_side = dict(ha="center", va="center", color="black", fontsize=fs)

    ax.text(xc, ymax + oy_top, top_label, **kw)
    ax.text(xc, ymin - oy_bottom, "Libertarian", **kw)
    ax.text(xmin - ox, yc, "Left", **kw_side)
    ax.text(xmax + ox, yc, "Right", **kw_side)


def _add_quadrant_fills(ax, xmin: float, xmax: float, ymin: float, ymax: float):
    """Four Political Compass tinted regions split at economic=0 / social=0."""
    rects = []

    nw_x0, nw_x1 = xmin, min(0.0, xmax)
    nw_y0, nw_y1 = max(0.0, ymin), ymax
    if nw_x0 < nw_x1 and nw_y0 < nw_y1:
        rects.append((nw_x0, nw_y0, nw_x1 - nw_x0, nw_y1 - nw_y0, "#EE7D79"))

    ne_x0, ne_x1 = max(0.0, xmin), xmax
    ne_y0, ne_y1 = max(0.0, ymin), ymax
    if ne_x0 < ne_x1 and ne_y0 < ne_y1:
        rects.append((ne_x0, ne_y0, ne_x1 - ne_x0, ne_y1 - ne_y0, "#5697DF"))

    sw_x0, sw_x1 = xmin, min(0.0, xmax)
    sw_y0, sw_y1 = ymin, min(0.0, ymax)
    if sw_x0 < sw_x1 and sw_y0 < sw_y1:
        rects.append((sw_x0, sw_y0, sw_x1 - sw_x0, sw_y1 - sw_y0, "#ADEB9F"))

    se_x0, se_x1 = max(0.0, xmin), xmax
    se_y0, se_y1 = ymin, min(0.0, ymax)
    if se_x0 < se_x1 and se_y0 < se_y1:
        rects.append((se_x0, se_y0, se_x1 - se_x0, se_y1 - se_y0, "#BA9CE7"))

    for x0, y0, w, h, c in rects:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                facecolor=c,
                alpha=0.3,
                edgecolor="none",
                zorder=0,
            )
        )


def draw_compass_background(
    ax,
    xlim=None,
    ylim=None,
    top_label="Authoritarian",
    *,
    direction_margin_scale: float = 1.0,
):
    """Quadrant shading + directional labels; split at economic=0 / social=0."""
    if xlim is None or ylim is None:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        _add_quadrant_fills(ax, -1.2, 1.2, -1.2, 1.2)
        kw = dict(ha="center", va="center", color="black", fontsize=12)
        ax.text(0, 1.18, top_label, **kw)
        ax.text(0, -1.3, "Libertarian", **kw)
        ax.text(-1.3, 0, "Left", **kw)
        ax.text(1.3, 0, "Right", **kw)
        return

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    xmin, xmax = xlim
    ymin, ymax = ylim
    _add_quadrant_fills(ax, xmin, xmax, ymin, ymax)
    _compass_direction_labels(ax, top_label, margin_scale=direction_margin_scale)


def style_compass_axes(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def _model_point_color(model: str) -> str:
    return llm_color(model)["edgecolor"]


def _load_pba_merged_matrices():
    df_l = merge_compass_coordinates(
        load_questionnaire_concat("compass"),
        load_questionnaire_concat("coordinates"),
    )
    df_ft = merge_compass_coordinates(
        load_free_text_matrix("compass"),
        load_free_text_matrix("coordinates"),
    )
    return df_l, df_ft


def _pba_limits_padded(pba_pad: float = PBA_PAD) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax = XLIM
    ymin, ymax = YLIM
    return (xmin - pba_pad, xmax + pba_pad), (ymin - pba_pad, ymax + pba_pad)


def draw_pba_single_on_ax(
    ax: plt.Axes,
    *,
    pba_pad: float = PBA_PAD,
    direction_margin_scale: float = 0.85,
    likert_size: float = LIKERT_SIZE,
    ft_size: float = FT_SIZE,
) -> None:
    df_l, df_ft = _load_pba_merged_matrices()
    xlim, ylim = _pba_limits_padded(pba_pad)

    draw_compass_background(
        ax,
        xlim=xlim,
        ylim=ylim,
        top_label=TOP_LABEL,
        direction_margin_scale=direction_margin_scale,
    )

    for llm in target_llms:
        color = _model_point_color(llm)
        eco_l, soc_l = economic_social_merged_mapped(df_l, llm, free_text=False)
        eco_f, soc_f = economic_social_merged_mapped(df_ft, llm, free_text=True)

        if np.isfinite(eco_l) and np.isfinite(soc_l):
            ax.scatter(
                eco_l,
                soc_l,
                c=color,
                marker=LIKERT_MARKER,
                s=likert_size,
                edgecolors="none",
                zorder=5,
            )
        if np.isfinite(eco_f) and np.isfinite(soc_f):
            ax.scatter(
                eco_f,
                soc_f,
                c=color,
                marker=FT_MARKER,
                s=ft_size,
                edgecolors="none",
                zorder=5,
            )

    style_compass_axes(ax)
    draw_eco_social_axis_labels(ax)
    ax.set_aspect("equal", adjustable="box")


def pba_modality_legend_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color="0.35",
            marker=LIKERT_MARKER,
            linestyle="None",
            markersize=7,
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            markeredgewidth=0,
            label="Likert",
        ),
        mlines.Line2D(
            [],
            [],
            color="0.35",
            marker=FT_MARKER,
            linestyle="None",
            markersize=11,
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            markeredgewidth=0,
            label="Free text",
        ),
    ]


if __name__ == "__main__":
    from build_main_plot import build_main_plot

    build_main_plot()
