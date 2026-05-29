"""PB-T vs PB-A scatter panels (draw helpers; combined figure in ``build_main_plot.py``)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_evaluation_root = Path(__file__).resolve().parents[1]
if str(_evaluation_root) not in sys.path:
    sys.path.insert(0, str(_evaluation_root))

from rq2_pba_pbt.utils import load_model_metrics, pba_l_mean
from utils import llm_color, llm_colors, llm_info, target_llms

LEGEND_FONTSIZE = 8
LEGEND_TITLE_FONTSIZE = 8
LEGEND_MARKER_SIZE = 5
DEFAULT_POINT_SIZE = 60

PBA_VALUE = "pba_combined_both"
PBA_X_LABELS = {
    "pba_combined_both": "PB-A (combined)",
    "pba_l_mean": "PB-A (Likert avg.)",
}
PBA_X_COL = "_pba_x"
SPEAKER_Y_COL = "speaker_R"
SPEAKER_Y_LABEL = "claim maker rel."


@dataclass(frozen=True)
class PbtScatterData:
    pba_xlabel: str
    scatter_1: pd.DataFrame
    scatter_2: pd.DataFrame
    scatter_3: pd.DataFrame
    xlim: tuple[float, float]
    pbt_ylim: tuple[float, float]
    speaker_ylim: tuple[float, float]
    metrics_df: pd.DataFrame


def _pba_series_and_xlabel(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if PBA_VALUE == "pba_l_mean":
        return df.apply(pba_l_mean, axis=1), PBA_X_LABELS["pba_l_mean"]
    if PBA_VALUE not in df.columns:
        raise KeyError(
            f"PBA column {PBA_VALUE!r} missing from model_metrics.csv; "
            "run evaluation/rq2_pba_pbt/_01_run_data_preperation.py first."
        )
    label = PBA_X_LABELS.get(PBA_VALUE, PBA_VALUE)
    return pd.to_numeric(df[PBA_VALUE], errors="coerce"), label


def _padded_limits(values: pd.Series) -> tuple[float, float]:
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.1 if hi > lo else 0.1
    return lo - pad, hi + pad


def prepare_pbt_scatter_data() -> PbtScatterData:
    df = load_model_metrics().copy()
    df[PBA_X_COL], pba_xlabel = _pba_series_and_xlabel(df)

    if SPEAKER_Y_COL not in df.columns:
        raise KeyError(
            f"Column {SPEAKER_Y_COL!r} missing from model_metrics.csv; "
            "run regression + evaluation/rq2_pba_pbt/_01_run_data_preperation.py first."
        )
    df[SPEAKER_Y_COL] = pd.to_numeric(df[SPEAKER_Y_COL], errors="coerce")

    scatter_1 = df[["llm", PBA_X_COL, "pbt_1_both"]].dropna()
    scatter_2 = df[["llm", PBA_X_COL, "pbt_2_both"]].dropna()
    scatter_3 = df[["llm", PBA_X_COL, SPEAKER_Y_COL]].dropna()

    if scatter_1.empty and scatter_2.empty and scatter_3.empty:
        raise ValueError("Insufficient data for PB-T scatter panels")

    all_pba = pd.concat([scatter_1[PBA_X_COL], scatter_2[PBA_X_COL], scatter_3[PBA_X_COL]])
    xlim = _padded_limits(all_pba)

    pbt_vals = pd.concat([scatter_1["pbt_1_both"], scatter_2["pbt_2_both"]])
    pbt_ylim = _padded_limits(pbt_vals) if not pbt_vals.empty else (-0.1, 0.1)
    speaker_ylim = _padded_limits(scatter_3[SPEAKER_Y_COL]) if not scatter_3.empty else (-0.1, 0.1)

    return PbtScatterData(
        pba_xlabel=pba_xlabel,
        scatter_1=scatter_1,
        scatter_2=scatter_2,
        scatter_3=scatter_3,
        xlim=xlim,
        pbt_ylim=pbt_ylim,
        speaker_ylim=speaker_ylim,
        metrics_df=df,
    )


def _scatter_panel(
    ax,
    scatter_df: pd.DataFrame,
    y_col: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for llm in scatter_df["llm"].unique():
        llm_data = scatter_df[scatter_df["llm"] == llm]
        color = llm_color(llm)["edgecolor"]
        ax.scatter(
            llm_data[PBA_X_COL],
            llm_data[y_col],
            c=color,
            s=DEFAULT_POINT_SIZE,
            edgecolors="black",
            linewidth=0.5,
        )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    corr = scatter_df[PBA_X_COL].corr(scatter_df[y_col])
    ax.text(
        0.025,
        0.90,
        f"r = {corr:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def draw_pbt_scatter_panels(axes: Sequence[plt.Axes], data: PbtScatterData) -> None:
    ax1, ax2, ax3 = axes
    if not data.scatter_1.empty:
        _scatter_panel(ax1, data.scatter_1, "pbt_1_both", xlim=data.xlim, ylim=data.pbt_ylim)
    ax1.set_ylabel("PB-T party-agnostic", fontsize=11)

    if not data.scatter_2.empty:
        _scatter_panel(ax2, data.scatter_2, "pbt_2_both", xlim=data.xlim, ylim=data.pbt_ylim)
    ax2.set_ylabel("PB-T party-aware", fontsize=11)

    if not data.scatter_3.empty:
        _scatter_panel(ax3, data.scatter_3, SPEAKER_Y_COL, xlim=data.xlim, ylim=data.speaker_ylim)
    ax3.set_xlabel(data.pba_xlabel, fontsize=11)
    ax3.set_ylabel(SPEAKER_Y_LABEL, fontsize=11)


def model_scatter_legend_handles() -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for llm in target_llms:
        if llm not in llm_colors:
            continue
        llm_name = llm_info.get(llm, {}).get("name", llm.capitalize())
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=llm_colors[llm]["edgecolor"],
                marker="o",
                linestyle="None",
                markersize=LEGEND_MARKER_SIZE,
                label=llm_name,
            )
        )
    return handles


def pbt_panel_legend_handles() -> list[mlines.Line2D]:
    """Explicit panel semantics (all markers are circles)."""
    kw = dict(color="0.35", marker="o", linestyle="None", markeredgecolor="black", markeredgewidth=0.5)
    return [
        mlines.Line2D([], [], markersize=6, label="PB-T party-agnostic", **kw),
        mlines.Line2D([], [], markersize=6, label="PB-T party-aware", **kw),
        mlines.Line2D([], [], markersize=6, label="claim maker rel.", **kw),
    ]


if __name__ == "__main__":
    from rq2_pba_pbt.build_main_plot import build_main_plot

    build_main_plot()
