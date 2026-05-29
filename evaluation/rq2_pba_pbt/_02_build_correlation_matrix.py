"""
Pearson correlations between PB-T and PB-A sources.

Reads precomputed metrics from ``data/interim_results/model_metrics.csv``.

Run from repo root: ``python evaluation/rq2_pba_pbt/_03_build_correlation_matrix.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

script_dir = Path(__file__).resolve().parent
evaluation_dir = script_dir.parent
if str(evaluation_dir) not in sys.path:
    sys.path.insert(0, str(evaluation_dir))

from .utils import (
    CORR_MATRIX_AXIS_GROUPS,
    CORR_MATRIX_COLUMNS,
    CORR_MATRIX_LABELS,
    load_model_metrics,
)

# White at r=0; dark gray (not black) at r=±1.
_CORR_GRAY = "dimgray"
_CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_bw", [_CORR_GRAY, "white", _CORR_GRAY], N=256
)

# Inches per matrix cell; annotation fontsize stays fixed below.
_CELL_IN = 0.95
_CBAR_PAD_IN = 1.35
_ANNOT_FONTSIZE = 11
_GROUP_LABEL_FONTSIZE = 10
_INNER_LABEL_FONTSIZE = 9
_LEFT_GROUP_X = -0.14
_LEFT_INNER_X = -0.055
_TOP_GROUP_Y = 1.14
_TOP_INNER_Y = 1.02


def _flat_inner_labels() -> list[str]:
    return [inner for _, cols in CORR_MATRIX_AXIS_GROUPS for _, inner in cols]


def _add_top_axis_labels(ax) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    idx = 0
    for group_label, columns in CORR_MATRIX_AXIS_GROUPS:
        span = len(columns)
        ax.text(
            idx + span / 2.0,
            _TOP_GROUP_Y,
            group_label,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=_GROUP_LABEL_FONTSIZE,
            fontweight="bold",
            clip_on=False,
        )
        idx += span

    for i, inner in enumerate(_flat_inner_labels()):
        ax.text(
            i + 0.5,
            _TOP_INNER_Y,
            inner,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=_INNER_LABEL_FONTSIZE,
            clip_on=False,
        )


def _add_left_axis_labels(ax) -> None:
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    idx = 0
    for group_label, columns in CORR_MATRIX_AXIS_GROUPS:
        span = len(columns)
        ax.text(
            _LEFT_GROUP_X,
            idx + span / 2.0,
            group_label,
            transform=trans,
            ha="right",
            va="center",
            fontsize=_GROUP_LABEL_FONTSIZE,
            fontweight="bold",
            clip_on=False,
        )
        idx += span

    for i, inner in enumerate(_flat_inner_labels()):
        ax.text(
            _LEFT_INNER_X,
            i + 0.5,
            inner,
            transform=trans,
            ha="right",
            va="center",
            fontsize=_INNER_LABEL_FONTSIZE,
            clip_on=False,
        )


def _annotation_color(value: float) -> str:
    if not np.isfinite(value):
        return "black"
    return "white" if abs(value) >= 0.45 else "black"


def _save_correlation_matrix_image(corr: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = corr.isna()
    nvar = len(corr.columns)
    fig_side = _CELL_IN * nvar
    fig_w = fig_side + _CBAR_PAD_IN
    fig_h = fig_side
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        corr,
        annot=False,
        cmap=_CORR_CMAP,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.45,
        linecolor="0.92",
        cbar_kws={"label": "Pearson $r$", "shrink": 0.78},
        mask=mask,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )

    for row_idx, row_label in enumerate(corr.index):
        for col_idx, col_label in enumerate(corr.columns):
            if mask.loc[row_label, col_label]:
                continue
            value = float(corr.loc[row_label, col_label])
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=_annotation_color(value),
                fontsize=_ANNOT_FONTSIZE,
            )

    _add_top_axis_labels(ax)
    _add_left_axis_labels(ax)
    fig.subplots_adjust(left=0.22, right=0.86, top=0.72, bottom=0.06)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def main() -> None:
    df = load_model_metrics()
    numeric = df[list(CORR_MATRIX_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.rename(columns=CORR_MATRIX_LABELS)
    display_order = [CORR_MATRIX_LABELS[col] for col in CORR_MATRIX_COLUMNS]
    numeric = numeric[display_order]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:7.4f}" if pd.notna(x) else "    nan")

    print("=== PB-T and PB-A (from model_metrics.csv) ===\n")
    print(numeric.to_string(index=False))
    print("\nNon-null counts per column:")
    print(numeric.count().to_string())
    print("\n--- Pearson correlation matrix (pairwise complete) ---\n")
    corr = numeric.corr(method="pearson", min_periods=3)
    print(corr.to_string())

    img_path = Path("output/images") / "pbt_pba_correlation_matrix.png"
    _save_correlation_matrix_image(corr, img_path)
    print(f"\nCorrelation heatmap saved: {img_path}")


if __name__ == "__main__":
    main()
